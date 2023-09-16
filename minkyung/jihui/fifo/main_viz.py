import os
import os.path as osp

import torch
import torch.nn as nn
from torch.utils import data
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import grad

import numpy as np
import random
import wandb
import csv
from tqdm import tqdm
from PIL import Image
from packaging import version
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from model.refinenetlw import rf_lw101
from model.fogpassfilter import FogPassFilter_conv1, FogPassFilter_res1
from utils.losses import CrossEntropy2d
from dataset.paired_cityscapes import Pairedcityscapes
from dataset.Foggy_Zurich import foggyzurichDataSet
from configs.train_config import get_arguments
from utils.optimisers import get_optimisers, get_lr_schedulers
from pytorch_metric_learning import losses
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.reducers import MeanReducer

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
RESTORE_FROM = "without_pretraining"
RESTORE_FROM_fogpass = "without_pretraining"



def loss_calc(pred, label, gpu):
    label = Variable(label.long()).cuda(gpu)
    criterion = CrossEntropy2d().cuda(gpu)
    return criterion(pred, label)


def gram_matrix(tensor):
    d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram


def setup_optimisers_and_schedulers(args, model):
    optimisers = get_optimisers(
        model=model,
        enc_optim_type="sgd",
        enc_lr=6e-4,
        enc_weight_decay=1e-5,
        enc_momentum=0.9,
        dec_optim_type="sgd",
        dec_lr=6e-3,
        dec_weight_decay=1e-5,
        dec_momentum=0.9,
    )
    schedulers = get_lr_schedulers(
        enc_optim=optimisers[0],
        dec_optim=optimisers[1],
        enc_lr_gamma=0.5,
        dec_lr_gamma=0.5,
        enc_scheduler_type="multistep",
        dec_scheduler_type="multistep",
        epochs_per_stage=(100, 100, 100),
    )
    return optimisers, schedulers


def make_list(x):
    """Returns the given input as a list."""
    if isinstance(x, list):
        return x
    elif isinstance(x, tuple):
        return list(x)
    else:
        return [x]


def main():
    """Create the model and start the training."""

    args = get_arguments()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    now = datetime.now().strftime("%m-%d-%H-%M")
    run_name = f"{args.file_name}-{now}"

    # wandb.login(key="a173aa08653488eb94627696bda5d3a5cc79443f")
    # wandb.init(project="FIFO", name=f"{run_name}")
    # wandb.config.update(args)
    sf_file = open("fog_factor_sf.csv", "a")
    cw_file = open("fog_factor_cw.csv", "a")
    rf_file = open("fog_factor_rf.csv", "a")

    sf_writer = csv.writer(sf_file)
    cw_writer = csv.writer(cw_file)
    rf_writer = csv.writer(rf_file)
    # writer = SummaryWriter(f"runs/{run_name}")

    w, h = map(int, args.input_size.split(","))
    input_size = (w, h)

    w_r, h_r = map(int, args.input_size_rf.split(","))
    input_size_rf = (w_r, h_r)

    cudnn.enabled = True
    gpu = args.gpu

    if args.restore_from == RESTORE_FROM:
        start_iter = 0
        model = rf_lw101(num_classes=args.num_classes)

    else:
        restore = torch.load(args.restore_from)
        model = rf_lw101(num_classes=args.num_classes)

        model.load_state_dict(restore["state_dict"])
        start_iter = 0

    model.train()
    model.cuda(args.gpu)

    lr_fpf1 = 1e-3
    lr_fpf2 = 1e-3

    if args.modeltrain == "train":
        lr_fpf1 = 5e-4

    FogPassFilter1 = FogPassFilter_conv1(2080)
    FogPassFilter1_optimizer = torch.optim.Adamax(
        [p for p in FogPassFilter1.parameters() if p.requires_grad == True], lr=lr_fpf1
    )
    FogPassFilter1.cuda(args.gpu)
    FogPassFilter2 = FogPassFilter_res1(32896)
    FogPassFilter2_optimizer = torch.optim.Adamax(
        [p for p in FogPassFilter2.parameters() if p.requires_grad == True], lr=lr_fpf2
    )
    FogPassFilter2.cuda(args.gpu)

    if args.restore_from_fogpass != RESTORE_FROM_fogpass:
        restore = torch.load(args.restore_from_fogpass)
        FogPassFilter1.load_state_dict(restore["fogpass1_state_dict"])
        FogPassFilter2.load_state_dict(restore["fogpass2_state_dict"])

    fogpassfilter_loss = losses.ContrastiveLoss(
        pos_margin=0.1,
        neg_margin=0.1,
        distance=CosineSimilarity(),
        reducer=MeanReducer(),
    )

    cudnn.benchmark = True

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    cwsf_pair_loader = data.DataLoader(
        Pairedcityscapes(
            args.data_dir,
            args.data_dir_cwsf,
            args.data_list,
            args.data_list_cwsf,
            max_iters=args.num_steps * args.iter_size * args.batch_size,
            mean=IMG_MEAN,
            set=args.set,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    rf_loader = data.DataLoader(
        foggyzurichDataSet(
            args.data_dir_rf,
            args.data_list_rf,
            max_iters=args.num_steps * args.iter_size * args.batch_size,
            mean=IMG_MEAN,
            set=args.set,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    cwsf_pair_loader_fogpass = data.DataLoader(
        Pairedcityscapes(
            args.data_dir,
            args.data_dir_cwsf,
            args.data_list,
            args.data_list_cwsf,
            max_iters=args.num_steps * args.iter_size * args.batch_size,
            mean=IMG_MEAN,
            set=args.set,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    rf_loader_fogpass = data.DataLoader(
        foggyzurichDataSet(
            args.data_dir_rf,
            args.data_list_rf,
            max_iters=args.num_steps * args.iter_size * args.batch_size,
            mean=IMG_MEAN,
            set=args.set,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    cwsf_pair_loader_iter_fogpass = enumerate(cwsf_pair_loader_fogpass)
    rf_loader_iter_fogpass = enumerate(rf_loader_fogpass)

    optimisers, schedulers = setup_optimisers_and_schedulers(args, model=model)
    opts = make_list(optimisers)


    for i_iter in tqdm(range(start_iter, args.num_steps)):

        for opt in opts:
            opt.zero_grad()

        for sub_i in range(args.iter_size):
            # train fog-pass filtering module
            # freeze the parameters of segmentation network

            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            for param in FogPassFilter1.parameters():
                param.requires_grad = False
            for param in FogPassFilter2.parameters():
                param.requires_grad = False

            _, batch = cwsf_pair_loader_iter_fogpass.__next__()
            sf_image, cw_image, label, size, sf_name, cw_name = batch

            _, batch_rf = rf_loader_iter_fogpass.__next__()
            rf_img, rf_size, rf_name = batch_rf
            img_rf = Variable(rf_img).cuda(args.gpu)
            (
                feature_rf0,
                feature_rf1,
                feature_rf2,
                feature_rf3,
                feature_rf4,
                feature_rf5,
            ) = model(img_rf)

            images = Variable(sf_image).cuda(args.gpu)
            (
                feature_sf0,
                feature_sf1,
                feature_sf2,
                feature_sf3,
                feature_sf4,
                feature_sf5,
            ) = model(images)

            images_cw = Variable(cw_image).cuda(args.gpu)
            (
                feature_cw0,
                feature_cw1,
                feature_cw2,
                feature_cw3,
                feature_cw4,
                feature_cw5,
            ) = model(images_cw)

            fsm_weights = {"layer0": 0.5, "layer1": 0.5}
            sf_features = {"layer0": feature_sf0, "layer1": feature_sf1}
            cw_features = {"layer0": feature_cw0, "layer1": feature_cw1}
            rf_features = {"layer0": feature_rf0, "layer1": feature_rf1}

            total_fpf_loss = 0

            for idx, layer in enumerate(fsm_weights):
                cw_feature = cw_features[layer]
                sf_feature = sf_features[layer]
                rf_feature = rf_features[layer]
                fog_pass_filter_loss = 0

                if idx == 0:
                    fogpassfilter = FogPassFilter1
                    fogpassfilter_optimizer = FogPassFilter1_optimizer
                elif idx == 1:
                    fogpassfilter = FogPassFilter2
                    fogpassfilter_optimizer = FogPassFilter2_optimizer

                fogpassfilter.eval()
                fogpassfilter_optimizer.zero_grad()

                sf_gram = [0] * args.batch_size
                cw_gram = [0] * args.batch_size
                rf_gram = [0] * args.batch_size
                vector_sf_gram = [0] * args.batch_size
                vector_cw_gram = [0] * args.batch_size
                vector_rf_gram = [0] * args.batch_size
                fog_factor_sf = [0] * args.batch_size
                fog_factor_cw = [0] * args.batch_size
                fog_factor_rf = [0] * args.batch_size

                for batch_idx in range(args.batch_size):
                    sf_gram[batch_idx] = gram_matrix(sf_feature[batch_idx])
                    cw_gram[batch_idx] = gram_matrix(cw_feature[batch_idx])
                    rf_gram[batch_idx] = gram_matrix(rf_feature[batch_idx])

                    vector_sf_gram[batch_idx] = Variable(
                        sf_gram[batch_idx][
                            torch.triu(
                                torch.ones(
                                    sf_gram[batch_idx].size()[0],
                                    sf_gram[batch_idx].size()[1],
                                )
                            )
                            == 1
                        ],
                        requires_grad=True,
                    )
                    vector_cw_gram[batch_idx] = Variable(
                        cw_gram[batch_idx][
                            torch.triu(
                                torch.ones(
                                    cw_gram[batch_idx].size()[0],
                                    cw_gram[batch_idx].size()[1],
                                )
                            )
                            == 1
                        ],
                        requires_grad=True,
                    )
                    vector_rf_gram[batch_idx] = Variable(
                        rf_gram[batch_idx][
                            torch.triu(
                                torch.ones(
                                    rf_gram[batch_idx].size()[0],
                                    rf_gram[batch_idx].size()[1],
                                )
                            )
                            == 1
                        ],
                        requires_grad=True,
                    )

                    fog_factor_sf[batch_idx] = fogpassfilter(vector_sf_gram[batch_idx])
                    fog_factor_cw[batch_idx] = fogpassfilter(vector_cw_gram[batch_idx])
                    fog_factor_rf[batch_idx] = fogpassfilter(vector_rf_gram[batch_idx])
                    sf_writer.writerow(fog_factor_sf[batch_idx].detach().tolist())
                    cw_writer.writerow(fog_factor_cw[batch_idx].detach().tolist())
                    rf_writer.writerow(fog_factor_rf[batch_idx].detach().tolist())
                
                fog_factor_embeddings = torch.cat(
                    (
                        torch.unsqueeze(fog_factor_sf[0], 0),
                        torch.unsqueeze(fog_factor_cw[0], 0),
                        torch.unsqueeze(fog_factor_rf[0], 0),
                        torch.unsqueeze(fog_factor_sf[1], 0),
                        torch.unsqueeze(fog_factor_cw[1], 0),
                        torch.unsqueeze(fog_factor_rf[1], 0),
                        torch.unsqueeze(fog_factor_sf[2], 0),
                        torch.unsqueeze(fog_factor_cw[2], 0),
                        torch.unsqueeze(fog_factor_rf[2], 0),
                        torch.unsqueeze(fog_factor_sf[3], 0),
                        torch.unsqueeze(fog_factor_cw[3], 0),
                        torch.unsqueeze(fog_factor_rf[3], 0),
                    ),
                    0,
                )

                fog_factor_embeddings_norm = torch.norm(
                    fog_factor_embeddings, p=2, dim=1
                ).detach()
                size_fog_factor = fog_factor_embeddings.size()
                fog_factor_embeddings = fog_factor_embeddings.div(
                    fog_factor_embeddings_norm.expand(size_fog_factor[1], 12).t()
                )
                fog_factor_labels = torch.LongTensor(
                    [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
                )
                fog_pass_filter_loss = fogpassfilter_loss(
                    fog_factor_embeddings, fog_factor_labels
                )

                total_fpf_loss += fog_pass_filter_loss

            total_fpf_loss.backward(retain_graph=False)

            FogPassFilter1_optimizer.step()
            FogPassFilter2_optimizer.step()
            
            # exit()

    sf_file.close()
    cw_file.close()
    rf_file.close()


if __name__ == "__main__":
    main()
