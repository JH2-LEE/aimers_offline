import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as T  # Transformation functions to manipulate images
import torchvision.transforms.functional as TF
import torch.nn.functional as F  # various activation functions for model

import torch.optim as optim  # various optimization functions for model
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.autograd import grad

# MULTI-GPU
from parallel import DataParallelModel, DataParallelCriterion
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp

# from torch.utils import data

import numpy as np
import random
from tqdm import tqdm
import os
import os.path as osp
import matplotlib.pyplot as plt
# import wandb
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from model.refinenetlw import rf_lw101
from model.fogpassfilter import FogPassFilter_conv1, FogPassFilter_res1
from dataset.paired_cityscapes import PairedCityscapesDataset
from dataset.foggy_zurich import FoggyZurichDataset
from configs.train_config import get_arguments
from utils.optimisers import get_optimisers, get_lr_schedulers

from utils.losses import CrossEntropy2d
from pytorch_metric_learning import losses
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.reducers import MeanReducer

RESTORE_FROM = "without_pretraining"
RESTORE_FROM_FOGPASS = "without_pretraining"


def loss_calc(pred, label, gpu):
    label = Variable(label.long()).cuda(gpu)
    criterion = CrossEntropy2d().cuda(gpu)
    return criterion(pred, label)


def gram_matrix(tensor):
    d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)  # flatten; (d, h, w) -> (d, h*w)
    gram = torch.mm(tensor, tensor.t())  # matmul; (d, h*w)*(h*w, d) -> (d, d)
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


# get arguments
args = get_arguments()
IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)  # BGR

# Visualization Setup
print(f"Random Seed: {args.random_seed}")
random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed(args.random_seed)
torch.cuda.manual_seed_all(args.random_seed)  # if use multi-GPU
cudnn.benchmark = False
cudnn.deterministic = True
cudnn.enabled = True
np.random.seed(args.random_seed)  # for numpy-based backend, scikit-learn

# logging
now = datetime.now().strftime("%m-%d-%H-%M")
run_name = f"{args.file_name}-{now}"

# wandb.login(key="a173aa08653488eb94627696bda5d3a5cc79443f")
# wandb.init(project="FIFO", name=f"{run_name}")
# wandb.config.update(args)

writer = SummaryWriter(f"runs/{run_name}")


w, h = map(int, args.input_size.split(","))
input_size = (w, h)
w_r, h_r = map(int, args.input_size_rf.split(","))
input_size_rf = (w_r, h_r)
print(input_size, input_size_rf)

# Dataloader
cwsf_pair_loader = DataLoader(
    PairedCityscapesDataset(
        src_dir=args.data_dir,
        tgt_dir=args.data_dir_cwsf,
        src_list_path=args.data_list,
        tgt_list_path=args.data_list_cwsf,
        max_iters=args.num_steps * args.iter_size * args.batch_size,
        mean=IMG_MEAN,
        split=args.split,
        img_norm=False,
    ),
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True,
)
rf_loader = DataLoader(
    FoggyZurichDataset(
        dir=args.data_dir_rf,
        list_path=args.data_list_rf,
        max_iters=args.num_steps * args.iter_size * args.batch_size,
        mean=IMG_MEAN,
        split=args.split,
        img_norm=False,
    ),
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True,
)
cwsf_pair_loader_fogpass = DataLoader(
    PairedCityscapesDataset(
        src_dir=args.data_dir,
        tgt_dir=args.data_dir_cwsf,
        src_list_path=args.data_list,
        tgt_list_path=args.data_list_cwsf,
        max_iters=args.num_steps * args.iter_size * args.batch_size,
        mean=IMG_MEAN,
        split=args.split,
        img_norm=False,
    ),
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True,
)
rf_loader_fogpass = DataLoader(
    FoggyZurichDataset(
        dir=args.data_dir_rf,
        list_path=args.data_list_rf,
        max_iters=args.num_steps * args.iter_size * args.batch_size,
        mean=IMG_MEAN,
        split=args.split,
        img_norm=False,
    ),
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True,
)

rf_loader_iter = enumerate(rf_loader)
cwsf_pair_loader_iter = enumerate(cwsf_pair_loader)
cwsf_pair_loader_iter_fogpass = enumerate(cwsf_pair_loader_fogpass)
rf_loader_iter_fogpass = enumerate(rf_loader_fogpass)

# Build the model
start_iter = 0
## segmentation
model = rf_lw101(num_classes=args.num_classes).cuda(args.gpu)

# print(model)
if args.restore_from != RESTORE_FROM:  # restore pretraining ckpt
    restore = torch.load(args.restore_from)  # dict
    # print("restore")
    # print(len(restore))
    model.load_state_dict(restore["state_dict"])
    start_iter = 0

model.train()  # ?

## fogpassfilter
FogPassFilter1 = FogPassFilter_conv1(2080).cuda(args.gpu)
FogPassFilter2 = FogPassFilter_res1(32896).cuda(args.gpu)

if args.restore_from_fogpass != RESTORE_FROM_FOGPASS:  # restore pretraining ckpt
    restore = torch.load(args.restore_from_fogpass)  # dict
    FogPassFilter1.load_state_dict(restore["fogpass1_state_dict"])
    FogPassFilter2.load_state_dict(restore["fogpass2_state_dict"])

# Hyperparameter
lr_fpf1 = 1e-3
lr_fpf2 = 1e-3
if args.modeltrain == "train":
    lr_fpf1 = 5e-4

# Optimizer and Scheduler
## segmentation model optimizer
optimisers, schedulers = setup_optimisers_and_schedulers(args, model=model)
opts = make_list(optimisers)
# print(opts)
# fogpass filter optimizer (only when it is not freezing)
FogPassFilter1_Optimizer = torch.optim.Adamax(
    [p for p in FogPassFilter1.parameters() if p.requires_grad == True], lr=lr_fpf1
)
FogPassFilter2_Optimizer = torch.optim.Adamax(
    [p for p in FogPassFilter2.parameters() if p.requires_grad == True], lr=lr_fpf2
)
# Loss
fogpassfilter_loss = losses.ContrastiveLoss(  # to split negative pairs well
    pos_margin=0.1,  # default is 1
    neg_margin=0.1,  # default is 0
    distance=CosineSimilarity(),
    reducer=MeanReducer(),
)
kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
m = nn.Softmax(dim=1)
log_m = nn.LogSoftmax(dim=1)
# Loss for MULTI-GPU

# Optimizing and Model Parameters
for i_iter in tqdm(range(start_iter, args.num_steps)):
    loss_seg_cw_value = 0  # segmentation loss(only for cityscapes)
    loss_seg_sf_value = 0  # segmentation loss(only for cityscapes)
    loss_fsm_value = 0  # foggy style matching loss
    loss_con_value = 0  # consistency loss

    for opt in opts:  # model optimizer : zero grad
        opt.zero_grad()
    # print("i_iter", i_iter)
    for sub_i in range(args.iter_size):
        # train fog pass filtering module(pre-training)
        # freeze segmentation module
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        for param in FogPassFilter1.parameters():
            param.requires_grad = True
        for param in FogPassFilter2.parameters():
            param.requires_grad = True

        # load images in the batch from each dataloader
        _, batch_cwsf = cwsf_pair_loader_iter_fogpass.__next__()
        sf_image, cw_image, label, size, sf_name, cw_name = batch_cwsf
        interp = nn.Upsample(size=(size[0][0], size[0][1]), mode="bilinear")  # 600, 600
        _, batch_rf = rf_loader_iter_fogpass.__next__()
        rf_image, rf_size, rf_name = batch_rf

        # load features from each layer
        # use variable func
        image_cw = Variable(cw_image).cuda(args.gpu)
        (
            feature_cw0,
            feature_cw1,
            feature_cw2,
            feature_cw3,
            feature_cw4,
            feature_cw5,
        ) = model(image_cw)
        image_sf = Variable(sf_image).cuda(args.gpu)
        (
            feature_sf0,
            feature_sf1,
            feature_sf2,
            feature_sf3,
            feature_sf4,
            feature_sf5,
        ) = model(image_sf)
        image_rf = Variable(rf_image).cuda(args.gpu)
        (
            feature_rf0,
            feature_rf1,
            feature_rf2,
            feature_rf3,
            feature_rf4,
            feature_rf5,
        ) = model(image_rf)

        # select l features in each layer with dict format
        fsm_weights = {"layer0": 0.5, "layer1": 0.5}  # 2 layers
        cw_features = {"layer0": feature_cw0, "layer1": feature_cw1}
        sf_features = {"layer0": feature_sf0, "layer1": feature_sf1}
        rf_features = {"layer0": feature_rf0, "layer1": feature_rf1}

        total_fpf_loss = 0

        for idx, layer in enumerate(fsm_weights):  # for each layer
            # print(f"idx: {idx}, layer: {layer}")
            cw_feature = cw_features[layer]
            # print("cw feature size:", cw_feature.size())
            sf_feature = sf_features[layer]
            rf_feature = rf_features[layer]
            fog_pass_filter_loss = 0

            # diff option
            if idx == 0:
                fogpassfilter = FogPassFilter1
                fogpassfilter_optimizer = FogPassFilter1_Optimizer
            elif idx == 1:
                fogpassfilter = FogPassFilter2
                fogpassfilter_optimizer = FogPassFilter2_Optimizer

            # fpf train option
            fogpassfilter.train()
            fogpassfilter_optimizer.zero_grad()

            # initialize list
            cw_gram = [0] * args.batch_size
            sf_gram = [0] * args.batch_size
            rf_gram = [0] * args.batch_size
            vector_cw_gram = [0] * args.batch_size
            vector_sf_gram = [0] * args.batch_size
            vector_rf_gram = [0] * args.batch_size
            fog_factor_cw = [0] * args.batch_size
            fog_factor_sf = [0] * args.batch_size
            fog_factor_rf = [0] * args.batch_size

            for batch_idx in range(args.batch_size):
                # print("gram matrix size:", gram_matrix(cw_feature[batch_idx]).size())
                cw_gram[batch_idx] = gram_matrix(cw_feature[batch_idx])
                sf_gram[batch_idx] = gram_matrix(sf_feature[batch_idx])
                rf_gram[batch_idx] = gram_matrix(rf_feature[batch_idx])
                # print("cw_gram:", cw_gram[batch_idx])
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
                # fog factor
                fog_factor_cw[batch_idx] = fogpassfilter(
                    vector_cw_gram[batch_idx]
                )  # (64)
                fog_factor_sf[batch_idx] = fogpassfilter(
                    vector_sf_gram[batch_idx]
                )  # (64)
                fog_factor_rf[batch_idx] = fogpassfilter(
                    vector_rf_gram[batch_idx]
                )  # (64)
                # print("fog factor:", fog_factor_cw[batch_idx])
                # print("fog factor size:", fog_factor_cw[batch_idx].size())
            fog_factor_embeddings = torch.cat(
                (  # batch 섞어보기
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
            )  # (12,64)
            # normalize fog factor embeddings
            fog_factor_embeddings_norm = torch.norm(
                fog_factor_embeddings, p=2, dim=1
            ).detach()
            size_fog_factor = fog_factor_embeddings.size()  # (12,64)
            fog_factor_embeddings = fog_factor_embeddings.div(
                fog_factor_embeddings_norm.expand(size_fog_factor[1], 12).t()
            )  # (64, 12) -> (12, 64); expand 함수에서 12가 뒤에 와야 함
            # fog_factor_labels = torch.LongTensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
            fog_factor_labels = torch.LongTensor(
                [[0, 1], [0, 0], [1, 2], [0, 1], [0, 0], [1, 2], [0, 1], [0, 0], [1, 2]]
            )
            # fog_pass_filter_loss = fogpassfilter_loss(
            #     fog_factor_embeddings, fog_factor_labels
            # )
            fog_pass_filter_loss = fogpassfilter_loss(
                fog_factor_embeddings, fog_factor_labels
            )
            total_fpf_loss += fog_pass_filter_loss  # just summation; don't need weights
            # wandb.log({f"layer{idx}/fpf loss": fog_pass_filter_loss}, step=i_iter)
            writer.add_scalar(f"layer{idx}/fpf loss", fog_pass_filter_loss, i_iter)

        # wandb.log({"total fpf loss": total_fpf_loss}, step=i_iter)  # 이게 맞는듯?
        total_fpf_loss.backward(retain_graph=False)
        writer.add_scalar("total fpf loss", total_fpf_loss, i_iter)
        #######################
        # exit()
        #######################

        if args.modeltrain == "train":
            # train segmentation module
            # freeze fog pas filtering module
            model.train()
            for param in model.parameters():
                param.requires_grad = True
            for param in FogPassFilter1.parameters():
                param.requires_grad = False
            for param in FogPassFilter2.parameters():
                param.requires_grad = False

            # load image from dataloader
            _, batch = cwsf_pair_loader_iter.__next__()
            sf_image, cw_image, label, size, sf_name, cw_name = batch

            # interpolation for prediction of the last layer in semseg
            interp = nn.Upsample(size=(size[0][0], size[0][1]), mode="bilinear")

            if i_iter % 3 == 0:  # CW-SF
                image_cw = Variable(cw_image).cuda(args.gpu)
                (
                    feature_cw0,
                    feature_cw1,
                    feature_cw2,
                    feature_cw3,
                    feature_cw4,
                    feature_cw5,
                ) = model(image_cw)
                image_sf = Variable(sf_image).cuda(args.gpu)
                (
                    feature_sf0,
                    feature_sf1,
                    feature_sf2,
                    feature_sf3,
                    feature_sf4,
                    feature_sf5,
                ) = model(image_sf)
                pred_cw5 = interp(feature_cw5)
                pred_sf5 = interp(feature_sf5)
                # Segmentation Loss
                loss_seg_cw = loss_calc(pred_cw5, label, args.gpu)
                loss_seg_sf = loss_calc(pred_sf5, label, args.gpu)
                # Fog Style Matching Loss -> Contd'
                fsm_weights = {"layer0": 0.5, "layer1": 0.5}
                sf_features = {"layer0": feature_sf0, "layer1": feature_sf1}
                # Prediction Consistency Loss
                feature_sf5_logsoftmax = log_m(feature_sf5)
                feature_cw5_softmax = m(feature_cw5)
                loss_con = kl_loss(feature_sf5_logsoftmax, feature_cw5_softmax)
            if i_iter % 3 == 1:  # SF-RF
                _, batch_rf = rf_loader_iter.__next__()
                rf_img, rf_size, rf_name = batch_rf
                image_sf = Variable(sf_image).cuda(args.gpu)
                (
                    feature_sf0,
                    feature_sf1,
                    feature_sf2,
                    feature_sf3,
                    feature_sf4,
                    feature_sf5,
                ) = model(image_sf)
                image_rf = Variable(rf_image).cuda(args.gpu)
                (
                    feature_rf0,
                    feature_rf1,
                    feature_rf2,
                    feature_rf3,
                    feature_rf4,
                    feature_rf5,
                ) = model(image_rf)
                pred_sf5 = interp(feature_sf5)
                # Segmentation Loss
                loss_seg_sf = loss_calc(pred_sf5, label, args.gpu)
                loss_seg_cw = 0
                # Fog Style Matching Loss -> Contd'
                fsm_weights = {"layer0": 0.5, "layer1": 0.5}
                sf_features = {"layer0": feature_sf0, "layer1": feature_sf1}
                rf_features = {"layer0": feature_rf0, "layer1": feature_rf1}
                # Prediction Consistency Loss -> No
                loss_con = 0
            if i_iter % 3 == 2:  # CW-RF
                _, batch_rf = rf_loader_iter.__next__()
                rf_img, rf_size, rf_name = batch_rf
                image_cw = Variable(cw_image).cuda(args.gpu)
                (
                    feature_cw0,
                    feature_cw1,
                    feature_cw2,
                    feature_cw3,
                    feature_cw4,
                    feature_cw5,
                ) = model(image_cw)
                image_rf = Variable(rf_image).cuda(args.gpu)
                (
                    feature_rf0,
                    feature_rf1,
                    feature_rf2,
                    feature_rf3,
                    feature_rf4,
                    feature_rf5,
                ) = model(image_rf)
                pred_cw5 = interp(feature_cw5)
                # Segmentation Loss
                loss_seg_sf = 0
                loss_seg_cw = loss_calc(pred_cw5, label, args.gpu)
                # Fog Style Matching Loss -> Contd'
                fsm_weights = {"layer0": 0.5, "layer1": 0.5}
                cw_features = {"layer0": feature_cw0, "layer1": feature_cw1}
                rf_features = {"layer0": feature_rf0, "layer1": feature_rf1}
                # Prediction Consistency Loss -> No
                loss_con = 0

            # Initialize FPF parameters -> 필요한지 프린트해보기
            loss_fsm = 0
            fog_pass_filter_loss = 0

            for idx, layer in enumerate(fsm_weights):
                # print(layer)

                # fog pass filter loss between different fog conditions a and b
                # a, b 순서 바뀌어도 상관없는지 보기
                if i_iter % 3 == 0:
                    a_feature = cw_features[layer]
                    b_feature = sf_features[layer]
                if i_iter % 3 == 1:
                    a_feature = rf_features[layer]
                    b_feature = sf_features[layer]
                if i_iter % 3 == 2:
                    a_feature = rf_features[layer]
                    b_feature = cw_features[layer]
                layer_fsm_loss = 0
                fog_pass_filter_loss = 0
                # feature size; layer0: (b,64,300,300), layer1: (b,256,150,150)
                na, da, ha, wa = a_feature.size()
                nb, db, hb, wb = b_feature.size()

                # diff option
                if idx == 0:
                    fogpassfilter = FogPassFilter1
                    fogpassfilter_optimizer = FogPassFilter1_Optimizer
                elif idx == 1:
                    fogpassfilter = FogPassFilter2
                    fogpassfilter_optimizer = FogPassFilter2_Optimizer

                # fpf option -> freezing
                fogpassfilter.eval()  # zero grad는 필요 없음

                #########

                #########
                for batch_idx in range(4):
                    a_gram = gram_matrix(
                        a_feature[batch_idx]
                    )  # layer 0: (64,64), layer 1: (256,256)
                    b_gram = gram_matrix(
                        b_feature[batch_idx]
                    )  # layer 0: (64,64), layer 1: (256,256)

                    if i_iter % 3 == 1 or i_iter % 3 == 2:
                        a_gram = a_gram * (hb * wb) / (ha * wa)  # ha=hb, wa=wb 왜 쓴거지?

                    vector_b_gram = b_gram[
                        torch.triu(
                            torch.ones(b_gram.size()[0], b_gram.size()[1])
                        ).requires_grad_()
                        == 1
                    ].requires_grad_()  # layer0: (2080), layer1: (32896)
                    vector_a_gram = a_gram[
                        torch.triu(
                            torch.ones(a_gram.size()[0], a_gram.size()[1])
                        ).requires_grad_()
                        == 1
                    ].requires_grad_()  # layer0: (2080), layer1: (32896)
                    fog_factor_a = fogpassfilter(vector_a_gram)  # (64)
                    fog_factor_b = fogpassfilter(vector_b_gram)  # (64)

                    half = int(fog_factor_b.shape[0] / 2)  # 32
                    layer_fsm_loss += (
                        fsm_weights[layer]
                        * torch.mean(
                            (fog_factor_b / (hb * wb) - fog_factor_a / (ha * wa)) ** 2
                        )
                        / half
                        / b_feature.size(0)  # 4
                    )

                loss_fsm += layer_fsm_loss / 4.0

            loss = (
                loss_seg_sf
                + loss_seg_cw
                + args.lambda_fsm * loss_fsm
                + args.lambda_con * loss_con
            )
            loss = loss / args.iter_size
            loss.backward()

            if loss_seg_cw != 0:
                loss_seg_cw_value += loss_seg_cw.data.cpu().numpy() / args.iter_size
            if loss_seg_sf != 0:
                loss_seg_sf_value += loss_seg_sf.data.cpu().numpy() / args.iter_size
            if loss_fsm != 0:
                loss_fsm_value += loss_fsm.data.cpu().numpy() / args.iter_size
            if loss_con != 0:
                loss_con_value += loss_con.data.cpu().numpy() / args.iter_size

            # wandb.log({"fsm loss": args.lambda_fsm * loss_fsm_value}, step=i_iter)
            # wandb.log({"SF_loss_seg": loss_seg_sf_value}, step=i_iter)
            # wandb.log({"CW_loss_seg": loss_seg_cw_value}, step=i_iter)
            # wandb.log(
            #     {"consistency loss": args.lambda_con * loss_con_value}, step=i_iter
            # )
            # wandb.log({"total_loss": loss}, step=i_iter)

            writer.add_scalar("fsm loss", args.lambda_fsm * loss_fsm_value, i_iter)
            writer.add_scalar("SF_loss_seg", loss_seg_sf_value, i_iter)
            writer.add_scalar("CW_loss_seg", loss_seg_cw_value, i_iter)
            writer.add_scalar(
                "consistency loss", args.lambda_con * loss_con_value, i_iter
            )
            writer.add_scalar("total_loss", loss, i_iter)
            # For debugging
            # exit()

            for opt in opts:
                opt.step()

        FogPassFilter1_Optimizer.step()
        FogPassFilter2_Optimizer.step()

    if i_iter < 20000:
        if args.modeltrain != "train":  # train fpf
            save_pred_every = 5000
        if args.modeltrain == "train":  # train model
            # save_pred_every = 2000
            save_pred_every = 5000
    else:
        save_pred_every = args.save_pred_every

    # if i_iter >= args.num_steps_stop - 1:
    #     print("save model ..")
    #     torch.save(
    #         model.state_dict(),
    #         osp.join(
    #             args.snapshot_dir,
    #             args.file_name + str(args.num_steps_stop) + ".pth",
    #         ),
    #     )
    #     break
    if args.modeltrain != "train":
        if i_iter == 5000:
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "fogpass1_state_dict": FogPassFilter1.state_dict(),
                    "fogpass2_state_dict": FogPassFilter2.state_dict(),
                    "train_iter": i_iter,
                    "args": args,
                },
                osp.join(args.snapshot_dir, run_name)
                + "_fogpassfilter_"
                + str(i_iter)
                + ".pth",
            )

    if (i_iter + 1) % save_pred_every == 0 and (i_iter + 1) != 0:
        print("taking snapshot ...")
        save_dir = osp.join(f"./result/FIFO_model", args.file_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(
            {
                "state_dict": model.state_dict(),
                "fogpass1_state_dict": FogPassFilter1.state_dict(),
                "fogpass2_state_dict": FogPassFilter2.state_dict(),
                "train_iter": i_iter,
                "args": args,
            },
            osp.join(args.snapshot_dir, run_name) + "_FIFO" + str(i_iter) + ".pth",
        )
