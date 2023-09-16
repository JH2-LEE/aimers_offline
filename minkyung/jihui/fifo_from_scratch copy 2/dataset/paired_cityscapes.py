import os.path as osp
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T  # Transformation functions to manipulate images
import torchvision.transforms.functional as TF


class PairedCityscapesDataset(Dataset):
    """
    Get synthetic fog(sf) and clear weather(cw) images
    Train
    """

    colors = [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    label_colours = dict(zip(range(19), colors))

    def __init__(
        self,
        src_dir,
        tgt_dir,
        src_list_path,
        tgt_list_path,
        max_iters=None,
        mean=(128, 128, 128),
        split="val",
        img_norm=False,
    ):
        """
        total classes : 35(-1~33)
        void classes : 16
        valid classes : 19
        src img : sf (left8bit foggyDBF)
        tgt img : cw (left8bit)
        """
        # open file
        self.src_dir = src_dir
        self.tgt_dir = tgt_dir
        self.src_list_path = src_list_path
        self.tgt_list_path = tgt_list_path
        self.mean = mean
        self.split = split  # train val test
        self.img_norm = img_norm

        self.files = []
        self.img_ids = [i_id.strip() for i_id in open(src_list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(
                np.ceil(float(max_iters) / len(self.img_ids))
            )

        # classes
        self.void_classes = [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            9,
            10,
            14,
            15,
            16,
            18,
            29,
            30,
            -1,
        ]  # 16
        self.valid_classes = [
            7,
            8,
            11,
            12,
            13,
            17,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            31,
            32,
            33,
        ]  # 19
        self.class_names = [
            "unlabelled",
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]  # 20 w/ unlabeled

        self.ignore_idx = 255
        self.class_map = dict(zip(self.valid_classes, range(19)))

        # initialize
        for name in self.img_ids:
            tgt_img_file = osp.join(
                self.tgt_dir, "leftImg8bit/%s/%s" % (self.split, name[:-21] + ".png")
            )
            # src_img_file = osp.join(self.src_dir, "./leftImg8bit_foggyDBF/%s/%s" % (self.split, name))
            src_img_file = osp.join(
                self.src_dir,
                "Cityscapes/leftImg8bit_foggyDBF/%s/%s" % (self.split, name),
            )
            label_file = osp.join(
                self.src_dir,
                "Cityscapes/gtFine/%s/%s"
                % (self.split, name[:-32] + "gtFine_labelIds.png"),
            )
            self.files.append(
                {
                    "src_img": src_img_file,
                    "tgt_img": tgt_img_file,
                    "label": label_file,
                    "tgt_name": name[:-21] + ".png",
                    "src_name": name,
                }
            )
        print("Load Paired Cityscapes Dataset...")
        print("Found %d %s images" % (len(self.files), self.split))

    def __len__(self):
        """
        return length
        usually directly calculated from idx length (csv file, list, dict, etc.)
        """
        return len(self.files)

    def __getitem__(self, idx):
        """
        convert file format
        """
        datafiles = self.files[idx]
        # open image
        src_image = Image.open(datafiles["src_img"]).convert("RGB")
        tgt_image = Image.open(datafiles["tgt_img"]).convert("RGB")
        label = Image.open(datafiles["label"])
        src_name = datafiles["src_name"]
        tgt_name = datafiles["tgt_name"]

        # resize
        w, h = src_image.size
        src_image, tgt_image, label = self._apply_transform(
            src_image, tgt_image, label, scale=0.8
        )

        # crop
        crop_size = min(600, min(src_image.size[:2]))
        i, j, h, w = T.RandomCrop.get_params(
            src_image, output_size=(crop_size, crop_size)
        )
        src_image = TF.crop(src_image, i, j, h, w)
        tgt_image = TF.crop(tgt_image, i, j, h, w)
        label = TF.crop(label, i, j, h, w)

        # flip
        if random.random() > 0.5:
            src_image = TF.hflip(src_image)
            tgt_image = TF.hflip(tgt_image)
            label = TF.hflip(label)

        src_image = np.asarray(src_image, np.float32)  # (600, 600, 3), dtype=float32
        tgt_image = np.asarray(tgt_image, np.float32)  # (600, 600, 3), dtype=float32
        label = self.encode_segmap(np.array(label, dtype=np.float32))

        lbl = label.astype(float)
        label = lbl.astype(int)

        size = src_image.shape
        src_image = src_image[:, :, ::-1]  # change to BGR
        src_image -= self.mean
        tgt_image = tgt_image[:, :, ::-1]  # change to BGR
        tgt_image -= self.mean
        if self.img_norm:
            src_image = src_image.astype(float) / 255.0
            tgt_image = tgt_image.astype(float) / 255.0
        src_image = src_image.transpose((2, 0, 1))  # (3, 600, 600)
        tgt_image = tgt_image.transpose((2, 0, 1))  # (3, 600, 600)

        return (
            src_image.copy(),
            tgt_image.copy(),
            label.copy(),
            np.array(size),
            src_name,
            tgt_name,
        )

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_idx
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def decode_segmap(self, temp):
        # masking with colors (not used right now)
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()

        for l in range(0, len(self.label_colours)):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        # normalizing
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def _apply_transform(self, img1, img2, lbl, scale=(0.7, 1.3)):
        (W, H) = img1.size[:2]  # PIL image has W H C

        if isinstance(scale, tuple):  # why do we need?
            scale = random.random() * 0.6 + 0.7

        transform = []
        transform.append(T.Resize((int(H * scale), int(W * scale))))
        transforms = T.Compose(transform)

        return transforms(img1), transforms(img2), transforms(lbl)
