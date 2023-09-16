import os.path as osp
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision  # You can load various Pretrained Model from this package
import torchvision.transforms as T  # Transformation functions to manipulate images
import torchvision.transforms.functional as TF


class FoggyDrivingDataset(Dataset):
    """
    Get dir_data_eval(all?) images
    no flip, add scale
    Eval
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
        dir,
        list_path,
        max_iters=None,
        mean=(104.00698793, 116.66876762, 122.67891434),
        split="val",
        img_norm=False,
        scale=None,
    ):
        """
        total classes : 35(-1~33)
        void classes : 16
        valid classes : 19
        img :
        """
        self.dir = dir
        self.list_path = list_path
        self.mean = mean
        self.split = split
        self.img_norm = img_norm
        self.scale = scale

        self.files = []
        self.img_ids = [i_id.strip() for i_id in open(list_path)]  # img file name list
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

        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(19)))

        # initialize
        for name in self.img_ids:
            img_file = osp.join(self.dir, name)
            self.files.append({"img": img_file, "name": name})

        print("Load Foggy Driving Dataset...")
        print("Found %d %s images" % (len(self.files), self.split))

    def __len__(self):
        """
        return length
        usually directly calculated from idx length (csv file, list, dict, etc.)
        """
        return len(self.files)

    def __getitem__(self, index):
        """
        convert file format
        """
        datafiles = self.files[index]

        # open image
        image = Image.open(datafiles["img"]).convert("RGB")
        name = datafiles["name"]

        # resize (new!)
        if self.scale != 1:
            w, h = image.size
            new_size = (int(w * self.scale), int(h * self.scale))
            image = image.resize(new_size, Image.BICUBIC)
        image = np.asarray(image, np.float32)  # (600, 600, 3), dtype=float32

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        if self.img_norm:  # additional
            image = image.astype(float) / 255.0
        image = image.transpose((2, 0, 1))  # (3, 600, 600)

        return image.copy(), np.array(size), name

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
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

    def _apply_transform(self, img, scale=(0.7, 1.3)):
        (W, H) = img.size[:2]  # PIL image has W H C

        if isinstance(scale, tuple):
            scale = random.random() * 0.6 + 0.7

        transform = []
        transform.append(T.Resize((int(H * scale), int(W * scale))))
        transforms = T.Compose(transform)

        return transforms(img)


if __name__ == "__main__":
    dst = FoggyDrivingDataset("/home/vil/jihui/data1", is_transform=True)
    trainloader = DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
