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


class CityscapesDataset(Dataset):
    """
    Get Cityscapes val lindau (cw) images
    no flip, add scale
    no class
    Eval
    """

    def __init__(
        self,
        dir,
        list_path,
        max_iters=None,
        crop_size=(321, 321),
        mean=(128, 128, 128),
        scale=True,
        mirror=True,
        ignore_label=255,
        split="val",
        img_norm=False,
    ):
        self.dir = dir
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.split = split
        self.img_norm = img_norm

        self.files = []
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(
                np.ceil(float(max_iters) / len(self.img_ids))
            )

        # initialize
        for name in self.img_ids:
            img_file = osp.join(self.dir, "leftImg8bit/%s/%s" % (self.split, name))
            self.files.append({"img": img_file, "name": name})

        print("Load Cityscapes Dataset...")
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

        # resize
        w, h = image.size

        image = np.asarray(image, np.float32)  # (600, 600, 3), dtype=float32

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        if self.img_norm:  # additional
            image = image.astype(float) / 255.0
        image = image.transpose((2, 0, 1))  # (3, 600, 600)

        return image.copy(), np.array(size), name


if __name__ == "__main__":
    dst = CityscapesDataset("/home/vil/jihui/data1", is_transform=True)
    trainloader = DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
