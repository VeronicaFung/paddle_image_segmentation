from basic_transforms import *


class TrainAugmentation():
    def __init__(self, image_size, mean_val=0, std_val=1.0):
        # TODO: add self.augment, which contains
        # random scale, pad, random crop, random flip, convert data type, and normalize ops
        self.image_size = image_size
        self.mean_val = mean_val
        self.std_val = std_val
        self.random_scale = RandomScale(scales=[0.5, 1, 2])
        self.pad = Pad(self.image_size)
        self.random_crop = RandomCrop(self.image_size)
        self.random_flip = RandomFlip(prob=0.5)
        self.normalize = Normalize(self.mean_val, self.std_val)
        self.convert_datatype = ConvertDataType()
        self.transform = Compose([self.random_scale,
                                  self.pad,
                                  self.random_crop,
                                  self.normalize,
                                  self.convert_datatype])

    def __call__(self, image, label):
        return self.augment(image, label)

    def augment(self, image, label):
        image, label = self.transform(image, label)
        return image, label