import torch
import torchvision.datasets as datasets
import os
from torchvision import transforms


class CelebADataset:

    def __init__(self, use_cond, train, dataset, image_path, image_size, label_path=None, shuffle=True, batchsize=64):
        self.use_cond = use_cond
        self.train = train
        self.batchsize = batchsize
        self.shuffle = shuffle
        self.image_path = image_path
        self.image_size = image_size
        self.dataset = dataset
        if self.use_cond and label_path is None:
            raise ValueError
        else:
            pass

    def transform(self, resize, totensor, normalize, centercrop):
        options = []
        if centercrop:
            options.append(transforms.CenterCrop(160))
        if resize:
            options.append(transforms.Resize(
                (self.image_size, self.image_size)))
        if totensor:
            options.append(transforms.ToTensor())
        if normalize:
            options.append(transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = transforms.Compose(options)
        return transform

    def load_lsun(self, classes='church_outdoor_train'):
        transforms = self.transform(True, True, True, False)
        dataset = datasets.LSUN(self.image_path, classes=[
            classes], transform=transforms)
        return dataset

    def load_celeb(self):
        transforms = self.transform(True, True, True, True)
        dataset = datasets.ImageFolder(self.image_path, transform=transforms)
        return dataset

    def loader(self):
        if self.dataset == 'lsun':
            dataset = self.load_lsun()
        elif self.dataset == 'celeb':
            dataset = self.load_celeb()
        else:
            raise NotImplementedError

        loader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=self.batchsize,
                                             shuffle=self.shuffle,
                                             num_workers=4,
                                             drop_last=True)
        return loader

# %%
# path = '/home/minteiko/developer/project/data/celebA'
# data = CelebADataset(False, True, "celeb", path, 64, batchsize=2)
# loader = data.loader()
# a = next(iter(loader))
# print(a, a[0].max(), a[0].min(), a[0].mean())