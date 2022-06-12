from torchvision import datasets, transforms
import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler



class Cifar10():
    def __init__(self, args):
        self.args = args

    def augmentation(self):

        # data augmentation and transform
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),  # Random Position Crop
            transforms.RandomHorizontalFlip(),  # right and left flip
            transforms.RandomRotation(90, expand=False),
            transforms.ToTensor(),  # change [0,255] Int value to [0,1] Float value
            transforms.Normalize(mean=(0.4914, 0.4824, 0.4467),  # RGB Normalize MEAN
                                 std=(0.2471, 0.2436, 0.2616))  # RGB Normalize Standard Deviation
        ])

        transform_valid = transforms.Compose([
            transforms.ToTensor(),  # change [0,255] Int value to [0,1] Float value
            transforms.Normalize(mean=(0.4914, 0.4824, 0.4467),  # RGB Normalize MEAN
                                 std=(0.2471, 0.2436, 0.2616))  # RGB Normalize Standard Deviation
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),  # change [0,255] Int value to [0,1] Float value
            transforms.Normalize(mean=(0.4914, 0.4824, 0.4467),  # RGB Normalize MEAN
                                 std=(0.2471, 0.2436, 0.2616))  # RGB Normalize Standard Deviation
        ])

        return transform_train, transform_valid, transform_test

    def data_load(self):

        transform_train, transform_valid, transform_test = self.augmentation()

        # data load
        train_dataset = datasets.CIFAR10(root=self.args.root_dir,
                                         train=True,
                                         transform= transform_train,
                                         download=True)

        valid_dataset = datasets.CIFAR10(root=self.args.root_dir,
                                         train=True,
                                         transform= transform_valid,
                                         download=True)

        test_dataset = datasets.CIFAR10(root=self.args.root_dir,
                                        transform= transform_test,
                                        train=False)


        # train, valid split
        valid_size = 0.1
        shuffle = True
        random_seed = 4

        # train, valid split
        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))

        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)


        # train, valid, test generate batch
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.args.batch_size, sampler=train_sampler,
            num_workers=4, pin_memory=True,
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=self.args.batch_size, sampler=valid_sampler,
            num_workers=4, pin_memory=True,
        )

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=self.args.batch_size,
                                                  shuffle=False,  # at Test Procedure, Data Shuffle = False
                                                  num_workers=4)  # CPU loader number

        return train_loader, valid_loader, test_loader
