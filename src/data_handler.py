import urllib.request
from urllib.error import HTTPError

import torch
import torch.utils
import torch.utils.data as tData

from torchvision.datasets import CIFAR10
from torchvision import transforms


class DataHandler:
    def __init__(self) -> None:
        self.dataPath = "data"
        self.InitializeData()

    def InitializeData(self) -> None:
        # Download data:
        dataSet: CIFAR10 = CIFAR10(root=self.dataPath, train=True, download=True)

        # Define transformations, first normalize test data:
        print(dataSet.data.shape)
        dataMean = (dataSet.data / 255.0).mean(axis=(0, 1, 2))
        dataStd = (dataSet.data / 255.0).std(axis=(0, 1, 2))
        testTransform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(dataMean, dataStd)]
        )
        trainTransform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(
                    (32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)
                ),
                transforms.ToTensor(),
                transforms.Normalize(dataMean, dataStd),
            ]
        )

        # Now get actual data:
        trainData = CIFAR10(
            root=self.dataPath, train=True, transform=trainTransform, download=True
        )

        validationData = CIFAR10(
            root=self.dataPath, train=True, transform=testTransform, download=True
        )
        testData = CIFAR10(
            root=self.dataPath, train=False, transform=testTransform, download=True
        )

        trainDataSet, _ = torch.utils.data.random_split(trainData, [45000, 5000])
        _, validationDataSet = torch.utils.data.random_split(
            validationData, [45000, 5000]
        )

        self.trainDataLoader: tData.DataLoader = tData.DataLoader(
            trainDataSet,
            128,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=4,
        )

        self.validationDataLoader: tData.DataLoader = tData.DataLoader(
            validationDataSet,
            128,
            shuffle=False,
            drop_last=False,
            num_workers=4,
        )

        self.testDataLoader: tData.DataLoader = tData.DataLoader(
            testData,
            128,
            shuffle=False,
            drop_last=False,
            num_workers=4,
        )
