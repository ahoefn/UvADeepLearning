import random
import os
import urllib.request
from urllib.error import HTTPError
import numpy as np

import torch
import torch.nn as nn

import pytorch_lightning as pl

from data_handler import DataHandler
from CIFAR10_module import CIFAR10Module


class Main:
    def __init__(self, seed) -> None:
        self.deviceHandle = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.InitializeSeeds(seed)
        self.DownloadInitializedModels()
        self.dataHandler = DataHandler()

    def InitializeSeeds(self, seed) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        pl.seed_everything(seed)

    def DownloadInitializedModels(self) -> None:
        self.checkpointPath = "checkpoints"
        if not os.path.isdir(self.checkpointPath):
            os.makedirs(self.checkpointPath)

        # Github URL where saved models are stored for this tutorial
        baseUrl = (
            "https://raw.githubusercontent.com/phlippe/saved_models/main/tutorial5/"
        )
        # Files to download
        pretrainedFiles = [
            "GoogleNet.ckpt",
            "ResNet.ckpt",
            "ResNetPreAct.ckpt",
            "DenseNet.ckpt",
            "tensorboards/GoogleNet/events.out.tfevents.googlenet",
            "tensorboards/ResNet/events.out.tfevents.resnet",
            "tensorboards/ResNetPreAct/events.out.tfevents.resnetpreact",
            "tensorboards/DenseNet/events.out.tfevents.densenet",
        ]

        for fileName in pretrainedFiles:
            filePath = os.path.join(self.checkpointPath, fileName)
            if "/" in fileName:
                os.makedirs(filePath.rsplit("/", 1)[0], exist_ok=True)
            if not os.path.isfile(filePath):
                fileUrl = baseUrl + fileName
                print(f"Dowloading from {fileUrl}")
                try:
                    urllib.request.urlretrieve(fileUrl, filePath)
                except HTTPError as e:
                    print("Download failed, error is:\n", e)


if __name__ == "__main__":
    module = Main(2101)
