import torch
import torch.nn as nn
import torch.optim as optim

import pytorch_lightning as pl


from data_handler import DataHandler
import model_builder as mb
from model_builder import ModelType, ModelHyperParameters


class CIFAR10Module(pl.LightningModule):
    def __init__(
        self,
        model_type: ModelType,
        model_hyperparams: ModelHyperParameters,
        optimizer_name,
        optimizer_hyperparams,
        data_handler: DataHandler,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()
        self.model = mb.create_model(model_type, model_hyperparams)
        self.loss_module = nn.CrossEntropyLoss()
        self.example_input_array = torch.zeros((1, 3, 32, 32), dtype=torch.float32)

        def forward(self, images):
            return self.model(images)

        def configure_optimizers(self) -> None:
            if self.hparams.optimizer_name == "Adam":
                self.optimizer = optim.AdamW(
                    self.parameters(), **self.hparams.optimizerParameters
                )

            elif self.hparams.optimizer_name == "SGD":
                self.optimizer = optim.SGD(
                    self.parameters(), **self.hparams.optimizerParameters
                )
            else:
                raise ValueError("Optimizer type unknown", self.hparams.optimizer_name)

            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=[100, 150], gamma=0.1
            )

    def training_step(self, batch, batch_id):
        images, labels = batch
        predictions = self.model(images)
        loss = self.loss_module(predictions, labels)
        accuracy = (predictions.argmax(dim=-1) == labels).float().mean()

        # Logging:
        self.log("train_acc", accuracy, on_step=False, on_epoch=True)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_id) -> None:
        images, labels = batch
        predictions = self.model(images).argmax(dim=-1)
        accuracy = (predictions == labels).float().mean()
        self.log("val_acc", accuracy)

    def test_step(self, batch, batch_id) -> None:
        images, labels = batch
        predictions = self.model(images).argmax(dim=-1)
        accuracy = (predictions == labels).float().mean()
        self.log("test_acc", accuracy)
