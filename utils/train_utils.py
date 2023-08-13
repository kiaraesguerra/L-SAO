from pytorch_lightning import LightningModule
import torch
from torchmetrics import Accuracy

from optimizers.optimizers import get_optimizer
from schedulers.schedulers import get_scheduler
from criterions.criterions import get_criterion


def get_plmodule(model, args):
    model = Model(model, args)
    return model


class Model(LightningModule):
    def __init__(self, model, args):
        super().__init__()

        self.model = model
        self.epochs = args.epochs
        self.criterion = get_criterion(args)
        self.optimizer = get_optimizer(model, args)
        self.scheduler = get_scheduler(self.optimizer, args)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=args.num_classes)
        self.train_accuracy = Accuracy(task="multiclass", num_classes=args.num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=args.num_classes)

    def forward(self, x):
        out = self.model(x)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        loss = self.criterion(out, y)
        preds = torch.argmax(out, dim=1)
        self.train_accuracy.update(preds, y)
        self.log("train/loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log(
            "train/acc", self.train_accuracy, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, y)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log(
            "val/acc", self.val_accuracy, on_epoch=True, prog_bar=True, logger=True
        )

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy.update(preds, y)
        # self.log("test/loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log(
            "test/acc", self.test_accuracy, on_epoch=True, prog_bar=True, logger=True
        )

    def configure_optimizers(self):
        return [self.optimizer], [{"scheduler": self.scheduler, "interval": "epoch"}]
