# faster_rcnn.py

# Python imports
from typing import Dict, List, Optional

# Torch imports
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights
import pytorch_lightning as pl

from util import log
from config import NUM_CLASSES
from methods.metrics import Metrics


class FasterRCNN(nn.Module):
    """
    Faster R-CNN model with ResNet-50 backbone and Feature Pyramid Network (FPN)
    """

    def __init__(
            self,
            num_classes: int = NUM_CLASSES,
            pretrained: bool = True,
            trainable_backbone_layers: int = 3,
            min_size: int = 512,
            max_size: int = 1024,
            image_mean: Optional[List[float]] = None,
            image_std: Optional[List[float]] = None,
    ):
        """
        Initialize Faster R-CNN model
        :param num_classes: number of classes (including background)
        :param pretrained: whether to use pretrained weights
        :param trainable_backbone_layers: number of trainable backbone layers
        :param min_size: minimum size of the image to be rescaled
        :param max_size: maximum size of the image to be rescaled
        :param image_mean: mean values for normalization
        :param image_std: std values for normalization
        """
        super().__init__()

        # default normalization values if not provided
        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]

        self.model = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None,
            trainable_backbone_layers=trainable_backbone_layers,
            min_size=min_size,
            max_size=max_size,
            image_mean=image_mean,
            image_std=image_std,
        )

        # new classifier for specific number of classes
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        log(f"Initialized Faster R-CNN model with {num_classes} classes")
        log(f"Trainable backbone layers: {trainable_backbone_layers}")

    def forward(self, images: List[torch.Tensor], targets: Optional[List[Dict[str, torch.Tensor]]] = None):
        """
        Forward pass of the model
        :param images: list of image tensors [B, 3, H, W]
        :param targets: list of target dictionaries with 'boxes' and 'labels' keys
        :return: loss dict if targets are provided, else detections
        """
        if self.training:
            assert targets is not None, "Targets must be provided in training mode"
            return self.model(images, targets)  # Returns loss dictionary
        else:
            return self.model(images)  # Returns predictions only


class FasterRCNNLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for Faster R-CNN
    """

    def __init__(
            self,
            model: FasterRCNN,
            learning_rate: float = 0.001,
            weight_decay: float = 0.0005,
            num_classes: int = NUM_CLASSES,
    ):
        """
        Initialize the Lightning module
        :param model: Faster R-CNN model instance
        :param learning_rate: learning rate for optimizer
        :param weight_decay: weight decay for optimizer
        """
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # initialize custom metrics class from methods.metrics
        self.train_metrics = Metrics(box_format='xyxy', num_classes=num_classes)
        self.val_metrics = Metrics(box_format='xyxy', num_classes=num_classes)
        self.test_metrics = Metrics(box_format='xyxy', num_classes=num_classes)

        # save hyperparameters for checkpointing callbacks
        self.save_hyperparameters(ignore=['model'])

        self.val_loss = None

    def forward(self, images: List[torch.Tensor], targets: Optional[List[Dict[str, torch.Tensor]]] = None):
        return self.model(images, targets)

    def configure_optimizers(self):
        """Configure Adam optimizer with weight decay"""
        return Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

    def training_step(self, batch, batch_idx):
        images, targets, _ = batch
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

        loss_dict = self(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        self.log_dict({f"train/{k}": v for k, v in loss_dict.items()}, on_step=True, on_epoch=True,
                      batch_size=len(images))
        self.log("train/loss", losses, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(images))

        return losses

    def validation_step(self, batch, batch_idx):
        images, targets, _ = batch
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

        self.model.eval()
        with torch.no_grad():
            predictions = self(images)

        self.val_metrics.update(predictions, targets)

        self.val_metrics.update(predictions, targets)

        self.model.train()
        with torch.no_grad():
            loss_dict = self(images, targets)
        self.model.eval()

        losses = sum(loss for loss in loss_dict.values())
        self.log("val/loss", losses, on_epoch=True, prog_bar=True, batch_size=len(images))

        self.log("val_loss", losses, on_epoch=True, prog_bar=True, batch_size=len(images))

    def on_validation_epoch_end(self):
        val_results = self.val_metrics.compute()
        self.log_dict({f"val/{k}": v for k, v in val_results.items()}, prog_bar=True)
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        images, targets, _ = batch
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

        predictions = self(images)
        self.test_metrics.update(predictions, targets)

    def on_test_epoch_end(self):
        test_results = self.test_metrics.compute()
        self.log_dict({f"test/{k}": v for k, v in test_results.items()})
        self.test_metrics.reset()

