# faster_rcnn.py

# Torch imports
import torch
import torchvision
import torch.nn as nn

# Pretrained model imports
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights


class FasterRCNN(nn.Module):

    mode = 'single_frame'

    def __init__(self, num_classes=2):
        super(FasterRCNN, self).__init__()

        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        )
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def forward(self, images, targets=None):
        return self.model(images, targets)

    def loss_fn(self, loss_dict):
        return sum(loss for loss in loss_dict.values())
