# detector_module.py

# Torch imports
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchvision.ops import box_iou
from torchmetrics.detection import MeanAveragePrecision
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

# Image logging in TensorBoard
from torchvision.utils import draw_bounding_boxes
from torch.utils.tensorboard import SummaryWriter

# Python imports
import os
import math
import numpy as np
from typing import Optional, Union, Any

# Local imports
from util import log
from config import (
    FOCAL_LOSS_ALPHA,
    FOCAL_LOSS_GAMMA,
    GIOU_LOSS_COEF,
    L1_LOSS_COEF,
    CLS_LOSS_COEF,
    POSITIVE_CLASS_ID,
    CLASSES
)

# tensorboard logging
PRED_COLOR = "blue"
GT_COLOR = "green"
LOG_SCORE_THRESHOLD = 0.15 # For the TensorBoard logger, boxes with confidence score > will be put on the image


class DetectionLightningModule(pl.LightningModule):
    """
    Pytorch Lightning Module for stenosis detection models

    Handles training, validation, testing loops, optimization, logging, and metric calculation
    """
    def __init__(
            self,
            model: nn.Module,
            model_stage: int = 1,  # 1 for EffNet, 2: TSM, 3: Transformer
            learning_rate: float = 1e-4,
            weight_decay: float = 1e-4,
            warmup_steps: int = 50,
            max_epochs: int = 24,
            batch_size: int = 32,
            accumulate_grad_batches: int = 1,
            # specific params for stage 3
            stem_learning_rate: float = 1e-5,  # differential LR
            # loss params (can be overridden by model-specific losses)
            focal_alpha: float = FOCAL_LOSS_ALPHA,
            focal_gamma: float = FOCAL_LOSS_GAMMA,
            smooth_l1_beta: float = 1.0 / 9.0,  # common default for smooth L1
            giou_loss_coef: float = GIOU_LOSS_COEF,
            cls_loss_coef: float = CLS_LOSS_COEF,
            # as extension for another future project :)
            positive_class_id: int = POSITIVE_CLASS_ID,

            # image logging/vis
            normalize_params: Optional[dict] = None,
            num_log_val_images: int = 1,  # how many images will be shown in the board (per epoch)
            num_log_test_images: Union[int, str] = 'all',
    ):
        super().__init__()
        self.model = model
        self.model_stage = model_stage
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.accumulate_grad_batches = accumulate_grad_batches
        self.stem_learning_rate = stem_learning_rate
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.smooth_l1_beta = smooth_l1_beta
        self.giou_loss_coef = giou_loss_coef
        self.cls_loss_coef = cls_loss_coef
        self.positive_class_id = positive_class_id


        if model_stage not in [1, 2, 3]:
            raise ValueError(f'Invalid model_stage: {model_stage}. Must be in [1, 2, 3]')

        self.save_hyperparameters(ignore=['model'])

        # ------ initialize metrics -------
        common_map_args = {
            "iou_type": "bbox", "iou_thresholds": [0.5],
            "rec_thresholds": torch.linspace(0, 1, 101).tolist(),
            "class_metrics": False,
        }

        self.val_map = MeanAveragePrecision(**common_map_args)
        self.test_map = MeanAveragePrecision(**common_map_args)

        self.val_preds, self.val_targets = [], []
        self.test_preds, self.test_targets = [], []


        # ----- redundancy, input validation ------
        self._train_input_validated = False
        self._val_input_validated = False
        self._test_input_validated = False


        # ----- tensorboard/logging ------
        self.normalize_params = normalize_params
        self.num_log_val_images = num_log_val_images
        self.num_log_test_images = num_log_test_images

        self.val_image_samples = []
        self.val_pred_samples = []
        self.val_target_samples = []

        self.test_image_samples = []
        self.test_pred_samples = []
        self.test_target_samples = []

        if not (isinstance(self.num_log_test_images, int) and self.num_log_test_images >= 0) and self.num_log_test_images != 'all':
            raise ValueError(f'num_log_test_images must be an integer >= 0 or "all", got {self.num_log_test_images}')

        if (self.num_log_val_images > 0 or self.num_log_test_images != 0) and self.normalize_params is None:
             log("Warning: normalize_params not provided to DetectionLightningModule. Cannot de-normalize images for logging.")
             self.num_log_val_images = 0
             self.num_log_test_images = 0


        self.slurm = "SLURM_JOB_ID" in os.environ


    def forward(
            self,
            images: Union[list[torch.Tensor], torch.Tensor],
            targets: Optional[list[dict[str, torch.Tensor]]] = None) -> Any:
        """
        Forward pass of the model. Input format depends on the model stage
        :param images: input images
            Stage 1: Expected list[tensor[C, H, W]]
            Stage 2/3: Expected [B, T, C, H, W]
        :param targets: expected targets
        """
        return self.model(images, targets)


    def _validate_input_shape(self, images: Union[list[torch.Tensor], torch.Tensor], stage: str) -> None:
        """Checks input shape at initialization per stage (train/val/test)."""
        is_correct_shape = True
        err_msg = ""

        if self.model_stage == 1:
            if isinstance(images, list):
                if not images or not (isinstance(images[0], torch.Tensor) and images[0].ndim == 3):
                    is_correct_shape = False
                    err_msg = (f"Stage 1 expected a list[tensor[c, h, w]], but got a list containing type {type(images[0])}"
                               f" with ndim {images[0].ndim if hasattr(images[0], 'ndim') else 'N/A'}.")

            elif isinstance(images, torch.Tensor):
                if images.ndim != 4:
                    is_correct_shape = False
                    err_msg = f"Stage 1 expected tensor [b, c, h, w] or list[tensor[c, h, w]], but got tensor with ndim {images.ndim}"

            else:
                is_correct_shape = False
                err_msg = f"Stage 1 received unexpected input type {type(images)}"

        elif self.model_stage in [2, 3]:
            if not (isinstance(images, torch.Tensor) and images.ndim == 5):
                is_correct_shape = False
                image_info = f"type {type(images)}"
                if hasattr(images, 'ndim'): image_info += f" with ndim {images.ndim}"
                elif isinstance(images, list): image_info += f" (List of len {len(images)})"
                err_msg = f"Stage {self.model_stage} expected Tensor[B,T,C,H,W], but got {image_info}."

        if not is_correct_shape:
            log(f"Input shape validation FAILED during stage {stage}: {err_msg}")


    def _step_logic(self, batch, batch_idx, step_type: str):
        """Shared logic for training, validation and test steps"""
        images, targets, _ = batch

        # ------ perform input shape validation -----
        if step_type == 'train' and not self._train_input_validated:
            self._validate_input_shape(images, 'train')
            self._train_input_validated = True
        elif step_type == 'val' and not self._val_input_validated:
            if not self.trainer.sanity_checking:
                self._validate_input_shape(images, 'val')
            self._val_input_validated = True
        elif step_type == 'test' and not self._test_input_validated:
            self._validate_input_shape(images, 'test')
            self._test_input_validated = True
        # ---------------------------------

        # --- step logic (forward pass, loss/pred, logging)
        if step_type == 'train':
            pass # moved to train_step()

        else:  # val/test
            is_val = step_type == 'val'
            is_test = step_type == 'test'
            map_metric = self.val_map if is_val else self.test_map
            pred_list = self.val_preds if is_val else self.test_preds
            target_list = self.val_targets if is_val else self.test_targets

            self.model.eval()
            with torch.no_grad():
                preds = self.model(images)  # val/test pass for model, get preds

            # pass preds and targets to cpu - for tensorboard
            cpu_preds_batch = [{k: v.cpu() for k, v in p.items()} for p in preds]
            cpu_targets_batch = [{k: v.cpu() for k, v in t.items()} for t in targets]

            map_metric.update(preds, targets)

            pred_list.extend(cpu_preds_batch)
            target_list.extend(cpu_targets_batch)

            # ----- get image samples for tensorboard logging (validation)  --------
            if is_val and not self.trainer.sanity_checking and self.num_log_val_images > 0:
                current_samples = len(self.val_image_samples)
                needed = self.num_log_val_images
                if current_samples < needed:
                    num_to_add = min(needed - current_samples, len(images))
                    self.val_image_samples.extend([img.cpu() for img in images[:num_to_add]])
                    self.val_pred_samples.extend(cpu_preds_batch[:num_to_add])
                    self.val_target_samples.extend(cpu_targets_batch[:num_to_add])

            # ------ same, for testing -----------
            elif is_test and self.num_log_test_images != 0:
                num_to_add = 0
                if self.num_log_test_images == "all": num_to_add = len(images)
                else:
                    current_samples = len(self.test_image_samples)
                    needed = self.num_log_test_images
                    if current_samples < needed:
                        num_to_add = min(needed - current_samples, len(images))

                if num_to_add > 0:
                    self.test_image_samples.extend([img.cpu() for img in images[:num_to_add]])
                    self.test_pred_samples.extend(cpu_preds_batch[:num_to_add])
                    self.test_target_samples.extend(cpu_targets_batch[:num_to_add])


            # --- calculate and log val loss -----
            if is_val:
                # switch to train mode to get losses
                original_mode = self.model.training
                self.model.train()
                with torch.no_grad():
                    targets_on_device = [{k: v.to(self.device) for k,v in t.items()} for t in targets]
                    loss_dict = self.model(images, targets_on_device)
                self.model.train(original_mode)

                if not loss_dict: val_losses = torch.tensor(0.0, device=self.device)
                else:
                    valid_losses = [loss for loss in loss_dict.values() if isinstance(loss, torch.Tensor)]
                    if not valid_losses: val_losses = torch.tensor(0.0, device=self.device)
                    else: val_losses = sum(valid_losses)


                log_kwargs_val = {'on_step': False, 'on_epoch': True, 'prog_bar': False, 'logger': True, 'sync_dist': True, 'batch_size': len(images)}
                if loss_dict:
                    for k, v in loss_dict.items():
                        if isinstance(v, torch.Tensor):
                             self.log(f'val/{k}', v, **log_kwargs_val)
                self.log(f'val_loss', val_losses, **log_kwargs_val)





    def training_step(self, batch, batch_idx):
        """The training step needs to explicitly return the losses for logging"""
        images, targets, _ = batch # Unpack batch
        if targets is None: raise ValueError("Targets must be provided during training")

        # we make sure targets are on the correct device
        targets_on_device = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

        loss_dict = self.model(images, targets)
        if not loss_dict:
            log(f"Epoch {self.current_epoch}, Step {self.global_step}: Loss dictionary is empty.")
            losses = torch.tensor(0.0, device=self.device, requires_grad=True) # ensure loss is on correct device and requires grad
        else:
            valid_losses = [loss for loss in loss_dict.values() if isinstance(loss, torch.Tensor)]
            if not valid_losses: losses = torch.tensor(0.0, device=self.device, requires_grad=True)
            else: losses = sum(valid_losses)

        # -------- logging & progress bar
        log_kwargs = {'on_step': True, 'on_epoch': True, 'prog_bar': True, 'logger': True, 'sync_dist': True, 'batch_size': len(images)}
        if loss_dict:
            for k,v in loss_dict.items():
                 if isinstance(v, torch.Tensor):
                    self.log(f'train/{k}', v, **log_kwargs)
        self.log(f'train/loss', losses, **log_kwargs)

        # log lr only during training
        scheduler = self.lr_schedulers()
        if scheduler:
            lr = scheduler.get_last_lr()[0]
            self.log('train/lr', lr, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        return losses


    def validation_step(self, batch, batch_idx):
        self._step_logic(batch, batch_idx, 'val')
        return 0

    def test_step(self, batch, batch_idx):
        self._step_logic(batch, batch_idx, 'test')
        return 0

    def _calc_metrics(self, stage: str) -> None:
        """
        Compute, log, and reset metrics for either 'val' or 'test' stage.
        """
        map_obj = getattr(self, f"{stage}_map")
        preds_attr = getattr(self, f"{stage}_preds")
        targs_attr = getattr(self, f"{stage}_targets")
        prefix = f"{stage}/"
        # prog_bar + sync_dist only for validation
        prog_bar = stage == "val"
        sync_dist = stage == "val"
        # choose logger

        try:
            # compute mAP/mAR
            metrics = map_obj.compute()
            to_num = lambda x: x.item() if isinstance(x, torch.Tensor) else x
            m50 = to_num(metrics["map_50"])
            m_all = to_num(metrics["map"])
            mar100 = to_num(metrics["mar_100"])

            self.log_dict(
                {f"{prefix}mAP_0.5": m50,
                 f"{prefix}mAP": m_all,
                 f"{prefix}mAR_100": mar100},
                prog_bar=prog_bar, logger=True, sync_dist=sync_dist, batch_size=1
            )

            # move preds/targets to CPU once
            cpu_preds = [{k: v.cpu() for k, v in p.items()} for p in preds_attr]
            cpu_targs = [{k: v.cpu() for k, v in t.items()} for t in targs_attr]

            ap_small = self.compute_ap_for_area(cpu_preds, cpu_targs, max_area=32**2)
            precision, recall, f1, avg_iou_tp = self.compute_prf1(cpu_preds, cpu_targs, iou_threshold=0.5)

            # log the rest
            extra = {
                f"{prefix}AvgIoU_TP_0.5": avg_iou_tp,
                f"{prefix}AP_small": ap_small,
                f"{prefix}Precision_0.5": precision,
                f"{prefix}Recall_0.5": recall,
                f"{prefix}F1_0.5": f1,
            }
            for name, val in extra.items():
                prog_bar_extra = prog_bar and name.endswith(("F1_0.5", "AvgIoU_TP_0.5"))
                self.log(name, val, prog_bar=prog_bar_extra ,logger=True, sync_dist=sync_dist)

        except Exception as e:
            log(f"Error computing {stage} metrics: {e}")
            zeros = {f"{prefix}{k}": 0.0 for k in
                     ["mAP_0.5", "mAP", "mAR_100", "AP_small",
                      "Precision_0.5", "Recall_0.5", "F1_0.5", "AvgIoU_TP_0.5"]}
            self.log_dict(zeros, logger=True, batch_size=1)

        finally:
            map_obj.reset()
            preds_attr.clear()
            targs_attr.clear()


    def on_train_start(self) -> None:
        self._train_input_validated = False

    def on_validation_start(self) -> None:
        if not self.trainer.sanity_checking:
            self._val_input_validated = False

            self.val_preds.clear()
            self.val_targets.clear()
            self.val_image_samples.clear()
            self.val_pred_samples.clear()
            self.val_target_samples.clear()

    def on_test_start(self) -> None:
        self._test_input_validated = False
        self.test_preds.clear()
        self.test_targets.clear()
        self.test_image_samples.clear()
        self.test_pred_samples.clear()
        self.test_target_samples.clear()


    def on_validation_epoch_end(self) -> None:
        if self.trainer.is_global_zero:  # only on GPU with rank:0
            self._calc_metrics('val')
            self._slrum_print_metrics()  # in slurm, only print the metrics if we're on gpu:0

        should_log_images = (  # will evaluate to true iff these are correct
                hasattr(self, 'logger') and
                self.logger is not None and
                isinstance(self.logger.experiment, SummaryWriter) and  #ensure proper tensorboard
                self.num_log_val_images > 0 and
                not self.trainer.sanity_checking and
                self.trainer.is_global_zero
        )

        if should_log_images and self.val_image_samples:  # also if samples exist
            self._log_image_samples_to_tensorboard(
                image_samples=self.val_image_samples,
                pred_samples=self.val_pred_samples,
                target_samples=self.val_target_samples,
                writer=self.logger.experiment,
                tag='Validation Image Samples',
                global_step=self.current_epoch
            )

        self.val_image_samples.clear()
        self.val_pred_samples.clear()
        self.val_target_samples.clear()



    def on_test_epoch_end(self) -> None:
        if self.trainer.is_global_zero:
            self._calc_metrics('test')

        should_log_images = (  #check for test images as well
                hasattr(self, 'logger') and
                self.logger is not None and
                isinstance(self.logger.experiment, SummaryWriter) and
                self.num_log_test_images != 0 and
                self.trainer.is_global_zero
        )

        if should_log_images and self.test_image_samples:
            self._log_image_samples_to_tensorboard(
                image_samples=self.test_image_samples,
                pred_samples=self.test_pred_samples,
                target_samples=self.test_target_samples,
                writer=self.logger.experiment,
                tag='Testing Image Samples',
                global_step=self.current_epoch
            )

        self.test_image_samples.clear()
        self.test_pred_samples.clear()
        self.test_target_samples.clear()


    def configure_optimizers(self):
        """
        Configures the optimizer (AdamW) and lr scheduler (linear warmup + cosine ann)
        Handles differential lr for stage 3
        """
        parameters = []
        if self.model_stage == 3:
            # TODO: implement
            raise NotImplementedError(f"Optimizers are not implemented yet for stage {self.model_stage}")

        else:
            log(f'Configuring optimizer for stage {self.model_stage} with single LR: {self.learning_rate}')
            parameters = [p for p in self.model.parameters() if p.requires_grad]

        # ---- optimizer -----
        optimizer = optim.AdamW(
            parameters,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # ---- scheduler calculation -----
        total_training_steps = 0
        warmup_steps = self.warmup_steps

        try:
            if self.trainer and self.trainer.datamodule and hasattr(self.trainer, 'estimated_stepping_batches'):
                total_training_steps = self.trainer.estimated_stepping_batches
                log(f"Using trainer.estimated_stepping_batches: {total_training_steps}")
            elif self.trainer and self.trainer.datamodule:
                gpus = self.trainer.num_devices if self.trainer.num_devices > 0 else 1
                num_samples = len(self.trainer.datamodule.train_dataloader().dataset)
                effective_batch_size = self.batch_size * gpus * self.accumulate_grad_batches
                steps_per_epoch = math.ceil(num_samples / effective_batch_size)
                total_training_steps = steps_per_epoch * self.max_epochs
                log(f"Manually estimated scheduler steps: Samples={num_samples}, EffBatch={effective_batch_size}, "
                    f"Steps/Epoch={steps_per_epoch}, Total={total_training_steps}")
            else:
                log(f"Warning: Trainer/datamodule not available during optimizer setup. "
                    f"Using potentially inaccurate fallback for scheduler steps")
                total_training_steps = warmup_steps + 10000

        except Exception as e:
            log(f'Error calculating scheduler steps, using fallback: {e}')
            total_training_steps = warmup_steps + 10000


        if total_training_steps <= warmup_steps:
            log(f"Warning: Total steps ({total_training_steps}) <= warmup steps ({warmup_steps})."
                f"Scheduler might not behave as expected. Check max_epochs, dataset size, batch size, accumulation")
            cosine_steps = 1
        else:
            cosine_steps = total_training_steps - warmup_steps

        log(f"Scheduler config: Total Steps={total_training_steps}, Warmup Steps={warmup_steps}, Cosine Steps={cosine_steps}")

        # ----- schedulers -----
        warmup_scheduler = LinearLR(optimizer, start_factor=1e-6, total_iters=warmup_steps)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, cosine_steps), eta_min=1e-7)
        lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1
            }
        }


    def compute_ap_for_area(self, predictions, targets, max_area=None, min_area=None, iou_threshold=0.5):
        """
        Computes average precision (AP) for the positive class (stenosis, label=1) within a specific area range.
        """
        if not predictions or not targets: return 0.0

        all_scores = []  # scores of preds for positive class
        all_matches = []  # 1 for TP (positive class), 0 for FP (positive class)
        num_gt_total = 0  # total number of gt boxes for the pos. cls.

        device = predictions[0]['boxes'].device if predictions and 'boxes' in predictions[0] and predictions[0]['boxes'].numel() > 0 else 'cpu'

        for img_preds, img_targets in zip(predictions, targets):
            gt_boxes_all, gt_labels_all = img_targets['boxes'].to(device), img_targets['labels'].to(device)
            pred_boxes_all, pred_scores_all, pred_labels_all = img_preds['boxes'].to(device), img_preds['scores'].to(device), img_preds['labels'].to(device)

            # --- filter gt for the positive class ---
            gt_pos_mask = (gt_labels_all == self.positive_class_id)
            gt_boxes = gt_boxes_all[gt_pos_mask]

            # --- Filter gt by area ---
            if gt_boxes.shape[0] > 0 and (max_area is not None or min_area is not None):
                gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
                area_mask = torch.ones(gt_areas.shape[0], dtype=torch.bool, device=device)
                if max_area is not None: area_mask &= (gt_areas < max_area)
                if min_area is not None: area_mask &= (gt_areas >= min_area)
                gt_boxes = gt_boxes[area_mask]

            num_gt_img = gt_boxes.shape[0]  # num. relevant gts for this image
            num_gt_total += num_gt_img

            # --- filter preds for the positive class ---
            pred_pos_mask = (pred_labels_all == self.positive_class_id)
            pred_boxes = pred_boxes_all[pred_pos_mask]
            pred_scores = pred_scores_all[pred_pos_mask]

            # sort positive preds by score
            sort_idx = torch.argsort(pred_scores, descending=True)
            pred_boxes = pred_boxes[sort_idx]
            pred_scores = pred_scores[sort_idx]

            num_pred_img = pred_boxes.shape[0]
            img_matches = torch.zeros(num_pred_img, dtype=torch.int8, device=device)  # 0 for FP, 1 for TP


            if num_pred_img == 0 or num_gt_img == 0: # if no preds of positive class or no relevant gts, all preds are FP (or no preds exist)
                all_scores.extend(pred_scores.tolist())
                all_matches.extend([0] * num_pred_img)
                continue

            # match positive predictions to positive ground truths
            gt_matched = torch.zeros(num_gt_img, dtype=torch.bool, device=device)
            overlaps = box_iou(pred_boxes, gt_boxes)  # [num_pos_preds, num_pos_gts]

            for i in range(num_pred_img):
                # Find the best gt match for this positive prediction
                if overlaps.shape[1] == 0: continue  # no GT boxes to match against
                best_iou, best_gt_idx = overlaps[i].max(dim=0)

                # Check iou thresh. and if the best match gt box hasn't been used
                if best_iou >= iou_threshold and not gt_matched[best_gt_idx]:
                    img_matches[i] = 1  # mark as TP
                    gt_matched[best_gt_idx] = True
                # else: prediction is FP

            all_scores.extend(pred_scores.tolist())  # scores for pos. cls. preds.
            all_matches.extend(img_matches.tolist())  # matches for them


        # --- calculate ap from collected scores and matches ---
        if num_gt_total == 0: # if no positive gts exist over all images, ap is 0 if there were positive preds, 1 otherwise
            return 0.0 if len(all_scores) > 0 else 1.0

        if len(all_scores) == 0: return 0.0  # no positive preds. made

        all_scores = torch.tensor(all_scores, device=device)
        all_matches = torch.tensor(all_matches, device=device)
        sort_idx = torch.argsort(all_scores, descending=True)
        all_matches = all_matches[sort_idx]

        tp = torch.cumsum(all_matches, dim=0)
        fp = torch.cumsum(1 - all_matches, dim=0)

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (num_gt_total + 1e-9)

        # standard ap calculation
        precision = torch.cat((torch.tensor([precision[0]], device=device), precision))
        recall = torch.cat((torch.tensor([0.0], device=device), recall))

        # ensure precision is monotonically decreasing
        for i in range(precision.shape[0] - 1, 0, -1):
            precision[i - 1] = torch.maximum(precision[i - 1], precision[i])
        i = torch.where(recall[1:] != recall[:-1])[0]
        ap = torch.sum((recall[i + 1] - recall[i]) * precision[i + 1])

        return ap.item()


    def compute_prf1(self, predictions, targets, iou_threshold=0.5, score_threshold=0.5):
        """
        Computes Precision, Recall, and F1 score for the positive class (stenosis, label=1) at given IoU and score thresholds.
        """
        tp, fp, fn = 0, 0, 0
        num_gt_total, num_pred_total = 0, 0
        tp_ious = []

        device = predictions[0]['boxes'].device if predictions and 'boxes' in predictions[0] and predictions[0]['boxes'].numel() > 0 else 'cpu'
        targets = [{k: v.to(device) for k,v in t.items()} for t in targets]
        predictions = [{k: v.to(device) for k,v in p.items()} for p in predictions]

        for img_preds, img_targets in zip(predictions, targets):
            gt_boxes_all, gt_labels_all = img_targets['boxes'].to(device), img_targets['labels'].to(device)
            pred_boxes_all, pred_scores_all, pred_labels_all = img_preds['boxes'].to(device), img_preds['scores'].to(device), img_preds['labels'].to(device)

            # --- Filter gt for pos cls ---
            gt_pos_mask = (gt_labels_all == self.positive_class_id)
            gt_boxes = gt_boxes_all[gt_pos_mask]
            num_gt_img = gt_boxes.shape[0]
            num_gt_total += num_gt_img

            # --- filter preds for pos. cls. id and score threshold ---
            pred_pos_mask = (pred_labels_all == self.positive_class_id) & (pred_scores_all >= score_threshold)
            pred_boxes = pred_boxes_all[pred_pos_mask]
            pred_scores =pred_scores_all[pred_pos_mask]
            num_pred_img = pred_boxes.shape[0]
            num_pred_total += num_pred_img

            if num_gt_img == 0 and num_pred_img == 0:
                continue  # no relevant boxes in this image
            elif num_pred_img == 0:
                fn += num_gt_img  # all pos gts are missed
                continue
            elif num_gt_img == 0:
                fp += num_pred_img  # all pos. preds are fps
                continue

            gt_matched = torch.zeros(num_gt_img, dtype=torch.bool, device=device)
            pred_matched_to_gt = torch.zeros(num_pred_img, dtype=torch.bool, device=device)
            overlaps = box_iou(pred_boxes, gt_boxes)  # [num_pos_preds_above_thresh, num_pos_gts]

            sort_idx = torch.argsort(pred_scores, descending=True)  # sort preds by score

            # greedy matching based on score
            for i in sort_idx:
                pred_idx = i.item()
                if overlaps.shape[1] == 0: continue   # no gt boxes left
                overlap_values = overlaps[pred_idx]
                if overlap_values.numel()==0: continue  # case where gt boxes are empty
                best_iou, best_gt_idx = overlap_values.max(dim=0)
                if best_iou >= iou_threshold and not gt_matched[best_gt_idx]:
                    gt_matched[best_gt_idx] = True
                    pred_matched_to_gt[best_gt_idx] = True
                    tp_ious.append(best_iou.item())


            # calculate TPs for this image
            img_tp = pred_matched_to_gt.sum().item()
            tp += img_tp

        # calculate overall FP and FN
        fp = num_pred_total - tp  # total pos predictions above threshold thats not TPs
        fn = num_gt_total - tp

        # finally, calculate precision, recall, f1
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
        avg_iou_tp = np.mean(tp_ious) if tp_ious else 0.0

        return float(precision), float(recall), float(f1), float(avg_iou_tp)


    def _denormalize_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """De-normalize an image tensor using normalization params"""
        if self.normalize_params is None:
            log("Cannot de-normalize: normalize_params not available.")
            return torch.clamp(image_tensor, 0, 1)

        try:
            mean_val, std_val = self.normalize_params['mean'], self.normalize_params['std']
        except KeyError:
            log("Cannot de-normalize: 'mean' or 'std' key missing in normalize_params.", show_func=False)
            return torch.clamp(image_tensor, 0, 1)
        except TypeError:
            log("Cannot de-normalize: normalize_params is not a dictionary or None.", show_func=False)
            return torch.clamp(image_tensor, 0, 1)

        num_channels = image_tensor.shape[0]

        if isinstance(mean_val, (float, int)):
            mean = torch.full((num_channels,), mean_val, device=image_tensor.device)
        else: # (assume list/tuple)
            if len(mean_val) != num_channels:
                log(f"Warning: De-norm mean length ({len(mean_val)}) doesn't match image channels ({num_channels}). Using first element.")
                mean = torch.full((num_channels,), mean_val[0], device=image_tensor.device)
            else:
                mean = torch.tensor(mean_val, device=image_tensor.device)

        if isinstance(std_val, (float, int)):
            std = torch.full((num_channels,), std_val, device=image_tensor.device)
        else:
            if len(std_val) != num_channels:
                log(f"Warning: De-norm std length ({len(std_val)}) doesn't match image channels ({num_channels}). Using first element.",
                    show_func=False)
                std = torch.full((num_channels,), std_val[0], device=image_tensor.device)
            else:
                std = torch.tensor(std_val, device=image_tensor.device)


        # reshape mean/std to [C, 1, 1] for broadcast
        mean, std = mean.view(num_channels, 1, 1), std.view(num_channels, 1, 1)

        # de-normalize: (tensor * std) + mean
        image_tensor = image_tensor * std + mean

        return torch.clamp(image_tensor, 0, 1) # clamp to [0, 1] range as slight inaccuracies can push values outside


    def _log_image_samples_to_tensorboard(
            self,
            image_samples: list[Union[torch.Tensor, list[torch.Tensor]]],
            pred_samples: list[dict[str, torch.Tensor]],
            target_samples: list[dict[str, torch.Tensor]],
            writer: SummaryWriter,
            tag: str,
            global_step: int
    ):
        """Helper function to process and log image samples with boxes to TensorBoard."""

        if not image_samples:
            log(f"No image samples provided for logging to TensorBoard tag '{tag}'.")
            return

        processed_images = []
        log(f"Processing {len(image_samples)} image samples for TensorBoard tag '{tag}'...")

        for i, (img_sample, pred, target) in enumerate(zip(image_samples, pred_samples, target_samples)):
            try:
                # --- 1. prepare image tensor ---
                if isinstance(img_sample, list) and len(img_sample) == 1 and isinstance(img_sample[0], torch.Tensor):
                    img_tensor = img_sample[0]
                elif isinstance(img_sample, torch.Tensor):
                    img_tensor = img_sample
                else:
                    log(f"Skipping image log for sample {i}: Unexpected item type or structure in image_samples: {type(img_sample)}")
                    continue

                img_tensor_cpu = img_tensor.cpu()  # ensure images are on cpu for denorm

                # denorm and convert to uint8
                img_denorm = self._denormalize_image(img_tensor_cpu)
                # clamp, multiply by 255
                img_uint8 = (img_denorm.clamp(0, 1) * 255).to(torch.uint8)

                # potential extra batch dim, ensure 3 channels (HWC)
                if img_uint8.dim() == 4 and img_uint8.shape[0] == 1:
                    img_uint8 = img_uint8.squeeze(0)
                if img_uint8.dim() != 3:
                    log(f"Skipping image log for sample {i}: Image tensor has unexpected dimensions after processing: {img_uint8.shape}")
                    continue
                # draw_bounding_boxes expects C, H, W with C=3
                if img_uint8.shape[0] != 3:
                    log(f"Skipping image log for sample {i}: Image tensor does not have 3 channels after processing: {img_uint8.shape}")
                    continue

                # --- 2. prep bboxes ---

                pred_scores = pred.get('scores', torch.tensor([]))
                pred_boxes = pred.get('boxes', torch.zeros((0, 4)))
                pred_labels_indices = pred.get('labels', torch.tensor([]))

                # filter predictions by score thr
                score_mask = pred_scores >= LOG_SCORE_THRESHOLD
                pred_boxes_filtered = pred_boxes[score_mask]
                pred_labels_indices_filtered = pred_labels_indices[score_mask]
                pred_scores_filtered = pred_scores[score_mask]

                # label indices should be valid before accessing CLASSES
                pred_labels_text = [f"{CLASSES[label_idx]}: {score:.2f}"
                                    for label_idx, score in
                                    zip(pred_labels_indices_filtered.int(), pred_scores_filtered)
                                    if 0 <= label_idx < len(CLASSES)]

                # --- 3. prep gt boxes ---
                gt_boxes = target.get('boxes', torch.zeros((0, 4)))
                gt_labels_indices = target.get('labels', torch.tensor([]))
                gt_labels_text = [f"GT: {CLASSES[label_idx]}"
                                  for label_idx in gt_labels_indices.int()
                                  if 0 <= label_idx < len(CLASSES)]

                # --- 4. draw boxes ---
                img_with_boxes = img_uint8.clone()

                if len(gt_boxes) > 0 and len(gt_boxes) == len(gt_labels_text):
                    try:
                        img_with_boxes = draw_bounding_boxes(img_with_boxes, gt_boxes, labels=gt_labels_text, colors=GT_COLOR, width=2)
                    except Exception as e:
                        log(f"Error drawing GT boxes for sample {i}: {e}. GT Boxes shape: {gt_boxes.shape}, Labels: {gt_labels_text}")
                elif len(gt_boxes) != len(gt_labels_text):
                    log(f"Skipping GT box drawing for sample {i}: Mismatch between number of boxes ({len(gt_boxes)}) and labels ({len(gt_labels_text)}).")

                if len(pred_boxes_filtered) > 0 and len(pred_boxes_filtered) == len(pred_labels_text):
                    try:
                        img_with_boxes = draw_bounding_boxes(img_with_boxes, pred_boxes_filtered, labels=pred_labels_text, colors=PRED_COLOR, width=2)
                    except Exception as e:
                        log(f"Error drawing Pred boxes for sample {i}: {e}. Pred Boxes shape: {pred_boxes_filtered.shape}, Labels: {pred_labels_text}")
                elif len(pred_boxes_filtered) != len(pred_labels_text):
                    log(f"Skipping Pred box drawing for sample {i}: Mismatch between number of boxes ({len(pred_boxes_filtered)}) and labels ({len(pred_labels_text)}).")

                processed_images.append(img_with_boxes)

            except Exception as e:
                log(f"Error processing image sample {i} for TensorBoard: {e}")
                import traceback
                traceback.print_exc()  #debugging
                continue

        # --- 5. log to board ---
        if processed_images:
            try:
                image_batch = torch.stack(processed_images)
                writer.add_images(
                    tag=tag,
                    img_tensor=image_batch,
                    global_step=global_step,
                    dataformats='NCHW'  # images are [N, C, H, W] after stacking
                )
            except Exception as e:
                log(f"Error logging image batch to TensorBoard for tag '{tag}', step {global_step}: {e}")
        else:
            log(f"No images were successfully processed to log for TensorBoard tag '{tag}'.")



    def _slrum_print_metrics(self):
        """Prints metrics of interest per validation epoch end, only in a slurm env, where pbar is diabled."""
        metrics = self.trainer.callback_metrics
        val_loss = metrics.get('val_loss')
        map_50 = metrics.get('val/mAP_0.5')
        map_all = metrics.get('val/mAP')
        mar_100 = metrics.get('val/mAR_100')
        f1_50 = metrics.get('val/F1_0.5')
        avg_iou_tp = metrics.get('val/AvgIoU_TP_0.5')
        precision_50 = metrics.get('val/Precision_0.5')
        recall_50 = metrics.get('val/Recall_0.5')
        ap_small = metrics.get('val/AP_Small')

        log_message = f"\n--- Epoch {self.current_epoch} Validation Summary (Console Log) ---"
        log_message += f"\n  {'val_loss:':<18} {val_loss:.4f}" if val_loss is not None else "\n  val_loss: N/A"
        log_message += f"\n  {'val/mAP_0.5:':<18} {map_50:.4f}" if map_50 is not None else "\n  val/mAP_0.5: N/A"
        log_message += f"\n  {'val/mAP:':<18} {map_all:.4f}" if map_all is not None else "\n  val/mAP: N/A"
        log_message += f"\n  {'val/mAR_100:':<18} {mar_100:.4f}" if mar_100 is not None else "\n  val/mAR_100: N/A"
        log_message += f"\n  {'val/F1_0.5:':<18} {f1_50:.4f}" if f1_50 is not None else "\n  val/F1_0.5: N/A"
        log_message += f"\n  {'val/AvgIoU_TP_0.5:':<18} {avg_iou_tp:.4f}" if avg_iou_tp is not None else "\n  val/AvgIoU_TP_0.5: N/A"
        log_message += f"\n  {'val/Precision_0.5:':<18} {precision_50:.4f}" if precision_50 is not None else "\n  val/Precision_0.5: N/A"
        log_message += f"\n  {'val/Recall_0.5:':<18} {recall_50:.4f}" if recall_50 is not None else "\n  val/Recall_0.5: N/A"
        log_message += f"\n  {'val/AP_small:':<18} {ap_small:.4f}" if ap_small is not None else "\n  val/AP_small: N/A"
        log_message += "\n----------------------------------------------------"

        log(log_message, verbose=self.slurm, flush=True)


