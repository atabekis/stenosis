# detector_module.py
from typing import Any, final

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
from typing import Optional, Union

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
LOG_SCORE_THRESHOLD = 0.0 # Only log predictions above this score


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
            num_log_images: int = 1,  # how many images will be shown in the board (per epoch)
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
        self.num_log_images = num_log_images

        self.val_image_samples = []
        self.val_pred_samples = []
        self.val_target_samples = []

        if self.num_log_images > 0 and self.normalize_params is None:
             log("Warning: normalize_params not provided to DetectionLightningModule. Cannot de-normalize images for logging.")
             self.num_log_images = 0



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
            self.model.eval()
            with torch.no_grad():
                preds = self.model(images)

            # ----- calculate & update metrics
            map_metric = self.val_map if step_type == 'val' else self.test_map
            map_metric.update(preds, targets)

            pred_list = self.val_preds if step_type == 'val' else self.test_preds
            target_list = self.val_targets if step_type == 'val' else self.test_targets
            pred_list.extend(preds); target_list.extend(targets)


            # ------ TensorBoard logging -------- #
            if step_type == 'val' and not self.trainer.sanity_checking:
                if len(self.val_image_samples) < self.num_log_images:
                    num_to_add = min(self.num_log_images - len(self.val_image_samples), len(images))
                    # detach the images
                    self.val_image_samples.extend([img.cpu()] for img in images[:num_to_add])
                    # detach the preds - all points of bbox
                    self.val_pred_samples.extend([{k: v.cpu() for k, v in p.items()} for p in preds[:num_to_add]])
                    self.val_target_samples.extend([{k: v.cpu() for k, v in t.items()} for t in targets[:num_to_add]])


        if step_type == 'val':
            original_mode = self.model.training
            self.model.train()
            with torch.no_grad():
                loss_dict = self.model(images, targets)
            self.model.train(original_mode)

            if not loss_dict:
                log(f"Validation Step {batch_idx}: Loss dictionary is empty.")
                val_losses = torch.tensor(0.0, device=self.device)
            else:
                val_losses = sum(loss for loss in loss_dict.values())

            log_kwargs_val = {'on_step': False, 'on_epoch': True, 'prog_bar': False, 'logger': True, 'sync_dist': True,
                              'batch_size': len(images)}
            for k, v in loss_dict.items():
                self.log(f'val/{k}', v, **log_kwargs_val)
            self.log(f'val_loss', val_losses, **log_kwargs_val)



    def training_step(self, batch, batch_idx):
        """The training step needs to explicitly return the losses for logging"""
        images, targets, _ = batch # Unpack batch
        if targets is None: raise ValueError("Targets must be provided during training")

        loss_dict = self.model(images, targets)
        if not loss_dict:
            log(f"Epoch {self.current_epoch}, Step {self.global_step}: Loss dictionary is empty.")
            losses = torch.tensor(0.0, device=self.device, requires_grad=True) # ensure loss is on correct device and requires grad
        else:
            losses = sum(loss for loss in loss_dict.values())

        # -------- logging & progress bar
        log_kwargs = {'on_step': True, 'on_epoch': True, 'prog_bar': True, 'logger': True, 'sync_dist': True, 'batch_size': len(images)}
        if loss_dict: # Only log if dict is not empty
            for k,v in loss_dict.items():
                self.log(f'train/{k}', v, **log_kwargs)
        self.log(f'train/loss', losses, **log_kwargs)
        # log lr only during training
        lr = self.lr_schedulers().get_last_lr()[0]
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
                prog_bar=prog_bar, logger=True, sync_dist=sync_dist
            )

            # move preds/targets to CPU once
            cpu_preds = [{k: v.cpu() for k, v in p.items()} for p in preds_attr]
            cpu_targs = [{k: v.cpu() for k, v in t.items()} for t in targs_attr]

            ap_small = self.compute_ap_for_area(cpu_preds, cpu_targs, max_area=32**2)
            precision, recall, f1 = self.compute_prf1(cpu_preds, cpu_targs, iou_threshold=0.5)

            # log the rest
            extra = {
                f"{prefix}AP_small": ap_small,
                f"{prefix}Precision_0.5": precision,
                f"{prefix}Recall_0.5": recall,
                f"{prefix}F1_0.5": f1,
            }
            for name, val in extra.items():
                self.log(name, val, prog_bar=(name.endswith("F1_0.5") and prog_bar),
                         logger=True, sync_dist=sync_dist)

        except Exception as e:
            log(f"Error computing {stage} metrics: {e}")
            zeros = {f"{prefix}{k}": 0.0 for k in
                     ["mAP_0.5", "mAP", "mAR_100", "AP_small",
                      "Precision_0.5", "Recall_0.5", "F1_0.5"]}
            self.log_dict(zeros, logger=True)

        finally:
            map_obj.reset()
            preds_attr.clear()
            targs_attr.clear()


    def on_train_start(self) -> None:
        self._train_input_validated = False

    def on_validation_start(self) -> None:
        if not self.trainer.sanity_checking:
            self._val_input_validated = False

    def on_test_start(self) -> None:
        self._test_input_validated = False

    def on_validation_epoch_end(self) -> None:
        if self.trainer.is_global_zero:  # only on GPU with rank:0
            self._calc_metrics('val')

        if (hasattr(self, 'logger') and
            self.logger is not None and
            isinstance(self.logger.experiment, SummaryWriter) and
            self.val_image_samples and
            self.num_log_images > 0 and
            not self.trainer.sanity_checking and
            self.trainer.is_global_zero):

            processed_images = []
            for i, (img, pred, target) in enumerate(zip(self.val_image_samples, self.val_pred_samples, self.val_target_samples)):
                if isinstance(img, list) and len(img) == 1 and isinstance(img[0], torch.Tensor):
                    img_tensor = img[0]  # the image comes as a list of one element
                elif isinstance(img, torch.Tensor):
                    img_tensor = img
                else:
                    log(f"Skipping image log for sample {i}: Unexpected item type or structure in val_image_samples: {type(img)}")
                    continue
                # step 1 - denorm and convert to uint8
                img_tensor_cpu = img_tensor.cpu()  # should alr. be on the cpu, but safety
                img_denorm = self._denormalize_image(img_tensor_cpu)
                img_uint8 = (img_denorm * 255).to(torch.uint8)

                if img_uint8.dim() == 4 and img_uint8.shape[0] == 1:  # remove batch dim if present
                    img_uint8 = img_uint8.squeeze(0)
                if img_uint8.dim() != 3:
                    continue

                # step 2: prep pred boxes and labels (filter by score)
                pred_cpu = {k: v.cpu() for k, v in pred.items()}
                target_cpu = {k: v.cpu() for k, v in target.items()}

                pred_scores = pred_cpu['scores']
                score_mask = pred_scores >= LOG_SCORE_THRESHOLD
                pred_boxes = pred_cpu['boxes'][score_mask]
                pred_labels_indices = pred_cpu['labels'][score_mask]
                pred_scores_filtered = pred_scores[score_mask]
                pred_labels = [f"{CLASSES[label_idx]}: {score:.2f}"
                               for label_idx, score in zip(pred_labels_indices, pred_scores_filtered)]

                # step 3: prep gt boxes and labels
                gt_boxes = target_cpu['boxes']
                gt_labels_indices = target_cpu['labels']
                gt_labels = [f"GT: {CLASSES[label_idx]}" for label_idx in gt_labels_indices]

                # 4 draw boxes
                img_with_boxes = img_uint8.clone()
                if len(gt_boxes) > 0:
                    try:
                        img_with_boxes = draw_bounding_boxes(img_with_boxes, gt_boxes, labels=gt_labels, colors=GT_COLOR, width=2)
                    except Exception as e:
                        log(f"Error drawing GT boxes for sample {i}: {e}. GT Boxes shape: {gt_boxes.shape}, Labels: {gt_labels}")


                if len(pred_boxes) > 0:
                    try:
                        img_with_boxes = draw_bounding_boxes(img_with_boxes, pred_boxes, labels=pred_labels, colors=PRED_COLOR, width=2)
                    except Exception as e:
                        log(f"Error drawing Pred boxes for sample {i}: {e}. Pred Boxes shape: {pred_boxes.shape}, Labels: {pred_labels}")
                processed_images.append(img_with_boxes)


            # 5. log all processed images at once using add_images
            if processed_images:
                try:
                    image_batch = torch.stack(processed_images)
                    self.logger.experiment.add_images(
                        'Validation Image Samples',
                        image_batch,
                        global_step=self.current_epoch, # use epoch for the slider step
                        dataformats='NCHW'
                    )
                except Exception as e:
                    log(f"Error logging image batch to TensorBoard for epoch {self.current_epoch}: {e}")


        self.val_image_samples.clear()
        self.val_pred_samples.clear()
        self.val_target_samples.clear()


    def on_test_epoch_end(self) -> None:
        if self.trainer.is_global_zero:
            self._calc_metrics('test')

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

        device = predictions[0]['boxes'].device if predictions and 'boxes' in predictions[0] and predictions[0][
            'boxes'].numel() > 0 else 'cpu'

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

            # greedy matching: iterate through predictions
            for i in range(num_pred_img):
                if overlaps.shape[1] == 0: continue  # no gts to match
                best_iou, best_gt_idx = overlaps[i].max(dim=0)
                if best_iou >= iou_threshold and not gt_matched[best_gt_idx]:
                    # tp += 1 # increment overall TP count
                    gt_matched[best_gt_idx] = True
                    pred_matched_to_gt[i] = True  # mark this prediction as a TP

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

        return float(precision), float(recall), float(f1)


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
