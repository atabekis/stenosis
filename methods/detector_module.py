# detector_module.py

# Python imports
import os
import math
import numpy as np
from typing import Optional, Union, Any

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
from pytorch_lightning.loggers import TensorBoardLogger

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
LOG_SCORE_THRESHOLD = 0.0 # For the TensorBoard logger, boxes with confidence score > will be put on the image


class DetectionLightningModule(pl.LightningModule):
    """
    Pytorch Lightning Module for stenosis detection models

    Handles training, validation, testing loops, optimization, logging, and metric calculation
    """

    VALID_METRIC_SUFFIXES = {  # these are the final metrics calculated in val/test steps.
        "mAP_0.5", "mAP", "mAR_100",
        "AvgIoU_TP_0.5", "AP_small",
        "Precision_0.5", "Recall_0.5", "F1_0.5"
    }

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

            use_scheduler: bool = True,

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

            # hparam logging/tuning
            hparams_to_log: Optional[dict] = None,
            hparam_primary_metric: str = 'F1_0.5',
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
        self.use_scheduler = use_scheduler
        self.stem_learning_rate = stem_learning_rate
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.smooth_l1_beta = smooth_l1_beta
        self.giou_loss_coef = giou_loss_coef
        self.cls_loss_coef = cls_loss_coef
        self.positive_class_id = positive_class_id
        self.hparam_primary_metric = hparam_primary_metric


        if model_stage not in [1, 2, 3]:
            raise ValueError(f'Invalid model_stage: {model_stage}. Must be in [1, 2, 3]')

        if self.hparam_primary_metric not in self.VALID_METRIC_SUFFIXES:
            raise ValueError(f'hparam_primary_metric_name ({self.hparam_primary_metric}) is not in the set of allowed'
                             f'metric suffixes: {self.VALID_METRIC_SUFFIXES}')

        self._hparams_to_log = hparams_to_log or {}
        self.save_hyperparameters(ignore=['model', 'normalize_params'])


        # ------ initialize metrics -------
        common_map_args = {
            "iou_type": "bbox", "iou_thresholds": [0.5],
            "rec_thresholds": torch.linspace(0, 1, 101).tolist(),
            "class_metrics": False,
        }

        self.val_map = MeanAveragePrecision(**common_map_args)
        self.test_map = MeanAveragePrecision(**common_map_args)

        self.val_preds_cpu, self.val_targets_cpu = [], []
        self.test_preds_cpu, self.test_targets_cpu = [], []


        # ----- redundancy, input validation ------
        self._train_input_validated = False
        self._val_input_validated = False
        self._test_input_validated = False


        # ----- tensorboard/logging ------
        self.normalize_params = normalize_params
        self.num_log_val_images = num_log_val_images
        self.num_log_test_images = num_log_test_images

        self.val_image_samples = []
        self.val_pred_samples = []   # stores list[dict] or list[list[dict]]
        self.val_target_samples = []
        self.val_mask_samples = []

        self.test_image_samples = []
        self.test_pred_samples = []
        self.test_target_samples = []
        self.test_mask_samples = []

        self.final_test_metrics = {}

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
            targets: Optional[Union[list[dict[str, torch.Tensor]], list[list[dict[str, torch.Tensor]]]]] = None,
            masks: Optional[torch.Tensor] = None,  # might be deprecated, will see with transformer
    ) -> Any:
        """
        Forward pass of the model. Input format depends on the model stage
        :param images: input images
            Stage 1: Expected list[tensor[C, H, W]]
            Stage 2/3: Expected [B, T, C, H, W]
        :param targets: expected targets
            Stage 1: Expects list[dict{boxes, labels}]
            Stage 2/3: Expects list[list[dict{boxes, labels}]]
        :return: Training dictionary of losses
            Stage 1: expected list[list[dict{boxes, scores, labels}]]
            Stage 2/3: expected list[list[dict{boxes, scores, labels}]], per-frame predictions.
        """

        return self.model(images, targets)


    def _validate_input_shape(self, images: Union[list[torch.Tensor], torch.Tensor], masks: torch.Tensor) -> None:
        """Checks input shape at initialization per stage (train/val/test)."""
        if self.model_stage == 1:
            if isinstance(images, list):
                assert images and torch.is_tensor(images[0]) and images[0].dim() == 3, (
                    f"Stage 1 expected list of [C,H,W] tensors, got {type(images[0])} with ndim "
                    f"{getattr(images[0], 'dim', lambda: None)()}"
                )
            elif torch.is_tensor(images):
                assert images.dim() == 4, f"Stage 1 expected Tensor[B,C,H,W], got ndim={images.dim()}"
            else:
                raise AssertionError(f"Stage 1 unexpected images type: {type(images)}")
            return

        elif self.model_stage in [2, 3]:
            assert torch.is_tensor(images) and images.dim() == 5, (
                f"Stage {self.model_stage} expected Tensor[B,T,C,H,W], got {type(images)} ndim="
                f"{getattr(images, 'dim', lambda: None)()}"
            )
            batch, t = images.size(0), images.size(1)

            assert masks is not None, "Mask is required for Stage 2/3 but was None"
            assert torch.is_tensor(masks) and masks.dim() == 2, (
                f"Stage {self.model_stage} expected mask Tensor[B,T], got {type(masks)} ndim="
                f"{getattr(masks, 'dim', lambda: None)()}"
            )
            assert masks.size(0) == batch and masks.size(1) == t, (
                f"Mask shape {tuple(masks.shape)} does not match images shape [B={batch},T={t}]"
            )
            return


    def _step_logic(self, batch, batch_idx, step_type):
        images, targets, masks, metadata = batch
        batch_size = len(images)

        is_val = (step_type == 'val')

        if self.model_stage == 1:
            images = images.squeeze(1)
            targets = [t[0] for t in targets if t]

        # --- 1. input shape validation (once) ---
        if step_type == 'train' and not self._train_input_validated:
            self._validate_input_shape(images, masks=masks)
            self._train_input_validated = True
        elif step_type == 'val' and not self._val_input_validated:
            if not self.trainer.sanity_checking:
                self._validate_input_shape(images, masks=masks)
            self._val_input_validated = True
        elif step_type == 'test' and not self._test_input_validated:
            self._validate_input_shape(images, masks=masks)
            self._test_input_validated = True

        # 2. Training step refactored to training_step()
        if step_type == 'train':
            return

        # 3. forward pass (for val & test)
        original_train_state_pred = self.model.training
        self.model.eval()
        with torch.no_grad():
            preds = self.forward(images, targets=None, masks=masks)
        self.model.train(original_train_state_pred)

        # 4. flatten preds/targets according to mask
        flat_mask = masks.view(-1)
        metric_preds, metric_tgts = [], []

        if self.model_stage == 1:
            # preds, targets are length-B lists
            for i, p in enumerate(preds):
                if i < flat_mask.size(0) and flat_mask[i]:
                    metric_preds.append(p)
                    metric_tgts.append(targets[i])

        else:
            # preds, targets are lists of lists [B, T]
            flat_tgts = [frame_tgt for video_tgts in targets for frame_tgt in video_tgts]

            if len(preds) != flat_mask.size(0) or len(flat_tgts) != flat_mask.size(0):
                log(f'WARNING: Shape mismatch in _step_logic. Preds len: {len(preds)}, Flat Tgts len: {len(flat_tgts)}, Flat Mask len: {flat_mask.size(0)}. Skipping batch for metrics.')
                return

            for i, keep_frame in enumerate(flat_mask):
                if keep_frame:
                    metric_preds.append(preds[i])
                    metric_tgts.append(flat_tgts[i])

        # 5. update metrics
        metric = self.val_map if is_val else self.test_map
        if metric_preds and metric_tgts:
            dev = metric.device
            preds_dev = [{k: v.to(dev) for k, v in p.items()} for p in metric_preds]
            tgts_dev = [{k: v.to(dev) for k, v in t.items()} for t in metric_tgts]
            metric.update(preds_dev, tgts_dev)

        # 6. store cpu copies for logging/metrics
        pred_store = self.val_preds_cpu if is_val else self.test_preds_cpu
        tgt_store = self.val_targets_cpu if is_val else self.test_targets_cpu

        pred_store.extend([{k: v.cpu().detach() for k, v in p.items()} for p in metric_preds])
        tgt_store.extend([{k: v.cpu().detach() for k, v in t.items()} for t in metric_tgts])

        # 7. store samples for logging
        img_store = self.val_image_samples if is_val else self.test_image_samples
        pred_s_store = self.val_pred_samples if is_val else self.test_pred_samples
        tgt_s_store = self.val_target_samples if is_val else self.test_target_samples
        mask_s_store = self.val_mask_samples if is_val else self.test_mask_samples
        max_log = self.num_log_val_images if is_val else self.num_log_test_images

        should_log = (
                (is_val and not self.trainer.sanity_checking and max_log > 0)
                or (not is_val and max_log != 0)
        )
        stored = len(img_store)
        limit = float('inf') if (not is_val and max_log == 'all') else max_log

        if should_log and stored < limit:
            for i in range(int(min((limit - stored), batch_size))):
                img_store.append(images[i].cpu().detach())

                if self.model_stage == 1:
                    pred_s_store.append({k: v.cpu().detach() for k, v in preds[i].items()})
                    tgt_s_store.append({k: v.cpu().detach() for k, v in targets[i].items()})

                else:
                    video_preds_for_log = []
                    current_pred_idx = 0
                    for b_idx in range(batch_size):
                        frames_in_this_video = images.size(1)
                        video_pred_sequence = []
                        for t_idx in range(frames_in_this_video):
                            if current_pred_idx < len(preds):
                                video_pred_sequence.append({k: v.cpu().detach() for k, v in preds[current_pred_idx].items()})
                            else:
                                video_pred_sequence.append({'boxes': torch.empty(0,4), 'scores':torch.empty(0), 'labels':torch.empty(0, dtype=torch.long)})

                            current_pred_idx += 1
                        video_preds_for_log.append(video_pred_sequence)

                    pred_s_store.append(video_preds_for_log[i])
                    tgt_s_store.append([{k: v.cpu().detach() for k, v in frame_target.items()} for frame_target in targets[i]])

                mask_s_store.append(masks[i].cpu().detach())

        # 7. val loss logging
        if is_val:
            orig_training_state = self.model.training
            self.model.train()

            # with torch.no_grad():
            loss_dict = self.forward(images, targets=targets)

            self.model.train(orig_training_state)

            valid_losses = [v for v in loss_dict.values() if isinstance(v, torch.Tensor) and not torch.isnan(v) and not torch.isinf(v)]
            total_loss = sum(valid_losses) if valid_losses else torch.tensor(0.0, device=self.device)

            self.log('val_loss', total_loss,
                     on_step=False, on_epoch=True, prog_bar=True,
                     logger=True, sync_dist=True, batch_size=batch_size)

            for name, val in loss_dict.items():
                if isinstance(val, torch.Tensor) and not torch.isnan(val) and not torch.isinf(val):
                    self.log(f'val/{name}', val,
                             on_step=False, on_epoch=True, prog_bar=False,
                             logger=True, sync_dist=True, batch_size=batch_size)


    def training_step(self, batch, batch_idx):
        """The training step needs to explicitly return the losses for logging"""
        images, targets, masks, _ = batch

        # # TODO: REMOVE DEBUG
        # if self.global_step < 2:  # Log only for the first 2 global steps
        #     log(f"--- DEBUG: training_step global_step {self.global_step}, batch_idx {batch_idx} ---")
        #     log(f"Raw images_orig shape: {images.shape}, dtype: {images.dtype}")
        #     log(f"Raw images_orig stats: min={images.min():.3f}, max={images.max():.3f}, mean={images.mean():.3f}, std={images.std():.3f}")
        #     log(f"Raw masks_orig shape: {masks.shape}, dtype: {masks.dtype}")
        #     if targets and targets[0]:  # Check if targets_orig and its first element exist
        #         log(f"Raw targets_orig[0] sample (first item in batch): {targets[0]}")
        #     else:
        #         log(f"Raw targets_orig is empty or first element is empty.")
        # # TODO: REMOVE DEBUG


        assert targets is not None, "Targets must be provided during training"

        # 1. forward to get losses
        if self.model_stage == 1:
            images = images.squeeze(1) # [B, C, H, W]
            targets = [t[0] for t in targets if t] # list[dict]

        # # TODO: REMOVE DEBUG
        # if self.global_step < 2 and self.model_stage == 1:
        #     log(f"Processed images_for_model (Stage 1) shape: {images.shape}, dtype: {images.dtype}")
        #     if targets:
        #         log(f"Processed targets_for_model[0] (Stage 1) sample: {targets[0]}")
        #     else:
        #         log(f"Processed targets_for_model (Stage 1) is empty.")
        # # TODO: REMOVE DEBUG



        loss_dict = self.forward(images, targets=targets)

        # 2. sum tensor losses that require grad
        if isinstance(loss_dict, dict):
            valid = [l for l in loss_dict.values() if torch.is_tensor(l) and l.requires_grad and not torch.isnan(l) and not torch.isinf(l)]
            total_loss = sum(valid) if valid else torch.tensor(0.0, device=self.device, requires_grad=True)
        else:
            total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            loss_dict = {}

        # 3. logging
        batch_size = images.size(0) if torch.is_tensor(images) else len(images)
        self.log(
            'train_loss', total_loss,
            on_step=True, on_epoch=True, prog_bar=True,
            logger=True, sync_dist=True, batch_size=batch_size
        )
        # 3a. individual losses
        for name, loss in loss_dict.items():
            if torch.is_tensor(loss):
                self.log(
                    f'train/{name}', loss,
                    on_step=True, on_epoch=True, prog_bar=False,
                    logger=True, sync_dist=True, batch_size=batch_size
                )

        # 3b. learning rate
        scheduler = self.lr_schedulers()
        current_lr = 0.0
        if scheduler:
            current_lr = scheduler.get_last_lr()[0]
        self.log('train/lr', current_lr, on_step=True, on_epoch=False, logger=True, batch_size=batch_size)

        return total_loss


    def validation_step(self, batch, batch_idx):
        self._step_logic(batch, batch_idx, 'val')
        return 0


    def test_step(self, batch, batch_idx):
        self._step_logic(batch, batch_idx, 'test')
        return 0


    def _calc_metrics(self, stage: str) -> dict[Any, Any]:
        """
        Compute, log, and reset metrics for either 'val' or 'test' stage.
        """
        map_obj = getattr(self, f"{stage}_map")
        preds_attr = getattr(self, f"{stage}_preds_cpu")
        targs_attr = getattr(self, f"{stage}_targets_cpu")
        prefix = f"{stage}/"
        # prog_bar + sync_dist only for validation
        prog_bar = stage == "val"
        sync_dist = stage == "val"

        metrics_dict = {}

        if not preds_attr or not targs_attr:
            log(f"Skipping {stage} metrics calculation: No predictions or targets collected.")
            # log zeros to prevent crashing later
            zeros = {f"{prefix}{k}": 0.0 for k in
                     ["mAP_0.5", "mAP", "mAR_100", "AP_small",
                      "Precision_0.5", "Recall_0.5", "F1_0.5", "AvgIoU_TP_0.5"]}
            self.log_dict(zeros, logger=True, batch_size=1, sync_dist=sync_dist)
            map_obj.reset(); preds_attr.clear(); targs_attr.clear()
            return zeros

        try:
            # compute mAP/mAR
            metrics = map_obj.compute()
            to_num = lambda x: x.item() if isinstance(x, torch.Tensor) else x
            m50 = to_num(metrics["map_50"])
            m_all = to_num(metrics["map"])
            mar100 = to_num(metrics["mar_100"])

            map_metrics = {
                 f"{prefix}mAP_0.5": m50,
                 f"{prefix}mAP": m_all,
                 f"{prefix}mAR_100": mar100
            }

            self.log_dict(map_metrics, prog_bar=prog_bar, logger=True, sync_dist=sync_dist, batch_size=1)
            metrics_dict.update(map_metrics)

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
            self.log_dict(extra, logger=True, sync_dist=sync_dist, batch_size=1)
            metrics_dict.update(extra)

        except Exception as e:
            log(f"Error computing {stage} metrics: {e}")
            zeros = {f"{prefix}{k}": 0.0 for k in
                     ["mAP_0.5", "mAP", "mAR_100", "AP_small",
                      "Precision_0.5", "Recall_0.5", "F1_0.5", "AvgIoU_TP_0.5"]}
            self.log_dict(zeros, logger=True, batch_size=1)
            metrics_dict.update(zeros)

        finally:
            map_obj.reset()
            preds_attr.clear()
            targs_attr.clear()

        return metrics_dict


    def on_train_start(self) -> None:
        self._train_input_validated = False


    def on_validation_start(self) -> None:
        if not self.trainer.sanity_checking:
            self._val_input_validated = False

            self.val_preds_cpu.clear()
            self.val_targets_cpu.clear()
            self.val_image_samples.clear()
            self.val_pred_samples.clear()
            self.val_target_samples.clear()
            self.val_mask_samples.clear()

            self.val_map.reset()


    def on_test_start(self) -> None:
        self.test_preds_cpu.clear()
        self.test_targets_cpu.clear()
        self.test_image_samples.clear()
        self.test_pred_samples.clear()
        self.test_target_samples.clear()
        self.test_mask_samples.clear()

        self.test_map.reset()


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
                mask_samples=self.val_mask_samples,
                writer=self.logger.experiment,
                tag='Validation Image Samples',
                global_step=self.current_epoch
            )

        self.val_image_samples.clear()
        self.val_pred_samples.clear()
        self.val_target_samples.clear()
        self.val_mask_samples.clear()


    def on_test_epoch_end(self) -> None:
        if self.trainer.is_global_zero:
            final_metrics = self._calc_metrics('test')
            self.final_test_metrics = final_metrics

            logger = self.logger
            if isinstance(logger, TensorBoardLogger):
                ckpt = self.trainer.checkpoint_callback
                best_score = getattr(ckpt, 'best_model_score', None)
                monitor = getattr(ckpt, 'monitor', 'val_metric').replace('/', '_')

                # prepare HParam metrics
                hp_metrics_to_log = {} # Use a distinct name for clarity
                if best_score is not None:
                    hp_metrics_to_log[f"hp/best_{monitor}"] = best_score.item()


                for k, v in final_metrics.items():
                    if isinstance(v, (int, float, torch.Tensor)):
                        metric_value = v.item() if torch.is_tensor(v) else v
                        hp_metrics_to_log[f"hp/{k.replace('/', '_')}"] = metric_value


                # determine the value for the hp metric (metric from self.hparam_primary_metric)
                primary_metric_val = None
                key_in_final_metrics = f'test/{self.hparam_primary_metric}'

                if key_in_final_metrics in final_metrics:
                    primary_metric_val = final_metrics[key_in_final_metrics]
                    primary_metric_val = primary_metric_val.item() if torch.is_tensor(primary_metric_val) else float(primary_metric_val)
                    hp_metrics_to_log["hp_metric"] = primary_metric_val
                else:  # will remove this later after adding sanity checking for hparams in __init__
                    log(f"Warning: Specified hparam_primary_metric_name '{self.hparam_primary_metric_name}' "
                        f"(expected key: '{key_in_final_metrics}') not found in final_metrics. "
                        f"Available keys: {list(final_metrics.keys())}. "
                        f"Setting 'hp/hp_metric' to 0.0 as a fallback.")
                    hp_metrics_to_log["hp_metric"] = 0.0 # fallback

                params_for_log = self.hparams.copy()
                logger.log_hyperparams(params_for_log, hp_metrics_to_log)
                logger.save()


            if self.num_log_test_images and getattr(logger, 'experiment', None):
                 self._log_image_samples_to_tensorboard(
                    image_samples=self.test_image_samples,
                    pred_samples=self.test_pred_samples,
                    target_samples=self.test_target_samples,
                    mask_samples=self.test_mask_samples,
                    writer=logger.experiment,
                    tag="Test Image Samples",
                    global_step=self.current_epoch
                )

        # clear buffers
        self.test_image_samples.clear()
        self.test_pred_samples.clear()
        self.test_target_samples.clear()
        self.test_mask_samples.clear()


    def configure_optimizers(self):
        """
        Configures the optimizer (AdamW) and lr scheduler (linear warmup + cosine ann)
        Handles differential lr for stage 3
        """
        parameters = []
        if self.model_stage == 3:
            log(f'Configuring optimizer for stage {self.model_stage} with differential LRs:')
            log(f'   Backbone/Stem LR: {self.stem_learning_rate}')
            log(f'   Main/Transformer/Head LR: {self.learning_rate}')

            backbone_params, other_params = [], []

            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if name.startswith('backbone'):
                    backbone_params.append(param)
                else:
                    other_params.append(param)

            if not backbone_params:
                log("Warning: No parameters found for 'backbone' prefix. Differential LR might not work as expected.")
            if not other_params:
                log("Warning: No parameters found for non-backbone parts. Check model structure.")

            parameters_to_opt = [
                {'params': backbone_params, 'lr': self.stem_learning_rate, 'name': 'backbone'},
                {'params': other_params, 'lr': self.learning_rate, 'name': 'transformer_head'}
            ]
            parameters = [p for p in parameters_to_opt if p['params']]



        else:
            log(f'Configuring optimizer for stage {self.model_stage} with single LR: {self.learning_rate}')
            parameters = [p for p in self.model.parameters() if p.requires_grad]

        # ---- optimizer -----
        optimizer = optim.AdamW(
            parameters,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        if not self.use_scheduler:
            log("Learning rate scheduler is DISABLED by 'use_scheduler=False'.")
            return optimizer

        # ---- scheduler calculation -----
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
                f" Scheduler might not behave as expected. Check max_epochs, dataset size, batch size, accumulation")
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
            mask_samples: list[torch.Tensor],
            writer: SummaryWriter,
            tag: str,
            global_step: int
    ):
        """Helper function to process and log image samples with boxes to TensorBoard."""
        if not image_samples:
            return

        to_log = []
        for img, pred, tgt_list, mask in zip(image_samples, pred_samples, target_samples, mask_samples):

            # select one real frame
            if self.model_stage == 1:
                if not mask[0]:
                    continue
                frame = img
                pred_frame = pred
                tgt_frame = tgt_list
            else:
                real_idxs = mask.nonzero(as_tuple=True)[0]
                if not real_idxs.numel():
                    continue
                idx = real_idxs[real_idxs.numel() // 2].item()
                frame = img[idx]
                pred_frame = pred[idx] if isinstance(pred, list) and len(pred) > idx else None
                tgt_frame = tgt_list[idx] if len(tgt_list) > idx else None

            if frame is None or pred_frame is None or tgt_frame is None:
                continue

            # denorm to uint8 3×H×W
            img_dn = (self._denormalize_image(frame).clamp(0, 1) * 255).to(torch.uint8)
            if img_dn.size(0) == 1:
                img_dn = img_dn.repeat(3, 1, 1)
            if img_dn.size(0) != 3:
                continue

            # gt boxes/text
            gt_boxes = tgt_frame.get('boxes', torch.zeros((0, 4)))
            gt_labels = tgt_frame.get('labels', torch.zeros(0, dtype=torch.int64)).tolist()
            gt_texts = [f"GT:{CLASSES[l]}" for l in gt_labels if 0 <= l < len(CLASSES)]

            # pred boxes/text (single box only)
            pboxes = pred_frame.get('boxes', torch.zeros((0, 4)))
            pscores = pred_frame.get('scores', torch.zeros(0))
            plabels = pred_frame.get('labels', torch.zeros(0, dtype=torch.int64))
            if pboxes.numel() and pscores[0] >= LOG_SCORE_THRESHOLD:
                pboxes = pboxes[:1]
                pl = plabels[0].item()
                pt = f"{CLASSES[pl]}:{pscores[0].item():.2f}"
                pred_boxes = pboxes
                pred_texts = [pt]
            else:
                pred_boxes = torch.zeros((0, 4))
                pred_texts = []

            # draw boxes
            canvas = img_dn.clone()
            if gt_boxes.numel():
                canvas = draw_bounding_boxes(canvas, gt_boxes, labels=gt_texts, colors=GT_COLOR, width=2)
            if pred_boxes.numel():
                canvas = draw_bounding_boxes(canvas, pred_boxes, labels=pred_texts, colors=PRED_COLOR, width=2)

            to_log.append(canvas)

        if to_log:
            batch = torch.stack(to_log, dim=0)  # (N, 3, H, W)
            writer.add_images(tag, batch, global_step, dataformats='NCHW')


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


