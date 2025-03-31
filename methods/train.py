# train.py

# Python imports
import time


import numpy as np
from tqdm import tqdm

# Torch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

if torch.__version__ < '2.6.0':  # so that these two works with the HPC
    from torch.cuda.amp import GradScaler, autocast
    version_flag = True  # we will need to set for GradScaler and autocast in the functions
else:
    from torch.amp import GradScaler, autocast
    version_flag = False


# Local imports
from config import DEVICE
from util import log, to_numpy, to_device

from HPC_config import get_local_rank, get_world_size, get_rank, get_gpus_per_node



class EarlyStopping:
    def __init__(self, tolerance=30, delta=0.001):
        self.tolerance = tolerance  # halts the training if the val_loss hasn't improved after the tolerance
        self.delta = delta  # minimum change in the loss to qualify as an improvement
        self.counter = 0
        self.early_stop = False
        self.best_score = None

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif (self.tolerance > 0) and (score < self.best_score + self.delta):  # when tolerance is negative early stopping is turned off
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def train(train_loader, val_loader, model,
          num_epochs=100, learning_rate=1e-3, early_stop=50,
          verbosity=-1, use_pbar=True):
    """
    Main training logic for a given model, see README.md for more details.
    """

    # ---- Initialize variables of interest and history ----- #
    train_hist, val_hist = [], []
    best_loss, best_weights = np.inf, None

    # ----- Initialize the methods ------- #
    early_stopper = EarlyStopping(tolerance=early_stop, delta=0.001)  # initialize the early stopping logic
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # ADAM optimizer + weight decay
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.01, patience=10, min_lr=1e-6)  # actively updating the learning rate


    # ----- Get the model ready ----- #
    model.to(DEVICE)  # pass it onto CUDA or CPU
    scaler = GradScaler(device=DEVICE if version_flag else None)  # CUDA automatic gradient scaling

    tqdm_pbar = tqdm(range(num_epochs), desc='Initializing', colour='green') if use_pbar else range(num_epochs)
    start_time = time.time()


    # -------- Training -------- #
    for epoch in tqdm_pbar:
        model.train()
        train_loss, val_loss = 0.0, 0.0

        for i, (X_batch, y_batch, meta) in enumerate(train_loader):
            X_batch, y_batch = to_device(X_batch), to_device(y_batch)
            optimizer.zero_grad()  # main forward pass
            with autocast(device_type=str(DEVICE) if version_flag else None):
                y_pred = model(X_batch, y_batch)
                # print(y_pred)
                loss = model.loss_fn(y_pred)
            scaler.scale(loss).backward()  # backward pass
            scaler.step(optimizer); scaler.update()

            train_loss += loss.item()

        # ----- Validation ----- #
        # model.eval()
        with torch.no_grad():
            for X_batch, y_batch, meta in val_loader:
                X_batch, y_batch = to_device(X_batch), to_device(y_batch)

                y_pred = model(X_batch, y_batch)
                # print(y_pred)
                loss = model.loss_fn(y_pred)
                val_loss += loss.item()
        # model.train()

        # Get the losses & store
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_hist.append(train_loss)
        val_hist.append(val_loss)

        scheduler.step(val_loss)  # update learning rate

        # ------- EarlyStopping -------#
        if val_loss < best_loss:  # getting the best model alongside early stopping
            best_loss = val_loss
            best_weights = model.state_dict()

        early_stopper(val_loss)
        if early_stopper.early_stop:
            log(f'Early stopping at epoch {epoch}, validation loss: {val_loss:.8f}')
            break

        # -------- Printing --------- #
        if (verbosity > 0) and (epoch % verbosity == 0):
            log(f"Epoch [{epoch}/{num_epochs}] Train Loss: {train_loss:.8f}, Validation Loss: {val_loss:.8f}")

        if use_pbar:
            tqdm_pbar.set_description(
                f"Epoch [{epoch + 1}/{num_epochs}] Train Loss: {train_loss:.8f},"
                f" Val Loss: {val_loss:.8f}, Early Stop: {early_stopper.counter}")
            tqdm_pbar.update(1)

    # ------- Returning best model --------
    if best_weights is not None:
        model.load_state_dict(best_weights)
        if use_pbar and (verbosity < 0):
            log(f'Returning the best model with validation loss: {best_loss:.4f}')

    # return (best) model parameters alongside train & val history
    return {
        'model_name': model.__class__.__name__, 'model': model,
        'train_hist': train_hist, 'val_hist': val_hist,
        'time_took': time.time() - start_time
    }



def train_hpc(train_loader, val_loader, model,
              num_epochs=100, learning_rate=1e-3, early_stop=50,
              verbosity=-1, use_pbar=True):
    world_size, rank, gpus_per_node = get_world_size(), get_rank(), get_gpus_per_node()

    if rank == 0:
        log(f"Is group initialized? {dist.is_initialized()}")

    local_rank = rank - gpus_per_node * (rank // gpus_per_node)

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')

    model = model.to(local_rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])


    # ---- Initialize variables of interest and history ----- #
    train_hist, val_hist = [], []
    best_loss, best_weights = np.inf, None

    # ----- Initialize the methods ------- #
    early_stopper = EarlyStopping(tolerance=early_stop, delta=0.001)  # initialize the early stopping logic
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # ADAM optimizer + weight decay
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.01, patience=10, min_lr=1e-6)  # actively updating the learning rate
    scaler = GradScaler(device=DEVICE if version_flag else None)  # CUDA automatic gradient scaling

    sampler = train_loader.sampler if hasattr(train_loader, 'sampler') else None

    tqdm_pbar = tqdm(range(num_epochs), desc='Initializing', colour='green') if use_pbar else range(num_epochs)
    start_time = time.time()

    device = torch.device(f"cuda:{local_rank}")

    # -------- Training -------- #
    for epoch in tqdm_pbar:
        model.train()
        train_loss, val_loss = 0.0, 0.0
        if sampler is not None:
            sampler.set_epoch(epoch)
            for i, (X_batch, y_batch, meta) in enumerate(train_loader):
                X_batch, y_batch = to_device(X_batch, device=device), to_device(y_batch, device=device)
                optimizer.zero_grad()  # main forward pass
                with autocast(device_type=str(DEVICE) if version_flag else None):
                    y_pred = model(X_batch, y_batch)
                    # print(y_pred)
                    loss = model.loss_fn(y_pred)
                scaler.scale(loss).backward()  # backward pass
                scaler.step(optimizer); scaler.update()

                train_loss += loss.item()

            # ----- Validation ----- #
            # model.eval()
            with torch.no_grad():
                for X_batch, y_batch, meta in val_loader:
                    X_batch, y_batch = to_device(X_batch), to_device(y_batch)

                    y_pred = model(X_batch, y_batch)
                    # print(y_pred)
                    loss = model.loss_fn(y_pred)
                    val_loss += loss.item()
            # model.train()

            # Get the losses & store
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            train_hist.append(train_loss)
            val_hist.append(val_loss)

            scheduler.step(val_loss)  # update learning rate

            # ------- EarlyStopping -------#
            if val_loss < best_loss:  # getting the best model alongside early stopping
                best_loss = val_loss
                best_weights = model.state_dict()

            early_stopper(val_loss)
            if early_stopper.early_stop:
                log(f'Early stopping at epoch {epoch}, validation loss: {val_loss:.8f}')
                break

            # -------- Printing --------- #
            if (verbosity > 0) and (epoch % verbosity == 0) and rank == 0:
                log(f"Epoch [{epoch}/{num_epochs}] Train Loss: {train_loss:.8f}, Validation Loss: {val_loss:.8f}")

            if use_pbar:
                tqdm_pbar.set_description(
                    f"Epoch [{epoch + 1}/{num_epochs}] Train Loss: {train_loss:.8f},"
                    f" Val Loss: {val_loss:.8f}, Early Stop: {early_stopper.counter}")
                tqdm_pbar.update(1)

        # ------- Returning best model --------
        if best_weights is not None:
            model.load_state_dict(best_weights)
            if use_pbar and (verbosity < 0):
                log(f'Returning the best model with validation loss: {best_loss:.4f}')

        dist.destroy_process_group()

        # return (best) model parameters alongside train & val history
        return {
            'model_name': model.__class__.__name__, 'model': model,
            'train_hist': train_hist, 'val_hist': val_hist,
            'time_took': time.time() - start_time
        }






def predict(model, validation_tensor):
    with torch.no_grad():
        y_pred = model(validation_tensor.to(DEVICE))
        return np.array(to_numpy(y_pred))