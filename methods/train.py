# train.py

# Python imports
import sys
import time


import numpy as np
from tqdm import tqdm

# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

# Local imports
from config import DEVICE
from util import log, to_numpy, to_device



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

def train_simple(train_loader, val_loader, model,
          num_epochs=100, learning_rate=1e-3, early_stop=50,
          verbosity=-1, use_pbar=True):
    """
    Main training logic for a given model, see README.md for more details.
    """
    # ---- Initialize variables of interest and history ----- #
    train_hist, val_hist = [], []

    # ----- Initialize the methods ------- #
    early_stopper = EarlyStopping(tolerance=early_stop, delta=0.001)  # initialize the early stopping logic
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # ADAM optimizer + weight decay

    # ----- Get the model ready ----- #
    model.to(DEVICE)  # pass it onto CUDA or CPU
    tqdm_pbar = tqdm(range(num_epochs), desc='Initializing', colour='green') if use_pbar else range(num_epochs)
    start_time = time.time()

    # -------- Training -------- #
    for epoch in range(num_epochs):
        model.train()
        train_loss, val_loss = 0.0, 0.0
        print(f'------- Epoch {epoch + 1} --------')
        for i, (X_batch, y_batch, meta) in enumerate(train_loader, 0):

            optimizer.zero_grad()

            y_pred = model(X_batch, y_batch)

            loss = model.loss_fn(y_pred)
            loss.backward()

            train_loss += loss.item()
            print(f'----- Pass {i} Loss {loss.item():.4f}')

        # ----- Validation ----- #
        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch, y_batch)
                loss = model.loss_fn(y_pred)
                val_loss += loss.item()
        model.train()

        # Get the losses & store
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_hist.append(train_loss)
        val_hist.append(val_loss)

        # -------- Printing --------- #
        if (verbosity > 0) and (epoch % verbosity == 0):
            log(f"Epoch [{epoch}/{num_epochs}] Train Loss: {train_loss:.8f}, Validation Loss: {val_loss:.8f}")

        if use_pbar:
            tqdm_pbar.set_description(
                f"Epoch [{epoch + 1}/{num_epochs}] Train Loss: {train_loss:.8f},"
                f" Val Loss: {val_loss:.8f}, Early Stop: {early_stopper.counter}")
            tqdm_pbar.update(1)

    return {
        'model_name': model.__class__.__name__, 'model': model,
        'train_hist': train_hist, 'val_hist': val_hist,
        'time_took': time.time() - start_time
    }


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
    scaler = GradScaler(device=DEVICE)  # CUDA automatic gradient scaling

    tqdm_pbar = tqdm(range(num_epochs), desc='Initializing', colour='green', file=sys.stdout) if use_pbar else range(num_epochs)
    start_time = time.time()


    # -------- Training -------- #
    for epoch in tqdm_pbar:
        model.train()
        train_loss, val_loss = 0.0, 0.0




        for i, (X_batch, y_batch, meta) in enumerate(train_loader):
            X_batch, y_batch = to_device(X_batch), to_device(y_batch)
            optimizer.zero_grad()  # main forward pass
            with autocast(device_type=str(DEVICE)):
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
                f"{'\033[34'}mEpoch [{epoch + 1}/{num_epochs}] Train Loss: {train_loss:.8f},"
                f" Val Loss: {val_loss:.8f}, Early Stop: {early_stopper.counter}{'\033[0m'}")
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



def predict(model, validation_tensor):
    with torch.no_grad():
        y_pred = model(validation_tensor.to(DEVICE))
        return np.array(to_numpy(y_pred))