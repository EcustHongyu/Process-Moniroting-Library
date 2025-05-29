import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math
import seaborn as sns

plt.switch_backend('agg')

def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss

def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def find_kde(input, confidence=0.95):
    """
    KDE nonparametric estimation computes control limit function
    :param1 data_original: one-dimensional ndarray data
    :param2 data_original: The parameter Confidence is the confidence level, which is generally above 0.9
    :return: Specific value of control limit
    """
    plt.figure()
    ax = sns.kdeplot(input, cumulative=True)
    # experiments have proven that the types of the kernel parameters is not important, which have less effect on the kde estimation
    # the kernel parameter is Gaussian by default, and non-Gaussian kernel is no longer supported in versions after 0.11.0
    # therefore, the following codes is no longer valid
    # ax = sns.kdeplot(input, kernel='Gaussian', bw=((input.max() - input.min()) / 1000))
    # ax = sns.kdeplot(input, cumulative=True, kernel='gau'
    #                  , bw=((input.max() - input.min()) / 1000))
    line = ax.lines[0]
    x1, y1 = line.get_data()
    for i in range(len(y1)):
        if y1[i] > confidence:
            # control_limit = x1[i - 1] + (x1[i] - x1[i - 1]) * (y1[i] - Confidence) / (y1[i] - y1[i - 1])
            control_limit = x1[i]
            break
    plt.close()
    return control_limit