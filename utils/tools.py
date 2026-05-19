import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import math
import os
import torch.nn.functional as F

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.8 ** ((epoch - 3) // 1))}
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate / 10 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}  
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
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


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

def visualize_cycle(self, setting):
    save_path = os.path.join('./cycle_visualization/', setting)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # cycleQuery: [cycle_len, enc_in]
    cycle_query = self.model.cycleQuery.detach().cpu().numpy()

    plt.figure(figsize=(14, 8))
    plt.imshow(cycle_query.T, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    plt.colorbar(label='Query Value')
    plt.xlabel('Cycle Step', fontsize=12)
    plt.ylabel('Channel Index', fontsize=12)
    plt.title('Cycle Query Heatmap (All Channels)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'cycle_heatmap_all.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # for i in range(enc_in):
        #     plt.figure(figsize=(12, 4))
        #
        #     # 提取第 i 个变量的周期模式 [cycle_len]
        #     cycle_pattern = cycle_query[:, i]
        #
        #     # 绘制折线图
        #     plt.plot(range(cycle_len), cycle_pattern, linewidth=1.5, color='steelblue')
        #     plt.xlabel('Cycle Step', fontsize=12)
        #     plt.ylabel('Query Value', fontsize=12)
        #     plt.title(f'Cycle Pattern - channel {i}', fontsize=14)
        #     plt.grid(True, alpha=0.3, linestyle='--')
        #     plt.tight_layout()
        #
        #     file_path = os.path.join(save_path, f'cycle_channel{i}.png')
        #     plt.savefig(file_path, dpi=150, bbox_inches='tight')
        #     plt.close()

def visualize_cycle_weight(self, setting):
    save_path = os.path.join('./cycle_visualization/', setting)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    valid_fre_points = self.model.valid_fre_points
    freqs_tensor = torch.arange(valid_fre_points, device=next(self.model.parameters()).device).float()
    with torch.no_grad():
        threshold = F.softplus(self.model.freq_threshold).item()
        temp = self.model.temperature
        weights = torch.sigmoid((F.softplus(self.model.freq_threshold) - freqs_tensor) / temp)
        weights = weights.cpu().numpy()
    freqs = np.arange(valid_fre_points)

    plt.figure(figsize=(10, 6))
    plt.plot(freqs, weights, 'b-', linewidth=2, label='Cycle Weight')
    plt.axvline(threshold, color='r', linestyle='--', linewidth=2,
                label=f'Learned Threshold={threshold:.2f}')
    plt.fill_between(freqs, weights, alpha=0.3)
    plt.xlabel('Frequency Index', fontsize=12)
    plt.ylabel('Cycle Weight (0=Seq, 1=Cycle)', fontsize=12)
    plt.title('Learnable Frequency Fusion Weight', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'frequency_weights.png'), dpi=150, bbox_inches='tight')
    plt.close()