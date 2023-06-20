import warnings
warnings.filterwarnings('ignore')

import os
import argparse
import random
from itertools import count, islice
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as ptf
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score

from dataset import Dataset
from model import MILModel


parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch-size', type = int, default = 4)
parser.add_argument('-t', '--train-steps', type = int, default = 1000)
parser.add_argument('-v', '--val-steps', type = int, default = 1000)
parser.add_argument('-x', '--epochs-no-gain', type = int, default = 20)
parser.add_argument('-s', '--seed', type = int, default = 42)


class EMA:
    
    def __init__(self, alpha, init = 0):
        self.m = init
        self.alpha = alpha
        
    def update(self, x):
        self.m = self.alpha*self.m + (1 - self.alpha)*x
        return self


if __name__ == "__main__":
    args = parser.parse_args()
    
    random.seed(args.seed + 1)
    np.random.seed(args.seed + 2)
    torch.manual_seed(args.seed + 3)
    
    os.makedirs('checkpoints/', exist_ok = True)
    
    train_dataset = Dataset(train = True)
    test_dataset = Dataset(train = False)
    
    # Skipped DataLoaders because of multiprocessing issues on Windows
    
    model = MILModel(janossy_samples = 4).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr = 2e-3, weight_decay = 1e-3)
    
    # metrics
    train_loss = EMA(alpha = 0.999, init = 0.69)
    inst_acc = EMA(alpha = 0.999, init = 0.5)
    bag_acc = EMA(alpha = 0.999, init = 0.5)

    best_score = 0
    best_epoch = 0
    for epoch in count():
        print(f"Epoch {epoch}:")

        # training
        model.train().requires_grad_(True)
        pbar = tqdm(range(args.train_steps)) 
        for batch in pbar:
            optimizer.zero_grad()
            for _ in range(args.batch_size):
                bag, target, idx = train_dataset[0]
                target = torch.tensor(target).float().cuda()

                embeddings, bag_prob, instance_probs = model(bag.cuda())
                
                bag_loss = ptf.binary_cross_entropy(bag_prob, target.cuda())
                instance_loss = ptf.binary_cross_entropy(instance_probs.max(), target.cuda())
                loss = (bag_loss + instance_loss) / args.batch_size
                loss.backward()
                
                # logging
                train_loss.update(float(loss.detach().cpu().numpy()))
                bag_acc.update((float(bag_prob.detach().cpu()) > 0.5) == target)
                mask = instance_probs.detach().cpu() > 0.5
                if mask.any():
                    idx_7 = idx[mask]
                    inst_labels = train_dataset.mnist.targets[idx_7]
                    inst_acc.update(float(torch.mean((inst_labels == 7).float())))

            optimizer.step()
            pbar.set_description(f"Train Loss: {train_loss.m:.6f}. Bag Acc: {bag_acc.m:.3f}. Inst. Acc: {inst_acc.m:.3f}")

        # validation
        y_true = []
        y_pred = []
        model.eval().requires_grad_(False)
        pbar = tqdm(range(args.val_steps)) 
        for _ in pbar:
            bag, target, _ = test_dataset[0]
            _, bag_prob, _ = model(bag.cuda())
            pred_label = float(bag_prob.detach().cpu()) > 0.5
            y_pred.append(pred_label)
            y_true.append(target)
            pbar.set_description(f"Val F1: {f1_score(y_true, y_pred):.3f}. Val Acc: {accuracy_score(y_true, y_pred):.3f}.")

        # checkpointing
        score = f1_score(y_true, y_pred)
        if score > best_score:
            torch.save(model.state_dict(), f'checkpoints/ckpt_best.pth')
            best_score = score
            best_epoch = epoch

        if epoch - best_epoch > args.epochs_no_gain:
            break

    print(f"Finished. Best F1-score: {best_score:.3f}")