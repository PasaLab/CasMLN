import json
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
import torch
import os.path as osp
import numpy as np
from cmln.utils import EarlyStopping
from tqdm import tqdm
import time
from torch import nn
from cmln.utils import setup_seed
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D


def train(model, optimizer, criterion, train_data, culmulate=1, grad_clip=0):
    model.train()

    for support, query in train_data:
        z = model.encode(support)
        edge_label_index = query.edge_label_index
        edge_label = query.edge_label
        out = model.decode(z, edge_label_index).view(-1)
        loss = criterion(out, edge_label)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data):
    def test_one(model, data):
        support, query = data
        model.eval()
        z = model.encode(support)
        out = model.decode(z, query.edge_label_index).view(-1).sigmoid()
        target = query.edge_label.cpu().numpy()
        preds = out.cpu().numpy()
        return roc_auc_score(target, preds), average_precision_score(target, preds)

    if isinstance(data, list):
        aucs = []
        aps = []
        for d in data:
            auc, ap = test_one(model, d)
            aucs.append(auc)
            aps.append(ap)
        
        return np.mean(aucs), np.mean(aps)
    return test_one(model, data)

def train_till_end(
    model,
    optimizer,
    criterion,
    dataset,
    args,
    max_epochs,
    patience,
    disable_progress=False,
    writer=None,
    grad_clip=0,
    device=None,
):
    # procedure
    setup_seed(args.seed)
    start_time = time.time()
    best_val_auc = final_test_auc = 0
    best_val_acc = final_test_acc = 0
    best_val_ap = final_test_ap = 0
    earlystop = EarlyStopping(mode="max", patience=patience)

    nw = []
    cw = []
    gw = []
    aucs = []

    with tqdm(range(max_epochs), disable=disable_progress) as bar:
        for epoch in bar:
            loss = train(
                model, optimizer, criterion, dataset.train_dataset, grad_clip=grad_clip
            )
            train_auc, train_ap = test(model, dataset.train_dataset)
            val_auc, val_ap = test(model, dataset.val_dataset)
            ts = time.time()
            test_auc, test_ap = test(model, dataset.test_dataset)
            # print('test ',(time.time()-ts)/len(dataset.test_dataset))
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                final_test_auc = test_auc
            if val_ap > best_val_ap:
                best_val_ap = val_ap
                final_test_ap = test_ap
            bar.set_postfix(
                loss=loss,
                train_auc=train_auc,
                val_auc=val_auc,
                test_auc=test_auc,
                btest_auc=final_test_auc,
            )
            if writer:
                writer.add_scalar("Model/train_loss", loss, epoch)
                writer.add_scalar("Model/val_auc", val_auc, epoch)
                writer.add_scalar("Model/test_auc", test_auc, epoch)

            aucs.append(test_auc)

            if earlystop.step(val_auc):
                break

    return {
        "test_auc": final_test_auc,
        "test_ap": final_test_ap,
        "val_auc": best_val_auc,
        "train_auc": train_auc,
        "epoch": epoch,
        "time": time.time() - start_time,
        "time_per_epoch": (time.time() - start_time) / (epoch + 1),
    }
