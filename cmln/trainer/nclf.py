import json
import torch
import os.path as osp
import numpy as np
from cmln.utils import EarlyStopping
from tqdm import tqdm
import time
from torch import nn
from sklearn.metrics import f1_score, roc_auc_score, recall_score
from cmln.utils import setup_seed
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D



def train(
    model, optimizer, criterion, train_data, culmulate=1, grad_clip=0, device="cpu"
):
    model.train()

    for support, query in train_data:
        # support = move_to(support, device)
        z = model.encode(support)
        out = model.decode_nclf(z)
        mask = query.mask
        loss = criterion(out[mask], query.y[mask])
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, device="cpu"):
    def test_one(model, data):
        support, query = data
        # support = move_to(support, device)
        model.eval()
        z = model.encode(support)
        out = model.decode_nclf(z)
        mask = query.mask
        target = query.y[mask]
        pred = out[mask]
        pred = pred.argmax(dim=-1)
        f1 = f1_score(target.cpu().numpy(), pred.cpu().numpy(), average="macro")
        recall = recall_score(target.cpu().numpy(), pred.cpu().numpy(), average="macro")
        return f1, recall

    if isinstance(data, list):
        f1s = []
        accs = []
        for d in data:
            f1, acc = test_one(model, d)
            f1s.append(f1)
            accs.append(acc)
        
        return np.mean(f1s), np.mean(accs)
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
    device="cpu",
):
    # procedure
    setup_seed(args.seed)
    start_time = time.time()
    best_val_f1 = final_test_f1 = 0
    best_val_auc = final_test_auc = 0
    earlystop = EarlyStopping(mode="max", patience=patience)

    with tqdm(range(max_epochs), disable=disable_progress) as bar:
        for epoch in bar:
            loss = train(
                model,
                optimizer,
                criterion,
                dataset.train_dataset,
                grad_clip=grad_clip,
                device=device,
            )
            train_f1, train_auc = test(model, dataset.train_dataset, device=device)
            val_f1, val_auc = test(model, dataset.val_dataset, device=device)
            ts = time.time()
            test_f1, test_auc = test(model, dataset.test_dataset, device=device)
            # print('test ',(time.time()-ts)/len(dataset.test_dataset))
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                final_test_f1 = test_f1
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                final_test_auc = test_auc
            bar.set_postfix(
                loss=loss,
                train_f1=train_f1,
                val_f1=val_f1,
                test_f1=test_f1,
                btest_f1=final_test_f1,
            )
            if writer:
                writer.add_scalar("Model/train_loss", loss, epoch)
                writer.add_scalar("Model/val_f1", val_f1, epoch)
                writer.add_scalar("Model/test_f1", test_f1, epoch)

            if earlystop.step(val_f1):
                break

    return {
        "test_f1": final_test_f1,
        "test_recall": final_test_auc,
        "val_f1": best_val_f1,
        "train_f1": train_f1,
        "epoch": epoch,
        "time": time.time() - start_time,
        "time_per_epoch": (time.time() - start_time) / (epoch + 1),
    }
