from .HLinear import FeatEmbed
from .CMLN import CMLN as Net
from torch import nn
import torch
import json
import os
from ..utils import count_parameters, cnt2str

from ..utils import setup_seed

from .LLM import LLM_model

def load_pre_post(args, dataset):
    feat_hid_dim = args.hid_dim
    if args.dataset == "Aminer":
        feat_hid_dim = 32 if args.homo else args.hid_dim
        featemb = FeatEmbed(dataset.dataset, "author venue".split(), feat_hid_dim)
        nclf_linear = None
    elif args.dataset == "Ecomm":
        featemb = FeatEmbed(dataset.dataset, "user item".split(), feat_hid_dim)
        nclf_linear = None
    elif args.dataset == "Yelp-nc":
        featemb = None
        nclf_linear = nn.Linear(args.hid_dim, args.num_classes)
    elif args.dataset == "covid":
        from cmln.trainer.nreg import NodePredictor

        featemb = None
        nclf_linear = NodePredictor(n_inp=8, n_classes=1)
    else:
        raise NotImplementedError(f"Unknown dataset {args.dataset}")
    return featemb, nclf_linear


def load_backbone(args, dataset, featemb, nclf_linear):
    in_dim, hid_dim, out_dim = args.in_dim, args.hid_dim, args.out_dim
    n_layers, dropout, metadata, predict_type, n_heads, time_window, device, norm = (
        args.n_layers,
        args.dropout,
        dataset.metadata,
        args.predict_type,
        args.n_heads,
        args.twin,
        args.device,
        args.norm,
    )

    llm_model = LLM_model(args.hid_dim, args.dataset, device)
    llm_graph_emb = llm_model.graph_encode()
    llm_cate_embs = llm_model.cate_encode()

    model = Net(
        in_dim=in_dim,
        hid_dim=hid_dim,
        num_layers=n_layers,
        dropout=dropout,
        time_window=time_window,
        metadata=metadata,
        dataset=args.dataset,
        predict_type=predict_type,
        device=device,
        featemb=featemb,
        nclf_linear=nclf_linear,
        amplifier=args.amplifier,
        llm_graph_emb = llm_graph_emb,
        llm_cate_embs = llm_cate_embs
    )

    return model


def load_lazy_hetero_weights(args, dataset, model):
    with torch.no_grad():  # Initialize lazy modules.
        if args.dataset in "Aminer Ecomm".split():
            out = model.encode(dataset.val_dataset[0])
        elif args.dataset in "Yelp-nc".split():
            print(dataset.val_dataset)
            out = model.encode(dataset.val_dataset[0][0])
        elif args.dataset in "covid ".split():
            out = model.encode(dataset.val_dataset[0][0])


def load_model(args, dataset):
    setup_seed(args.seed)
    featemb, nclf_linear = load_pre_post(args, dataset)
    model = load_backbone(args, dataset, featemb, nclf_linear)
    load_lazy_hetero_weights(args, dataset, model)
    return model
