import torch
from torch_geometric.nn import RGCNConv
from cmln.data.utils import make_hodata
import networkx as nx
from torch_geometric.utils import degree
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

from .LLM import LLM_model


class cate_level_extraction(torch.nn.Module):
    def __init__(
        self,
        hid_dim,
        dropout,
        metadata
    ):
        super().__init__()
        self.cate_num = len(metadata[0])

    def get_cate_emb(self, x, degrees):
        cate_emb = []
        for i in range(self.cate_num):
            cate_i_emb = torch.mean(x[i]*torch.unsqueeze(degrees[i],-1), dim=0) # 节点类别i的均值
            cate_emb.append(cate_i_emb)
            
        return cate_emb # [类别数目,emb_dim]
    
class graph_level_extraction(torch.nn.Module):
    def __init__(
        self,
        hid_dim,
        dropout,
        metadata,
    ):
        super().__init__()
        self.cate_num = len(metadata[0])
        self.graph_cate_w = torch.nn.Sequential(
                            torch.nn.Linear(in_features=hid_dim,
                                            out_features=1,
                                            bias=True),
                            torch.nn.Sigmoid()
        )
        self.layer_norm = torch.nn.LayerNorm(hid_dim, eps=1e-5, elementwise_affine=True)



    def get_graph_emb(self, cate_emb):
        cate_w = self.graph_cate_w(torch.stack(cate_emb))

        weighted_cate_emb = torch.stack(cate_emb)*cate_w

        graph_emb = torch.mean(weighted_cate_emb,dim=0) 
        graph_emb = self.layer_norm(graph_emb)
            
        return graph_emb, cate_w
    
class temporal_attention(torch.nn.Module):
    def __init__(
        self,
        hid_dim,
    ):
        super().__init__()
        self.alpha = torch.nn.Sequential(
                        torch.nn.Linear(in_features=hid_dim,
                                        out_features=1,
                                        bias=True),
                        torch.nn.Sigmoid()
        )
        self.norm = torch.nn.LayerNorm(hid_dim, eps=1e-5, elementwise_affine=True)
    
    def temporal_encode(self, feats, dataset):
        temporal_w = self.alpha(torch.stack(feats))
        feats = torch.stack(feats)*torch.softmax(temporal_w,dim=0)

        x = torch.mean(feats,dim=0)
        if dataset!='covid':
            x = self.norm(x)

        return x

class CMLN(torch.nn.Module):
    def __init__(
        self,
        in_dim,
        hid_dim,
        num_layers,
        dropout,
        time_window,
        metadata,
        dataset,
        predict_type,
        device,
        featemb=None,
        nclf_linear=None,
        amplifier=None,
        llm_graph_emb = None,
        llm_cate_embs = None
    ):
        super().__init__()
        num_relations = len(metadata[1])
        self.num_cates = len(metadata[0])
        self.cates = metadata[0]
        convs = torch.nn.ModuleList()
        convs.append(RGCNConv(in_dim, hid_dim, num_relations))

        for _ in range(num_layers - 1):
            convs.append(RGCNConv(hid_dim, hid_dim, num_relations))
        self.convs = convs
        # self.hlinear = HLinear(hid_dim, metadata, act='None')
        self.timeframe = list(range(time_window))
        self.predict_type = predict_type
        self.featemb = featemb if featemb else lambda x: x
        self.nclf = nclf_linear
        self.dataset = dataset

        if dataset == "Aminer":
            init_ratio = [0.2, -0.1, 0.6]
        elif dataset == "Ecomm":
            init_ratio = [-0.1, 0.1, -0.1]
        elif dataset == "Yelp-nc":
            init_ratio = [0.6, 0.2, 0.2]
        elif dataset == "covid":
            init_ratio = [1.0, -0.1, 0.0]
        self.node_w = torch.nn.Parameter(torch.FloatTensor([init_ratio[0]]))

        self.cate_emb_module = cate_level_extraction(hid_dim=hid_dim, dropout=dropout, metadata=metadata)
        self.cate_w = torch.nn.Parameter(torch.FloatTensor([init_ratio[1]]))

        self.graph_emb_module = graph_level_extraction(hid_dim=hid_dim, dropout=dropout, metadata=metadata)
        self.graph_w = torch.nn.Parameter(torch.FloatTensor([init_ratio[2]]))

        self.temporal_attn = temporal_attention(hid_dim=hid_dim)

        self.amplifier = amplifier

        # self.llm_model = LLM_model(hid_dim, dataset, device)
        self.device = device

        self.llm_graph_emb = llm_graph_emb
        self.llm_cate_embs = llm_cate_embs

        self.graph_mlp = torch.nn.Sequential(
                            torch.nn.Linear(in_features=1536,
                                            out_features=hid_dim*2,
                                            bias=True),
                            torch.nn.ReLU(),
                            torch.nn.Linear(in_features=hid_dim*2,
                                            out_features=hid_dim,
                                            bias=True),
        )
        self.cate_mlp = torch.nn.Sequential(
                            torch.nn.Linear(in_features=1536,
                                            out_features=hid_dim*2,
                                            bias=True),
                            torch.nn.ReLU(),
                            torch.nn.Linear(in_features=hid_dim*2,
                                            out_features=hid_dim,
                                            bias=True),
        )



    def encode(self, data, *args, **kwargs):
        feats = []

        embedding_vector = self.llm_graph_emb.to(self.device)
        self.graph_mlp.to(embedding_vector.device)
        llm_graph_emb = self.graph_mlp(embedding_vector)

        llm_cate_embs = []
        for i in range(len(self.llm_cate_embs)):
            embedding_vector = self.llm_cate_embs[i].to(self.device)
            self.cate_mlp.to(embedding_vector.device)
            llm_cate_embs.append(self.cate_mlp(embedding_vector))

        for ttype in self.timeframe:
            graph = data[ttype]
            
            x_dict = self.featemb(graph.x_dict)
            # x_dict = self.hlinear(x_dict)
            e_dict = graph.edge_index_dict

            x, e, predict_mask, cate_mask, hodata = make_hodata(x_dict, e_dict, self.predict_type)

            num_nodes = x.shape[0]
            degrees = degree(e[0], num_nodes=num_nodes)+degree(e[1], num_nodes=num_nodes)

            degrees = torch.softmax(degrees, dim=0)
            degrees = torch.pow(self.amplifier,degrees) 

            edge_type = hodata.edge_type
            for i, conv in enumerate(self.convs):
                x = conv(x, e, edge_type)
                if i != len(self.convs) - 1:
                    x = x.relu()

            x = [x[m] for m in cate_mask]
            degrees = [degrees[m] for m in cate_mask]
            
            cate_emb = self.cate_emb_module.get_cate_emb(x,degrees)
            for i in range(len(cate_emb)):
                cate_emb[i] = cate_emb[i]*(torch.log(torch.abs(llm_cate_embs[i]))).to(cate_emb[i].device)

            graph_emb, cate_w = self.graph_emb_module.get_graph_emb(cate_emb)
            graph_emb = graph_emb*(torch.log(torch.abs(llm_graph_emb))).to(graph_emb.device)
            
            for i in range(len(x)):
                x[i] = x[i]*self.node_w +cate_emb[i]*self.cate_w +graph_emb*self.graph_w
            

            x = torch.cat(x,dim=0)

            if ttype == self.timeframe[0]:
                feats = [x]
            else:
                feats.append(x)

        x = self.temporal_attn.temporal_encode(feats, self.dataset)

        if isinstance(predict_mask, list):
            x = [x[predict_mask[0]], x[predict_mask[1]]]
        else:
            x = x[predict_mask]

        return x

    def decode(self, z, edge_label_index, *args, **kwargs):
        if isinstance(z, list) or isinstance(z, tuple):
            return (z[0][edge_label_index[0]] * z[1][edge_label_index[1]]).sum(dim=-1)
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_nclf(self, z):
        out = self.nclf(z)

        return out
