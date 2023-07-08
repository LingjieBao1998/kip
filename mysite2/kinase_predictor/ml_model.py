import numpy as np
import pandas as pd
import os
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import dgl
from dgl.nn.pytorch.conv import GINConv
from rdkit.Chem import PandasTools
from rdkit import Chem
from dgllife.utils import BaseAtomFeaturizer
from dgllife.utils import (
    atom_type_one_hot,
    atom_degree_one_hot,
    atom_implicit_valence_one_hot,
    atom_formal_charge,
    atom_num_radical_electrons,
    atom_hybridization_one_hot,
    atom_is_aromatic,
    atom_total_num_H_one_hot,
    atom_chiral_tag_one_hot,
)
from dgllife.utils import ConcatFeaturizer
from dgllife.utils import smiles_to_bigraph, mol_to_bigraph
from dgllife.utils import CanonicalBondFeaturizer
from dgl.data.utils import save_graphs
from rdkit.Chem import PandasTools
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
from rdkit.Chem.Draw import rdMolDraw2D
from collections import Counter

np.set_printoptions(precision=4)  # 保留四位小数
def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True
init_seeds(seed=6)


class Classifer(nn.Module):
    def __init__(self, in_feats, graph_hidden, depth, mlp_layers, n_classes, dropout):
        super(Classifer, self).__init__()
        self.len_mlp_layers = len(mlp_layers)

        # GCN
        self.gcn1 = GINConv(
            apply_func=nn.Linear(in_feats, graph_hidden), aggregator_type="sum"
        )
        self.gcn1_bn = torch.nn.BatchNorm1d(
            graph_hidden, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.gcn2 = nn.ModuleList(
            [
                GINConv(
                    apply_func=nn.Linear(graph_hidden, graph_hidden),
                    aggregator_type="sum",
                )
                for _ in range(depth - 1)
            ]
        )
        self.gcn2_bns = nn.ModuleList(
            [
                torch.nn.BatchNorm1d(
                    graph_hidden,
                    eps=1e-05,
                    momentum=0.1,
                    affine=True,
                    track_running_stats=True,
                )
                for _ in range(depth - 1)
            ]
        )

        # predict
        if len(mlp_layers) != 0:
            self.fc1 = nn.Linear(graph_hidden, mlp_layers[0])
            self.bn1 = torch.nn.BatchNorm1d(
                mlp_layers[0],
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            )
            self.linears = nn.ModuleList(
                [
                    nn.Linear(mlp_layers[i], mlp_layers[i + 1])
                    for i in range(len(mlp_layers) - 1)
                ]
            )
            self.bns = nn.ModuleList(
                [
                    torch.nn.BatchNorm1d(
                        mlp_layers[i + 1],
                        eps=1e-05,
                        momentum=0.1,
                        affine=True,
                        track_running_stats=True,
                    )
                    for i in range(len(mlp_layers) - 1)
                ]
            )

            self.fc2 = nn.Linear(mlp_layers[-1], n_classes * 2)
        else:
            self.fc2 = nn.Linear(graph_hidden, n_classes * 2)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.sigmoid = nn.Sigmoid()
        self.embedding = None

    def forward(self, g, h, edge_weights=None):
        h = self.gcn1(g, h, edge_weights) + self.gcn1.apply_func(h)
        h = self.gcn1_bn(h)
        h = self.dropout(h)
        h = self.act(h)
        for gcn2, bn2 in zip(self.gcn2, self.gcn2_bns):
            h = gcn2(g, h, edge_weights) + gcn2.apply_func(h)
            h = bn2(h)
            h = self.dropout(h)
            h = self.act(h)

        # readout
        g.ndata["h"] = h
        g.apply_edges(
            lambda edges: {
                "embedding": torch.cat([edges.src["h"], edges.dst["h"]], dim=1)
            }
        )
        self.embedding = g.edata["embedding"]

        x = dgl.readout_nodes(g, "h", op="sum")

        if self.len_mlp_layers != 0:
            x = self.bn1(self.fc1(x))
            x = self.dropout(x)
            x = self.act(x)
            for (i, linear), bn in zip(enumerate(self.linears), self.bns):
                x = linear(x)
                x = bn(x)
                x = self.dropout(x)
                x = self.act(x)
        x = self.fc2(x)
        x = self.sigmoid(x[:, -n_classes:])
        return x

def collate(sample):
    batched_graph = dgl.batch(sample)
    batched_graph.set_n_initializer(dgl.init.zero_initializer)
    batched_graph.set_e_initializer(dgl.init.zero_initializer)
    return batched_graph

atom_featurizer = BaseAtomFeaturizer(
    {
        "h": ConcatFeaturizer(
            [
                atom_type_one_hot,
                atom_degree_one_hot,
                atom_implicit_valence_one_hot,
                atom_formal_charge,
                atom_num_radical_electrons,
                atom_hybridization_one_hot,
                atom_is_aromatic,
                atom_total_num_H_one_hot,
                atom_chiral_tag_one_hot,
            ]
        )
    }
)

bond_featurizer = CanonicalBondFeaturizer(bond_data_field="feat", self_loop=False)
Base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
os.makedirs(os.path.join(Base_dir, r"static/file/download_file/grobal_result"), exist_ok=True)
os.makedirs(os.path.join(Base_dir, r"static/file/download_file/invalid_smiles"), exist_ok=True)
os.makedirs(os.path.join(Base_dir, r"static/file/download_file/single_mol"), exist_ok=True)

model_path = r"static/file/model_file/seed=6, graph_hidden=512,depth=6,mlp_layers=[1500, 1000, 1000],learning_rate=1e-05,weight_decay=0.0001,dropout=0.05,Batch_size=64.pth"
model_path = os.path.join(Base_dir, model_path)

seed = 6
GNN_graph_hidden = 512
GNN_depth = 6
GNN_mlp_layers = [1500, 1000, 1000]
GNN_learning_rate = 1e-05
GNN_weight_decay = 0.0001
GNN_dropout = 0.05
GNN_Batch_size = 64
in_feats = 78
n_classes = 204
# due to our severs only support cpu
device = torch.device("cpu")  # "cuda:0" if torch.cuda.is_available() else "cpu"



net = Classifer(
    in_feats=in_feats,
    graph_hidden=GNN_graph_hidden,
    depth=GNN_depth,
    mlp_layers=GNN_mlp_layers,
    n_classes=n_classes,
    dropout=GNN_dropout,
).to(
    device
)  ### 实例化模型

model_file = torch.load(model_path, map_location=device)
net.load_state_dict(model_file["model_state_dict"])
net.eval()

df_uniprot = pd.read_csv(os.path.join(Base_dir, r"static/file/UniProt.csv"))
uniprot_array = np.array(df_uniprot["UniProt"])
target_array = np.arange(uniprot_array.shape[0])

def model_output(df):
    length_fold = None
    PandasTools.AddMoleculeColumnToFrame(df, smilesCol="smiles")
    invalid_smiles = [
        df.loc[i, "smiles"] for i in df.index if df.loc[i, "ROMol"] is None
    ]
    df_invalid = pd.DataFrame(data=invalid_smiles, columns=["smiles"])
    if len(df_invalid) > 0:
        length_fold = len(
            os.listdir(
                os.path.join(Base_dir, r"static/file/download_file/invalid_smiles")
            )
        )
        df_invalid.to_csv(
            os.path.join(
                Base_dir,
                r"static/file/download_file/invalid_smiles/invalid_smiles_%d.csv"
                % (length_fold),
            ),
            index=False,
        )

    df = df[~df.ROMol.isnull()]
    PandasTools.RemoveSaltsFromFrame(df)
    df["canonical_smiles"] = df["ROMol"].map(Chem.MolToSmiles)
    del df["ROMol"]

    # graph 生成
    train_g = [
        smiles_to_bigraph(
            smiles,
            node_featurizer=atom_featurizer,
            edge_featurizer=bond_featurizer,
            add_self_loop=False,
        )
        for smiles in df.canonical_smiles
    ]
    dataset = DataLoader(train_g, GNN_Batch_size, shuffle=False, collate_fn=collate)
    outputs_list = []
    with torch.no_grad():
        for _, bg in enumerate(dataset):
            bg = bg.to(device)
            atom_feats = bg.ndata["h"].to(device)
            atom_feats.requires_grad = False
            outputs = net(bg, atom_feats).data.detach()
            outputs_list.append(outputs)
    outputs_numpy = torch.cat(outputs_list, dim=0).data.cpu().numpy()
    outputs_numpy = np.around(outputs_numpy, 4)
    outputs_numpy = np.clip(outputs_numpy, 0.0001, 0.9999)
    result = pd.DataFrame(
        data=outputs_numpy,
        columns=[i for i in df_uniprot.UniProt],
        index=df.canonical_smiles,
    )

    result_length = len(
        os.listdir(
            os.path.join(
                os.path.join(Base_dir, r"static/file/download_file/grobal_result")
            )
        )
    )
    result.to_csv(
        os.path.join(
            Base_dir,
            r"static/file/download_file/grobal_result/result_%d.csv" % (result_length),
        )
    )
    result = pd.read_csv(
        os.path.join(
            Base_dir,
            r"static/file/download_file/grobal_result/result_%d.csv" % (result_length),
        )
    )

    df_smiles = pd.DataFrame(df["smiles"])
    df_smiles.columns = ["original_smiles"]
    print("smiles", df_smiles)
    result = pd.concat([df_smiles, result], axis=1)
    result.to_csv(
        os.path.join(
            Base_dir,
            r"static/file/download_file/grobal_result/result_%d.csv" % (result_length),
        ),
        index=False,
    )
    return result, result_length, invalid_smiles, length_fold
