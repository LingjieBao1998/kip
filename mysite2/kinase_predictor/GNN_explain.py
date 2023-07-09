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

# 一些常量参数
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

class Classifer(nn.Module):
    def __init__(
        self, 
        in_feats, 
        graph_hidden, 
        depth, 
        mlp_layers, 
        n_classes, 
        dropout, 
        ):

        """ 
        initialization parameters for model

        Args:
            in_feats (int): the input dimention of the freature of molecule graph
            graph_hidden (int): the hidden dimention of the freature transformed by GNN model
            depth (int): the number of iteration of GNN model
            mlp_layers (list): the list containing hidden layer of MLP
            n_classes (int): the output dimention that solve diifferent tasks
            dropout (float): the ratio of dropout in MLP layers and GNN layers  
        """
        super(Classifer, self).__init__()
        self.len_mlp_layers = len(mlp_layers)

        # GCN
        self.gcn1 = GINConv(apply_func=nn.Linear(in_feats, graph_hidden), aggregator_type="sum")
        self.gcn1_bn = torch.nn.BatchNorm1d(
            graph_hidden, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.gcn2 = nn.ModuleList([GINConv(apply_func=nn.Linear(graph_hidden, graph_hidden), 
                                    aggregator_type="sum",) for _ in range(depth - 1)])
        self.gcn2_bns = nn.ModuleList([torch.nn.BatchNorm1d(graph_hidden, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True,)
                                                            for _ in range(depth - 1)])

        # predict
        if len(mlp_layers) != 0:
            self.fc1 = nn.Linear(graph_hidden, mlp_layers[0])
            self.bn1 = torch.nn.BatchNorm1d(mlp_layers[0], eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True,)
            self.linears = nn.ModuleList([nn.Linear(mlp_layers[i], mlp_layers[i + 1]) 
                                                    for i in range(len(mlp_layers) - 1)])
            self.bns = nn.ModuleList([torch.nn.BatchNorm1d(mlp_layers[i + 1], eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True,)
                                                    for i in range(len(mlp_layers) - 1)])

            self.fc2 = nn.Linear(mlp_layers[-1], n_classes * 2)
        else:
            self.fc2 = nn.Linear(graph_hidden, n_classes * 2)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, h, edge_weights=None, index=-1):
        with g.local_scope():
            h = self.gcn1(g, h, edge_weights) + self.gcn1.apply_func(h)
            h = self.gcn1_bn(h)
            h = self.dropout(h)
            h = self.act(h)

            # iteration in GNN
            for gcn2, bn2 in zip(self.gcn2, self.gcn2_bns):
                h = gcn2(g, h, edge_weights) + gcn2.apply_func(h)
                h = bn2(h)
                h = self.dropout(h)
                h = self.act(h)

            # readout
            g.ndata["h"] = h
            # g.apply_edges(lambda edges: {"embedding": torch.cat([edges.src["h"], edges.dst["h"]], dim=1)})

            x = dgl.readout_nodes(g, "h", op="sum")

            # iteration in MLP
            if self.len_mlp_layers != 0:
                x = self.bn1(self.fc1(x))
                x = self.dropout(x)
                x = self.act(x)
                for (_, linear), bn in zip(enumerate(self.linears), self.bns):
                    x = linear(x)
                    x = bn(x)
                    x = self.dropout(x)
                    x = self.act(x)
            x = self.fc2(x)
            x = self.sigmoid(x[:, -n_classes:])

            # output by index
            if index == -1:
                return x
            else:
                return x[:, index]


## base config
bond_featurizer = CanonicalBondFeaturizer(bond_data_field="feat", self_loop=False)

Base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
model_path = r"static/file/model_file/seed=6, graph_hidden=512,depth=6,mlp_layers=[1500, 1000, 1000],learning_rate=1e-05,weight_decay=0.0001,dropout=0.05,Batch_size=64.pth"
model_path = os.path.join(Base_dir, model_path)

## base config
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
device = torch.device("cpu")

# 模型
net = Classifer(
    in_feats = in_feats,
    graph_hidden = GNN_graph_hidden,
    depth = GNN_depth,
    mlp_layers = GNN_mlp_layers,
    n_classes = n_classes,
    dropout = GNN_dropout,).to(device)  ### 实例化模型
print("net", net)

model_file = torch.load(model_path, map_location=device)
net.load_state_dict(model_file["model_state_dict"])
net = net.to(device)
net.eval()

## the information of protein kinase
df_uniprot = pd.read_csv(os.path.join(Base_dir, r"static/file/UniProt.csv"))
uniprot_array = np.array(df_uniprot["UniProt"])
target_array = np.arange(uniprot_array.shape[0])

def explain_GNN(smiles, uniprot, mode="group"):
    """return explanation of prediction againset kinase

    Args:
        smiles (str): _description_
        uniprot (str): _description_
        mode (str, optional): Determining the size of picture. 
                            Single Prediciton mode: 'uniprot', big size; 
                            Multiple Prediciton mode: 'group', small size. 
                            Defaults to "group".

    Returns:
        str: return explanation of prediction againset kinase
    """

    # graph size
    graph_size = (360, 270)
    if mode == "uniprot":
        graph_size = (400, 320)
    df = pd.DataFrame(data=[smiles], columns=["smiles"])
    PandasTools.AddMoleculeColumnToFrame(df, smilesCol="smiles")
    PandasTools.RemoveSaltsFromFrame(df)
    df["canonical_smiles"] = df["ROMol"].map(Chem.MolToSmiles)
    del df["ROMol"]
    train_g = [smiles_to_bigraph(smiles, 
                                node_featurizer = atom_featurizer, 
                                edge_featurizer = bond_featurizer, 
                                add_self_loop = False,)
                                for smiles in df.canonical_smiles]

    print("train_g", train_g[0])
    dataset = DataLoader(train_g, 1, shuffle=False, collate_fn=collate)

    target_index = int(target_array[uniprot_array == uniprot])
    canonical_smiles = df["canonical_smiles"][0]
    mol = Chem.MolFromSmiles(canonical_smiles)

    if True:
        for _, bg in enumerate(dataset):
            bg = bg.to(device)
            atom_feats = bg.ndata["h"].to(device)
            atom_feats.requires_grad = False

            # get base value
            base_value = net(bg, atom_feats, index=target_index).item()

            # keep a list to record the result
            edge_weight_list = []

            for atom_idx in range(mol.GetNumAtoms()):
                mask = torch.ones(bg.num_edges()).to(device)
                edge_mask = []

                # iteration of edges that contain atom with specifical atom_idx
                for bond_idx in range(mol.GetNumBonds()):
                    if mol.GetBondWithIdx(bond_idx).GetBeginAtomIdx() == atom_idx:
                        edge_mask.append(bond_idx)

                    if mol.GetBondWithIdx(bond_idx).GetEndAtomIdx() == atom_idx:
                        edge_mask.append(bond_idx)

                for edge in edge_mask:
                    mask[edge * 2] = 0
                    mask[edge * 2 + 1] = 0

                # 屏蔽边
                edge_mask_value = net(bg, atom_feats, mask, index=target_index).item()
                edge_weight_list.append(base_value - edge_mask_value)

            edge_weight = np.array(edge_weight_list)
            edge_weight = edge_weight / np.max(np.abs(edge_weight))

            # edge_mask
            norm = matplotlib.colors.Normalize(vmin=np.min(edge_weight), vmax=np.max(edge_weight))
            cmap = cm.get_cmap("Oranges")
            plt_colors = cm.ScalarMappable(norm=norm, cmap=cmap)
            edge_colors = {
                _: plt_colors.to_rgba(edge_weight[_]) for _ in range(len(edge_weight))
            }

            d2d = rdMolDraw2D.MolDraw2DSVG(graph_size[0], graph_size[1])
            d2d.DrawMolecule(
                mol,
                highlightAtoms=[int(_) for _ in range(len(edge_weight))],
                highlightBonds=[],
                highlightAtomColors=edge_colors,
            )
            d2d.FinishDrawing()
            edge_mask = d2d.GetDrawingText()
            edge_mask = edge_mask.replace("fill:#FFFFFF;", "fill:transparent;")
            edge_mask_index = edge_mask.find(r"<!-- END OF HEADER -->") - 2
            style = (
                "style='width: 90%; max-width: "
                + str(graph_size[0])
                + "px; height: auto;'"
            )
            edge_mask = (
                edge_mask[:edge_mask_index] + style + edge_mask[edge_mask_index:]
            )

            return edge_mask, base_value