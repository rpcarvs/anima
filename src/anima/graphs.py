import networkx as nx
import numpy as np
import torch
from joblib import Parallel, delayed
from rdkit import Chem
from torch_geometric.data import Data

from .smiles import SMILES


class MolGraphs(SMILES):
    def Smiles_To_Graph(self, s, vocab):
        """Read a SMILES string and convert it into a Graph. The Graph attributes
        will be based on a natural language processing approach, so a vocabullary
        needs to be supplied. The vocabullary for bonds is fixed as:
         ['AROMATIC', 'DOUBLE', 'SINGLE', 'TRIPLE'].

        Parameters
        ----------
        s : string
            The SMILES string
        vocab : list
                vocabullary

        Returns
        -------
        networkx Graph

        """

        vocab_bonds = ["AROMATIC", "DOUBLE", "SINGLE", "TRIPLE"]
        G = nx.Graph()
        molecule = Chem.MolFromSmiles(s, sanitize=False)

        for atom in molecule.GetAtoms():
            temp = self.letterToIndex(atom.GetSymbol(), vocab)
            G.add_node(atom.GetIdx(), symbol=temp)
        for bond in molecule.GetBonds():
            temp = self.letterToIndex(bond.GetBondType().name, vocab_bonds)
            G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_type=temp)
        return G

    def batch_Smiles_To_Graph(self, smiles, vocab):
        """Function to parallelize the Smiles_To_Graph function.

        Parameters
        ----------
        smiles : list
            The list of SMILES string
        vocab : list
                vocabullary

        Returns
        -------
        list of networkx Graphs

        """

        def compute(i, vocab=vocab):
            try:
                return self.Smiles_To_Graph(i, vocab)
            except Exception:
                return None

        return Parallel(n_jobs=-1, verbose=1, max_nbytes="200M", backend="threading")(
            delayed(compute)(i) for i in smiles
        )

    def smiles_to_torch(self, vocab, smiles, targets=None):
        # sourcery skip: list-comprehension, use-assigned-variable
        """Return a list of torch_geometric Data format packing the graphs and
        targets (if supplied). Final graph attributes will be based on a natural
        language processing basing, thus the vocabullary. For bonds, the vobac is
        fixed as ['AROMATIC', 'DOUBLE', 'SINGLE', 'TRIPLE'].

        Requires PyTorch and Torch_geometric libraries.

        Parameters
        ----------
        vocab : list
                vocabullary
        smiles : list
                 The list of SMILES string
        targets : list
                  list of target values

        Returns
        -------
        list of torch_geonetric Data graphs and torch tensors.

        """

        # vocab = ['Br', 'C', 'Cl', 'F', 'Li', 'N', 'O', 'S']
        x = np.array(smiles)
        x_graphs = self.batch_Smiles_To_Graph(x, vocab)
        if targets is not None:
            y = targets
            # in case targets are a classification (yes/no based only!)
            if np.any(np.isin(["yes", "no"], list(targets))):
                y = torch.tensor(np.array(y.map(dict(yes=1, no=0))))
            y = torch.tensor(np.array(y))

        dataset = []
        for ii in range(len(x_graphs)):  # type: ignore
            gg = x_graphs[ii]  # type: ignore
            node_feats = []
            edge_feats = []
            for i in gg.nodes:  # type: ignore
                node_feats.append(gg.nodes[i]["symbol"])  # type: ignore
            for i in gg.edges:  # type: ignore
                edge_feats.append(gg.get_edge_data(i[0], i[1])["bond_type"])  # type: ignore
            edge_index = torch.tensor(np.array(gg.edges).T)  # type: ignore
            node_feats = torch.tensor(node_feats)
            edge_feats = torch.tensor(edge_feats)
            if targets is not None:
                dataset.append(
                    Data(
                        x=node_feats,
                        edge_index=edge_index,
                        edge_attr=edge_feats,
                        y=y[ii],  # type: ignore
                    )
                )
            else:
                dataset.append(
                    Data(x=node_feats, edge_index=edge_index, edge_attr=edge_feats)
                )

        return dataset
