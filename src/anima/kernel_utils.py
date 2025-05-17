"""
A collection of trained models.

Created by © Rodrigo Carvalho 2021
Mainteined by © Rodrigo Carvalho
"""

import os
from typing import List

import numpy as np
import torch
from joblib import Parallel, delayed

from .smiles import SMILES

"""Notes:
-add a proper logging instead of prints in the future
-improve some functions. I was a bit stressed here with the PhD"""

smiles_exceptions = []
big_smiles = []
path = os.path.dirname(__file__) + "/lib/aikernel/"

# reading vocab
with open(path + "vocab.dat", "r") as f:
    vocab = f.read().splitlines()
sml = SMILES()


def transform(molecule: str, fix: bool = False) -> List[List[int]]:
    """Transform SMILES to a standard format"""
    if fix:
        molecule = sml.PS_fix(molecule)
    transformed = sml.OB_standard_smiles(sml.standard_smiles(molecule))
    cleaned = sml.smiles_cleaner(transformed)
    return sml.smilesToSequence(cleaned, vocab)


def smiles_sequence(smiles_list, n_jobs, max_length):
    """Parallel Process a list of SMILES using the transform function"""
    global smiles_exceptions
    global big_smiles

    def compute(i):
        global smiles_exceptions
        global big_smiles

        if len(sml.smilesSEP(i)) > max_length:
            big_smiles.append(i)
            return
        else:
            try:
                try:
                    transformed = transform(i)
                    return torch.tensor(transformed)
                except Exception:
                    transformed = transform(i, fix=True)
                    return torch.tensor(transformed)
            except Exception:
                smiles_exceptions.append(i)
                return

    all_sequences = Parallel(
        n_jobs=n_jobs,
        verbose=1,
        max_nbytes="200M",
        backend="threading",
    )(delayed(compute)(i) for i in smiles_list)

    if smiles_exceptions:
        print("Error when processing SMILES:\n", smiles_exceptions)
        with open("invalid_smiles.dat", "w") as f:
            for i in smiles_exceptions:
                f.write(str(i) + "\n")
    if big_smiles:
        print("SMILES bigger than the max allowed length (54):\n", big_smiles)
        with open("big_smiles.dat", "w") as f:
            for i in big_smiles:
                f.write(str(i) + "\n")

    return all_sequences


# Defining the linear model
def linear_model_redox(ox, red):
    return np.multiply(0.1539, ox) + np.multiply(0.8298, red) - 1.5822


# Defining the linear model (only reduction potential)
def linear_model_red_only(red):
    return np.multiply(0.8809, red) - 0.7833


# defining the neural model
def neural_inference(processed_smiles, batch_size, n_jobs, max_length):
    # checking cuda
    use_cuda = True
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    # prediction function
    def predictions(model, processed_smiles, device, batch_size=batch_size):
        # predictions

        model.eval().to(device)

        pred_data = torch.utils.data.TensorDataset(processed_smiles)
        pred_loader = torch.utils.data.DataLoader(
            pred_data, shuffle=False, batch_size=batch_size, drop_last=False
        )
        batches = len(processed_smiles) / batch_size

        temp = np.empty([])
        with torch.no_grad():
            for batch_idx, data in enumerate(pred_loader):
                print(
                    "Batch: {:010.2f} of {:010.2f} batches".format(
                        batch_idx + 1, batches
                    ),
                    end="\r",
                )
                inputs = data[0]
                inputs = inputs.to(device)

                output = model(inputs)
                if batch_idx == 0:
                    temp = output.cpu().detach().numpy()
                else:
                    temp = np.append(temp, output.cpu().detach().numpy())
                # temp.append(*[i for i in output.cpu().detach().numpy()])

        return np.reshape(temp, -1)

    # defining the NN model
    class NN(torch.nn.Module):
        def __init__(
            self,
            hidden_dim,
            output_dim,
            n_layers,
            decoder_in,
            decoder_out,
            vocab_size,
            emb_dim,
            max_length,
            dropout,
        ):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.n_layers = n_layers

            self.gruA = torch.nn.GRU(
                emb_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout
            )
            self.gruB = torch.nn.GRU(
                emb_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout
            )
            self.gruC = torch.nn.GRU(
                emb_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout
            )
            self.gruD = torch.nn.GRU(
                emb_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout
            )

            self.fcA = torch.nn.Linear(hidden_dim, decoder_in)
            self.fcB = torch.nn.Linear(hidden_dim, decoder_in)
            self.fcC = torch.nn.Linear(hidden_dim, decoder_in)
            self.fcD = torch.nn.Linear(hidden_dim, decoder_in)

            self.decoder = torch.nn.Linear(max_length * 4 * decoder_in, decoder_out)
            self.pre_output = torch.nn.Linear(decoder_out, 1)
            self.output = torch.nn.Linear(decoder_out, output_dim)

            self.activation = torch.nn.Mish()
            self.embeddings = torch.nn.Embedding(
                vocab_size + 1, emb_dim, max_norm=1.0, padding_idx=0
            )
            # self.dropout = torch.nn.Dropout(dropout)

        def forward(self, inputs):
            batch = len(inputs)
            inputs = self.embeddings(inputs)

            hidden = self.initHidden(batch, self.hidden_dim)
            gruA, hidden = self.gruA(inputs, hidden)
            hidden = self.initHidden(batch, self.hidden_dim)
            gruB, hidden = self.gruB(inputs, hidden)
            hidden = self.initHidden(batch, self.hidden_dim)
            gruC, hidden = self.gruC(inputs, hidden)
            hidden = self.initHidden(batch, self.hidden_dim)
            gruD, hidden = self.gruD(inputs, hidden)

            fcA = self.fcA(gruA)
            fcB = self.fcB(gruB)
            fcC = self.fcC(gruC)
            fcD = self.fcD(gruD)

            cat = torch.cat((fcA, fcB, fcC, fcD), -1)
            cat = cat.reshape(batch, -1)
            cat = self.activation(cat)

            decoder = self.decoder(cat)
            decoder = self.activation(decoder)
            output = self.output(decoder)

            return output[:, 0]

        def initHidden(self, batch_size, hidden_dim):
            return torch.zeros(
                self.n_layers,
                batch_size,
                hidden_dim,
                dtype=torch.float,
                device=device,
            )

    # loading trained models
    decoder_in = 32
    decoder_out = 64
    hidden_dim = 256
    n_layers = 2
    emb_dim = 512
    output_dim = 1
    vocab_size = len(vocab)
    nn_ox = NN(
        hidden_dim,
        output_dim,
        n_layers,
        decoder_in,
        decoder_out,
        vocab_size,
        emb_dim,
        max_length,
        0,
    )
    nn_red = NN(
        hidden_dim,
        output_dim,
        n_layers,
        decoder_in,
        decoder_out,
        vocab_size,
        emb_dim,
        max_length,
        0,
    )
    nn_ox.load_state_dict(
        torch.load(path + "nn_ox.pt", map_location=torch.device(device))
    )
    nn_red.load_state_dict(
        torch.load(path + "nn_red.pt", map_location=torch.device(device))
    )
    nn_ox.eval()
    nn_red.eval()

    oxpot = predictions(nn_ox, processed_smiles, device, batch_size=batch_size)
    redpot = predictions(nn_red, processed_smiles, device, batch_size=batch_size)
    return oxpot, redpot
