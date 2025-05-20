import numpy as np
import torch

from .kernel_utils import (
    big_smiles,
    linear_model_red_only,
    max_length,
    neural_inference,
    smiles_exceptions,
    smiles_sequence,
)


def AIkernel(
    smiles_list,
    index=None,
    n_jobs=-1,
    batch_size=64,
    return_redox=False,
    return_smiles=False,
):
    """The AI-kernel as developed in REF_PAPER. This function receives a
    list of SMILES-strings as inputs and return voltages (or voltges and
    redox potentials). The voltages are referred to the Lithium reference
    electrode (i.e. vs. Li/Li+). The oxidation and reduction potentials
    are referred to the vacuum.

    Args:
        smiles_list ([string]): List of the SMILES-string. (must be a list)
        index ([integer]): list of indices for the smiles. If provided, a list of
                            updated indices will be returned without the invalid SMILES's indices.
        n_jobs (int, optional): Number of jobs to process the SMILES. Defaults to -1 (all available cores).
        batch_size (int, optional): size of mini_batches
        return_redox (bool): if True the kernel will also returns the oxidation and reduction potential
        return_smiles (bool) = if True a list with the valid SMILES will be returned

    Returns:
        {}: a dictionary containing all the relevant results
    """

    all_sequences = smiles_sequence(smiles_list, n_jobs, max_length)
    all_sequences = [ii for ii in all_sequences if ii is not None]  # type: ignore
    if not any([ii.tolist() for ii in all_sequences]):
        return
    packing = torch.nn.utils.rnn.pack_sequence(all_sequences, enforce_sorted=False)
    packing_padding = torch.nn.utils.rnn.pad_packed_sequence(
        packing, batch_first=True, total_length=max_length
    )
    processed_smiles = packing_padding[0][:, :, 0]
    del packing, packing_padding, all_sequences

    # try:
    ox, red = neural_inference(processed_smiles, batch_size, n_jobs, max_length)
    out = {}
    rr = []
    if return_smiles:
        temp = smiles_list
        intersct = np.intersect1d(
            temp, big_smiles + smiles_exceptions, return_indices=True
        )[1]
        rr.append(intersct)
        while rr[-1].size > 0:
            temp = np.delete(temp, rr[-1])
            rr.append(
                np.intersect1d(
                    temp, big_smiles + smiles_exceptions, return_indices=True
                )[1]
            )
        out["smiles"] = list(temp)

    if index:
        temp = index
        # ll = []
        for i in rr:
            if i.size > 0:
                # ll += list(i)
                temp = np.delete(temp, i)
        out["index"] = list(temp)

    out["voltages"] = list(linear_model_red_only(red))

    if return_redox:
        out["ox/red"] = [list(ox), list(red)]

    # if return_redox == True: return linear(ox, red), [ox, red]
    # return linear(ox, red)
    if smiles_list is not None:
        return out
    else:
        return print("No valid input/SMILES")
