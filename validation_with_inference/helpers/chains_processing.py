from getdist import MCSamples
import numpy as np

import os
import re


def clean_label(label):
    """
    Clean a LaTeX label by removing LaTeX commands and unnecessary characters.

    Parameters:
        label (str): LaTeX label

    Returns:
        str: Cleaned label suitable for variable naming
    """
    return re.sub(r'\\[^a-zA-Z]*|[{}]', '', re.sub(r'\\mathrm{([^}]*)}', r'\1', label)).replace('_', '')


def get_mc_samples(chain, labels, add_s8=True, burn_in=0.0, oms8=False):
    """
    Generate Monte Carlo samples from a given chain with options to add S_8 parameter.

    Parameters:
        chain (str): chain name
        labels (list of str): LaTeX labels for each parameter in the chain
        add_s8 (bool): If True, add the S_8 parameter to the chain as a new parameter.
        burn_in (float): Fraction of the chain to discard as burn-in. E.g., 0.1 to discard the first 10%.
        oms8 (bool): If True, only return samples for the Omega_m and S_8 parameters.

    Returns:
        MCSamples: An object containing the Monte Carlo samples, parameter names, labels, and weights
    """
    start_chain = int(chain.shape[0] * burn_in)
    names = [clean_label(l) for l in labels]
    if add_s8 and 'S_8' not in labels:
        s8 = chain[:, 6] * np.sqrt(chain[:, 0] / 0.3)
        chain = np.hstack([chain[:, :-1], s8[:, np.newaxis], chain[:, -1][:, np.newaxis]])
        names.append('S8')
        labels.append('S_8')
    indices = list(range(len(labels)))
    samps = chain[start_chain:, indices] if not oms8 else chain[start_chain:, [0,1]]
    weights = chain[start_chain:, -1]
    return MCSamples(samples=samps, names=names, labels=labels, weights=weights)


def load_chain(model, target, chain_path, suffix=""):
    """
    Load a chain file for a given model and target from the specified path.

    Parameters:
        model (str): Name of the model
        target (str): Name of the target
        chain_path (str): Path to the directory containing chain files
        suffix (str): Suffix to append to the chain file name, if applicable

    Returns:
        np.ndarray or None: Loaded chain data as a NumPy array, or None if the file does not exist or is empty
    """
    file_path = os.path.join(chain_path, f"{model}_chains", f"{model}_with_{target}{suffix}.txt")
    return np.loadtxt(file_path) if os.path.exists(file_path) and os.path.getsize(file_path) > 0 else None


# Configurations we investigated

# Path to chain files
chain_path = "./output/chains/"

# The dictionary to map models to their labels and truth values
# Note that the order matters for the labels and truth values
labels_dict = {
    "TATT": ["\\Omega_\\mathrm{m}", "S_8", "A_\\mathrm{IA}", "C_2", "b_\\mathrm{TA}"],
    "TATT_w": ["\\Omega_\\mathrm{m}", "w", "S_8", "A_\\mathrm{IA}", "C_2", "b_\\mathrm{TA}"],
    "NLA": ["\\Omega_\\mathrm{m}", "S_8", "A_\\mathrm{IA}"],
    "NLA_w": ["\\Omega_\\mathrm{m}", "w", "S_8", "A_\\mathrm{IA}"],
    "deltaNLA": ["\\Omega_\\mathrm{m}", "S_8", "A_\\mathrm{IA}", "b_\\mathrm{TA}"],
    "TT": ["\\Omega_\\mathrm{m}", "S_8", "C_2"],
    "deltaTT": ["\\Omega_\\mathrm{m}", "S_8", "C_2", "b_\\mathrm{TA}"],
}

# Truth values from the Outer Rim cosmology simulations (Heitmann et al. 2019)
truth_values_dict = {
    "NLA": [0.2648, 0.7525, 1.0],  # omega_m, S_8, A_IA
    "deltaNLA": [0.2648, 0.7525, 1.0, 1.0],  # omega_m, S_8, A_IA, b_TA
    "TT": [0.2648, 0.7525, 0.0],  # omega_m, S_8, C_2
    "deltaTT": [0.2648, 0.7525, 0.0, 1.0],  # omega_m, S_8, C_2, b_TA
    "TATT": [0.2648, 0.7525, 1.0, 0.0, 1.0]  # omega_m, S_8, A_IA, C_2, b_TA
}

# Models and targets
# For example, this will mean we have a chain called "NLA_with_TATT_w.txt"
# for the NLA model with the TATT target and an extra suffix "_w"
models_targets = {
    "NLA": ["NLA", "TATT", "NLA_w", "TATT_w"],
    "deltaNLA": ["deltaNLA", "TATT"],
    "TT": ["TT", "TATT"],
    "deltaTT": ["TT", "deltaTT", "TATT"],
    "HOD_NLA": ["deltaNLA", "TATT"],
    "HOD_TT": ["TT", "deltaTT", "TATT"],
}

# Load everything dynamically as a dictionaries of chains, labels, truth values, and samples
chains, labels, truth_values, samples = {}, {}, {}, {}
for model, targets in models_targets.items():
    chains[model] = {}
    for target in targets:
        suffix = "_w" if target.endswith("_w") else ""
        target_base = target.replace("_w", "")
        chain = load_chain(model, target_base, chain_path, suffix)
        if chain is not None:
            key = f"{model}_with_{target}"
            chains[model][key] = chain
            labels[key] = labels_dict[target_base]
            if suffix:
                # if w is varied, we add it to the labels
                # Note that we have to add it in the right position
                labels[key] = labels[key][:2] + ["w"] + labels[key][2:]
            truth_values[key] = truth_values_dict[target_base]
            if suffix:
                truth_values[key] = truth_values[key][:1] + [-1.0] + truth_values[key][1:]
            samples[key] = get_mc_samples(chain, labels[key], add_s8=True, burn_in=0.0, oms8=False)
