from getdist import MCSamples
import numpy as np
import os
import re


def get_mc_samples(chain, labels, add_s8=True, burn_in=0.0, oms8=False):
    """
    Generate Monte Carlo samples from a given chain with options to add S_8 parameter,
    apply a burn-in period, and select specific parameters (Omega_m and S_8).

    Parameters:
    - chain (np.array): The MCMC chain from which samples are generated. Each row is a sample.
    - labels (list of str): LaTeX labels for each parameter in the chain.
    - add_s8 (bool): If True, add the S_8 parameter to the chain as a new parameter.
    - burn_in (float): Fraction of the chain to discard as burn-in. E.g., 0.1 to discard the first 10%.
    - oms8 (bool): If True, only return samples for the Omega_m and S_8 parameters.

    Returns:
    - samples (MCSamples object): An object containing the Monte Carlo samples, parameter names,
      labels, and weights for the samples after applying burn-in and any parameter modifications.
    """

    # Determine the length of the chain
    chain_len = np.shape(chain)
    # Generate indices based on the number of labels
    indices = list(range(len(labels)))
    names = []

    for label in labels:
        # Remove LaTeX commands and unnecessary characters to clean up the label names
        label = re.sub(r'\\mathrm{([^}]*)}', r'\1', label)  # Convert \mathrm{} to plain text
        label = re.sub(r'\\[^a-zA-Z]*', '', label)  # Remove other LaTeX commands
        label = re.sub(r'[{}]', '', label)  # Remove braces
        label = label.replace('_', '')  # Remove underscores for variable naming
        names.append(label)

    # Optionally add S_8 parameter to the chain
    if add_s8:
        chain_s8 = np.zeros([chain_len[0], 1])
        chain_s8[:, 0] = chain[:, 6] * np.sqrt(chain[:, 0] / 0.3)  # Calculate S_8 values
        chain = np.hstack([chain, chain_s8])  # Add as the last column

    # Calculate start index for chain after applying burn-in
    start_chain = int(np.floor(chain_len[0] * burn_in))

    # Select samples based on oms8 flag
    if oms8:
        samps = chain[start_chain:, indices[:2]]  # Only use Omega_m and S_8 parameters
        names = names[:2]  # Update names to reflect parameter selection
        labels = labels[:2]  # Update labels accordingly
    else:
        samps = chain[start_chain:, indices]  # Use all parameters

    # Extract weights for the samples
    weights = chain[start_chain:, chain_len[1] - 1]

    # Create MCSamples object with the processed samples and information
    samples = MCSamples(samples=samps,
                        names=names,
                        labels=labels,
                        weights=weights)

    return samples

# Path to chain files
chain_path = "./output/chains/"

# Base models and their associated targets
base_targets = {
    "NLA": ["NLA", "TATT"],
    "deltaNLA": ["deltaNLA", "TATT"],
    "TT": ["TT", "TATT"],
    "deltaTT": ["deltaTT", "TATT"],
    "HOD_NLA": ["deltaNLA", "TATT"],
    "HOD_TT": ["TT", "deltaTT", "TATT"],
}

# Dynamically generate chains and targets
chains = {}
targets = {}
for model, model_targets in base_targets.items():
    model_path = os.path.join(chain_path, f"{model}_chains")
    chains[model] = {}
    targets[model] = []
    for target in model_targets:
        file_name = f"{model}_with_{target}"
        file_path = os.path.join(model_path, f"{file_name}.txt")
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            chains[model][file_name] = np.loadtxt(file_path)
            targets[model].append(target)  # Add target only if the file exists
        else:
            print(f"Warning: File {file_path} for model {model} is missing or empty. Skipping.")

# Define parameter labels for IA models
labels_dict = {
    "TATT": ["\\Omega_\\mathrm{m}", "S_8", "A_\\mathrm{IA}", "C_2", "b_\\mathrm{TA}"],
    "NLA": ["\\Omega_\\mathrm{m}", "S_8", "A_\\mathrm{IA}"],
    "deltaNLA": ["\\Omega_\\mathrm{m}", "S_8", "A_\\mathrm{IA}", "b_\\mathrm{TA}"],
    "TT": ["\\Omega_\\mathrm{m}", "S_8", "C_2"],
    "deltaTT": ["\\Omega_\\mathrm{m}", "S_8", "C_2", "b_\\mathrm{TA}"],
}

labels = {f"{model}_with_{target}": labels_dict[target] for model in chains.keys() for target in targets[model]}


# Generate truth values
truth_values = {
    f"{model}_with_{target}": [
        0.2648,  # Example value for Omega_m
        0.7525,  # Example value for S_8
        *([1.0] if "NLA" in model else [0.0]),  # A_IA
        *([0.0] if "NLA" in model else [1.0]),  # C_2
        *([1.0] if "delta" in model else [0.0]),  # b_TA
    ]
    for model in chains.keys()
    for target in targets[model]
}


# Generate MCSamples for all models
samples = {
    f"{model}_with_{target}": get_mc_samples(chains[model][f"{model}_with_{target}"], labels[f"{model}_with_{target}"])
    for model in chains.keys()
    for target in targets[model]
    if f"{model}_with_{target}" in chains[model]
}
