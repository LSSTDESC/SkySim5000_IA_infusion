import helpers.chains_processing as cp

from getdist import plots


truth_values = cp.truth_values  # dictionary with the truth values
labels = cp.labels  # dictionary with the labels
samples = cp.samples  # dictionary with the chains


def get_kwargs(key1, key2):
    """
    Generate keyword arguments for contour plots based on two keys.

    Parameters:
        key1 (str): the key of the first model (e.g., "NLA" or "TATT")
        key2 (str): the key of the second model (e.g., "NLA" or "TATT")

    Returns:
        dict: keyword arguments for contour plots, including colors,
              filled status, line styles, and line widths.
    """
    colors = get_colors()
    colors_list = [colors[key1]] * 2 + [colors[key2]] * 2

    kwargs = {
        'contour_colors': colors_list,
        "filled": [False, True] * (len(colors_list)),
        "contour_ls": ["-"] * len(colors_list),
        "contour_lws": [2] * len(colors_list),
    }
    return kwargs


def get_legend_labels(sample1_key, sample2_key):
    """
    Generate legend labels dynamically based on the provided sample keys.

    Parameters:
        sample1_key: Key for the first sample (e.g., "HOD_NLA_with_deltaNLA").
        sample2_key: Key for the second sample (e.g., "HOD_NLA_with_TATT").

    Returns:
        list: Legend labels for the triangle plot.
    """
    def format_model_name(key):
        """Extract and format model names from the sample key."""
        model1, model2 = key.split("_with_")
        formatted_model1 = f"${{\\delta}}$" + model1[5:] if model1.startswith("delta") else \
                           f"HOD {model1[4:]}" if model1.startswith("HOD") else model1
        formatted_model2 = f"${{\\delta}}$" + model2[5:] if model2.startswith("delta") else \
                           f"HOD {model2[4:]}" if model2.startswith("HOD") else model2
        return formatted_model1, formatted_model2

    # Format models for sample1 and sample2
    model1_1, model1_2 = format_model_name(sample1_key)
    model2_1, model2_2 = format_model_name(sample2_key)

    # Generate legend labels for sample1 and sample2
    legend_labels = [
        "_nolegend_",
        f"{model1_1} data with {model1_2} model",
        "_nolegend_",
        f"{model2_1} data with {model2_2} model",
    ]

    return legend_labels


def plot_path():
    """
    Return the path where corner plots are saved.

    Returns:
        str: Path to the directory for corner plots.
    """
    path = "./output/corner_plots/"
    return path


def get_colors():
    """
    Return a dictionary of colors for keys that correspond to different models (file names).
    """
    base_colors = {
        "TATT": '#ffc000',  # yellow
        "NLA_with_NLA": '#329acd',
        "deltaNLA_with_deltaNLA": '#a64da6',
        "TT_with_TT": '#ff7f0e',
        "deltaTT_with_deltaTT": '#ff3333',
        "deltaTT_with_TT": '#ff0000',
        "HOD_NLA_with_deltaNLA": '#a64da6',
        "HOD_TT_with_TT": '#ff7f0e',
        "HOD_TT_with_deltaTT": '#ff3333',
    }

    extended_colors = {}
    for key in base_colors.keys():
        extended_colors[key] = base_colors[key]
        # Also add a _with_TATT version for each
        base_model = key.split("_with_")[0]
        extended_colors[f"{base_model}_with_TATT"] = '#ffc000'

    return extended_colors


def get_plot_settings():
    """
    Create a GetDist plotter with custom settings.

    Returns:
        g (GetDistPlotter): A GetDist plotter object with custom settings.
    """
    g = plots.get_subplot_plotter(width_inch=8)

    # Plot settings
    plot_settings = {
        'linewidth': 2.5,  # Line width for plots and contours
        'scaling_factor': False,  # Disable automatic scaling
        'alpha_filled_add': 0.85,  # Opacity for filled elements
        'axes_labelsize': 20,  # Font size for axes labels
        'legend_rect_border': False,  # Disable border around legend rectangles
        'axes_fontsize': 15,  # Font size for axes ticks
        'figure_legend_frame': False,  # Disable frame around figure legend
        'legend_fontsize': 14,  # Font size for legend
        'linewidth_contour': 2.5,  # Line width for contour lines
        'legend_frac_subplot_margin': 0.1,  # Margin fraction for subplot legends
        'line_styles': ["-", "-", "-", "-", "-", "-", "-", "-"],  # Line styles for plots
        'solid_contour_palefactor': 0.45,  # Pale factor for solid contours
        'axis_marker_color': 'k',  # Color for markers
        'axis_marker_ls': ':',  # Line style for markers
        'axis_marker_lw': 1.5,  # Line width for markers
    }

    for setting, value in plot_settings.items():
        setattr(g.settings, setting, value)

    return g
