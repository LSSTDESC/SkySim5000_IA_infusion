import helpers.chains_processing as cp
import matplotlib.pyplot as plt
from getdist import plots


truth_values = cp.truth_values  # dictionary with the truth values
labels = cp.labels  # dictionary with the labels
samples = cp.samples  # dictionary with the chains


def get_kwargs(labels, colors, lw=2.5):
    kwargs = {
        'contour_colors': colors,
        "filled": [False] * len(labels),
        "contour_ls": ["-"] * len(labels),
        "contour_lws": [lw] * len(labels),
    }
    return kwargs


# Plotting settings
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


def plot_path():
    path = "./output/corner_plots/"
    return path


def corner_colors(input_key):
    """
    Return the color for a given key, dynamically adding combinations with TATT.

    Parameters:
        input_key (str): The key for which to retrieve the color (e.g., "NLA_with_TATT").

    Returns:
        str: The color associated with the given key (e.g., "#ffc000").
    """
    # Base color definitions
    base_colors = {
        "NLA_with_NLA": '#329acd',  # blue
        "deltaNLA_with_deltaNLA": '#800080',  # purple
        "TT_with_TT": '#ff7f0e',  # orange
        "deltaTT_with_deltaTT": '#ff0000',  # red
        "deltaTT_with_TT": '#ff0000',  # red
        "HOD_NLA_with_deltaNLA": '#9acd32',  # green
        "HOD_TT_with_TT": 'deeppink',  # hotpink
        "HOD_TT_with_deltaTT": 'hotpink',  # pink
    }

    # Add combinations with TATT
    extended_colors = base_colors.copy()
    for key in base_colors:
        base_model = key.split("_with_")[0]  # Extract the base model (e.g., NLA)
        tatt_key = f"{base_model}_with_TATT"
        extended_colors[tatt_key] = '#ffc000'  # Yellow for TATT combinations

    # Return the color for the given input key
    return extended_colors.get(input_key, '#000000')  # Default to black if the key is not found


def get_colors():
    """
    Return a dictionary of colors for various keys, dynamically adding combinations with TATT.

    Returns:
        dict: A dictionary mapping keys to their associated colors.
    """
    # Base color definitions
    base_colors = {
        "NLA_with_NLA": '#329acd',  # blue
        "deltaNLA_with_deltaNLA": '#800080',  # purple
        "TT_with_TT": '#ff7f0e',  # orange
        "deltaTT_with_deltaTT": '#ff0000',  # red
        "deltaTT_with_TT": '#ff0000',  # red
        "HOD_NLA_with_deltaNLA": '#9acd32',  # green
        "HOD_TT_with_TT": 'deeppink',  # hotpink
        "HOD_TT_with_deltaTT": 'hotpink',  # pink
    }

    # Add combinations with TATT
    extended_colors = base_colors.copy()
    for key in base_colors:
        base_model = key.split("_with_")[0]  # Extract the base model (e.g., NLA)
        tatt_key = f"{base_model}_with_TATT"
        extended_colors[tatt_key] = '#ffc000'  # Yellow for TATT combinations

    return extended_colors


def get_legend_labels(sample1_key, sample2_key, sample3_key=None):
    """
    Generate legend labels dynamically based on the provided sample keys.

    Parameters:
        sample1_key: Key for the first sample (e.g., "HOD_NLA_with_deltaNLA").
        sample2_key: Key for the second sample (e.g., "HOD_NLA_with_TATT").
        sample3_key: Optional. Key for the third sample (e.g., "HOD_NLA_with_TT").

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

    # Add sample3 if provided
    if sample3_key:
        model3_1, model3_2 = format_model_name(sample3_key)
        legend_labels += [
            "_nolegend_",
            f"{model3_1} data with {model3_2} model",
        ]

    return legend_labels


def plot_triangle_combination(sample1_key, sample2_key, sample3_key=None, save_fig=True):
    """
    Generate and save a triangle plot comparing two or three samples dynamically.

    Parameters:
        sample1_key: Key for the first sample (mandatory, e.g., "HOD_NLA_with_deltaNLA").
        sample2_key: Key for the second sample (mandatory, e.g., "HOD_NLA_with_TATT").
        sample3_key: Optional. Key for the third sample (e.g., "HOD_NLA_with_TT").
        save_fig: Boolean flag to save the figure as a PDF.
    """
    # Collect samples and labels
    sample_keys = [
        samples[sample2_key],  # First sample
        samples[sample2_key],  # First sample
        samples[sample1_key],  # Second sample
        samples[sample1_key],  # Second sample
    ]

    # Add sample3 if provided
    if sample3_key:
        sample_keys.extend([samples[sample3_key], samples[sample3_key]])

    # Generate legend labels using the helper function
    legend_labels = get_legend_labels(sample1_key, sample2_key, sample3_key)

    # Dynamically configure colors
    colors_list = [
        corner_colors(sample2_key),
        corner_colors(sample2_key),
        corner_colors(sample1_key),
        corner_colors(sample1_key),
    ]
    if sample3_key:
        colors_list.extend([corner_colors(sample3_key), corner_colors(sample3_key)])

    # Plot parameters
    kwargs = {
        'contour_colors': colors_list,
        "filled": [False, True] * (len(colors_list) // 2),
        "contour_ls": ["-"] * len(colors_list),
        "contour_lws": [2] * len(colors_list),
    }

    g = plots.get_subplot_plotter(width_inch=8)
    for setting, value in plot_settings.items():
        setattr(g.settings, setting, value)

    # Use the truth values from the first sample
    truth_values_key = sample2_key  # Adjusted to use sample2's truth values

    # Generate the triangle plot
    g.triangle_plot(
        sample_keys,
        markers=truth_values[truth_values_key],
        legend_labels=legend_labels,
        legend_ncol=1,
        **kwargs,
    )

    # Save the plots
    if save_fig:
        keys_used = [key for key in [sample2_key, sample1_key, sample3_key] if key]
        filename = "_".join([key.replace("_with_", "-") for key in keys_used]) + "_combo.pdf"
        plt.savefig(f"{plot_path()}{filename}", dpi=300)

def get_plot_settings():
    g = plots.get_subplot_plotter(width_inch=8)
    for setting, value in plot_settings.items():
        setattr(g.settings, setting, value)

    return g
