"""
Unified figure style configuration for the manuscript.
All figures should import this module to ensure consistent styling.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# ============================================================================
# COMMON MATH LABELS (Matplotlib mathtext)
# ============================================================================

# Sound pressure level notation: italic L with upright Aeq subscript.
MATH_L_AEQ = r"$L_{\mathrm{Aeq}}$"
MATH_DELTA_L_AEQ = r"$\Delta L_{\mathrm{Aeq}}$"

# PM2.5 notation: upright PM with upright subscript.
MATH_PM25_INTERIOR = r"$\mathrm{PM}_{2.5,\mathrm{interior}}$"
MATH_PM25_P1 = r"$\mathrm{PM}_{2.5,\mathrm{P1}}$"
MATH_LOG_PM25_RATIO = (
    r"$\log\left(\mathrm{PM}_{2.5,\mathrm{interior}}/\mathrm{PM}_{2.5,\mathrm{P1}}\right)$"
)

# ============================================================================
# COLOR PALETTE - Academic style using a cohesive blue-orange-gray scheme
# ============================================================================

# Primary colors for main data categories
COLORS = {
    # Main category colors
    'large': '#2166AC',      # Deep blue for Large parks
    'medium': '#67A9CF',     # Medium blue for Medium parks
    'small': '#EF8A62',      # Coral/orange for Small parks

    # Grayscale
    'primary': '#2C3E50',    # Dark blue-gray for main elements
    'secondary': '#7F8C8D',  # Medium gray
    'light_gray': '#BDC3C7', # Light gray for reference lines

    # Accent colors
    'highlight': '#E74C3C',  # Red for emphasis/failure
    'success': '#27AE60',    # Green for success/compliance
    'warning': '#F39C12',    # Orange for warnings

    # Quantile regression palette
    'q10': '#2166AC',        # Darkest blue
    'q25': '#67A9CF',        # Medium blue
    'q50': '#1A1A1A',        # Black for median
    'q75': '#FDDBC7',        # Light orange
    'q90': '#EF8A62',        # Orange

    # ROC curve
    'roc_curve': '#2166AC',  # Blue
    'roc_reference': '#BDC3C7',  # Gray dashed
    'optimal_point': '#E74C3C',  # Red

    # Bayesian colors
    'posterior': '#2166AC',   # Blue for posterior
    'credible_interval': '#67A9CF',  # Light blue for CI
    'zero_line': '#E74C3C',   # Red for reference
}

# Park type color mapping
PARK_TYPE_COLORS = {
    'Large': COLORS['large'],
    'Medium': COLORS['medium'],
    'Small': COLORS['small'],
}

# Quantile colors for gradient visualization
QUANTILE_COLORS = [COLORS['q10'], COLORS['q25'], COLORS['q50'], COLORS['q75'], COLORS['q90']]

# ============================================================================
# FIGURE STYLE CONFIGURATION
# ============================================================================

def setup_style():
    """Configure matplotlib for consistent academic figure style."""

    # Use a clean, modern style
    plt.style.use('seaborn-v0_8-white')

    # Font settings - use sans-serif for clarity
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    mpl.rcParams['font.size'] = 10
    mpl.rcParams['axes.titlesize'] = 11
    mpl.rcParams['axes.labelsize'] = 10
    mpl.rcParams['xtick.labelsize'] = 9
    mpl.rcParams['ytick.labelsize'] = 9
    mpl.rcParams['legend.fontsize'] = 9
    mpl.rcParams['legend.title_fontsize'] = 10

    # Figure settings
    mpl.rcParams['figure.dpi'] = 300
    mpl.rcParams['savefig.dpi'] = 300
    mpl.rcParams['savefig.bbox'] = 'tight'
    mpl.rcParams['savefig.pad_inches'] = 0.1
    mpl.rcParams['figure.facecolor'] = 'white'
    mpl.rcParams['axes.facecolor'] = 'white'

    # Line settings
    mpl.rcParams['lines.linewidth'] = 1.5
    mpl.rcParams['lines.markersize'] = 6

    # Grid settings
    mpl.rcParams['axes.grid'] = False
    mpl.rcParams['grid.alpha'] = 0.3
    mpl.rcParams['grid.linestyle'] = '-'
    mpl.rcParams['grid.linewidth'] = 0.5

    # Axes settings
    mpl.rcParams['axes.linewidth'] = 0.9
    mpl.rcParams['axes.edgecolor'] = 'black'
    mpl.rcParams['axes.labelcolor'] = 'black'
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.right'] = False

    # Tick settings
    mpl.rcParams['xtick.direction'] = 'out'
    mpl.rcParams['ytick.direction'] = 'out'
    mpl.rcParams['xtick.major.width'] = 0.8
    mpl.rcParams['ytick.major.width'] = 0.8
    mpl.rcParams['xtick.color'] = 'black'
    mpl.rcParams['ytick.color'] = 'black'

# ============================================================================
# FIGURE SIZE CONSTANTS
# ============================================================================

# Standard figure widths (in inches) for single/double column
SINGLE_COL_WIDTH = 3.5     # ~89 mm, typical single column
DOUBLE_COL_WIDTH = 7.0     # ~178 mm, full page width
GOLDEN_RATIO = 1.618

# Common figure sizes
FIG_SIZES = {
    'single': (SINGLE_COL_WIDTH, SINGLE_COL_WIDTH / GOLDEN_RATIO),
    'single_square': (SINGLE_COL_WIDTH, SINGLE_COL_WIDTH),
    'double': (DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH / GOLDEN_RATIO),
    'double_tall': (DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.7),
    'double_square': (DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.5),
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def add_subplot_label(ax, label, x=-0.12, y=1.05, fontsize=12, fontweight='bold'):
    """Add a subplot label (a), (b), etc. to an axes."""
    ax.text(x, y, f'({label})', transform=ax.transAxes,
            fontsize=fontsize, fontweight=fontweight,
            va='bottom', ha='right')

def format_axis_labels(ax, xlabel=None, ylabel=None):
    """Format axis labels with consistent styling."""
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, labelpad=8)

def add_reference_line(ax, value=0, orientation='h', **kwargs):
    """Add a reference line to the plot."""
    defaults = {'color': COLORS['light_gray'], 'linestyle': '--', 'linewidth': 1, 'zorder': 0}
    defaults.update(kwargs)
    if orientation == 'h':
        ax.axhline(y=value, **defaults)
    else:
        ax.axvline(x=value, **defaults)

def create_figure_with_subplots(nrows=1, ncols=1, figsize=None, subplot_labels=True):
    """Create a figure with consistent styling and optional subplot labels."""
    if figsize is None:
        if nrows == 1 and ncols == 1:
            figsize = FIG_SIZES['single']
        else:
            figsize = FIG_SIZES['double']

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    if subplot_labels and (nrows > 1 or ncols > 1):
        axes_flat = np.array(axes).flatten()
        labels = 'abcdefghijklmnopqrstuvwxyz'
        for i, ax in enumerate(axes_flat):
            add_subplot_label(ax, labels[i])

    return fig, axes

def save_figure(fig, filename, formats=['png', 'pdf']):
    """Save figure in multiple formats."""
    for fmt in formats:
        fig.savefig(f'{filename}.{fmt}', format=fmt, dpi=300, bbox_inches='tight')
    plt.close(fig)

# Initialize style when module is imported
setup_style()
