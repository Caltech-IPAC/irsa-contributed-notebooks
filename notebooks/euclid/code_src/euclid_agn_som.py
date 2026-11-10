"""Helper functions for ``euclid_agn_som.md``.

The notebook keeps the data access and the science choices inline.
This module holds the routine pieces that would distract from them:
flux-to-magnitude conversion, per-cell statistics on the self-organizing map (SOM) grid,
and the plotting overlays used in the SOM figures.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from scipy.interpolate import NearestNDInterpolator
from scipy.ndimage import gaussian_filter

# Colormaps of the overlays. They go from transparent to dark green, so the SOM background stays visible.
AGN_CMAP = LinearSegmentedColormap.from_list('agn_density_green', [
    (0.82, 0.96, 0.70, 0.34),
    (0.66, 0.88, 0.58, 0.58),
    (0.28, 0.68, 0.35, 0.78),
    (0.00, 0.33, 0.18, 0.96),
])
METHOD_CMAP = LinearSegmentedColormap.from_list('method_excess_green', [
    (1.0, 1.0, 1.0, 0.00),
    (0.86, 0.96, 0.78, 0.34),
    (0.36, 0.74, 0.42, 0.72),
    (0.00, 0.33, 0.18, 0.95),
])
OVERLAY_COLOR = '#064d2c'

# Width (in SOM cells) of the Gaussian used to smooth the AGN counts per cell before plotting.
SMOOTH_SIGMA = 1.35


# ---------------------------------------------------------------------------
# Photometry
# ---------------------------------------------------------------------------
def as_float_array(column):
    """Convert a table column to a plain float array.

    Parameters
    ----------
    column : astropy.table.Column, astropy.table.MaskedColumn or array-like
        Values to convert. Masked entries (for example the missing values of a TAP result) become NaN.

    Returns
    -------
    numpy.ndarray
        Float array of the same length as ``column``.
    """
    return np.asarray(np.ma.filled(np.ma.asarray(column, dtype=float), np.nan), dtype=float)


def any_flag_set(table, flag_columns):
    """Find the rows of a table with at least one of several flag columns set.

    Parameters
    ----------
    table : astropy.table.Table
        Table with flag columns that are 1 when the flag is set.
    flag_columns : list of str
        Names of the flag columns. Names that are not in the table are skipped.

    Returns
    -------
    numpy.ndarray
        Boolean array, True for the rows with at least one of the flags set.
    """
    result = np.zeros(len(table), dtype=bool)
    for column in flag_columns:
        if column in table.colnames:
            result |= np.asarray(table[column]) == 1
    return result


def abmag(flux_ujy):
    """Convert fluxes in microjansky to AB magnitudes.

    Parameters
    ----------
    flux_ujy : array-like
        Fluxes in microjansky, the unit of the fluxes in the Euclid MER catalogs.

    Returns
    -------
    numpy.ndarray
        AB magnitudes, ``-2.5 log10(flux) + 23.9``. NaN where the flux is missing, zero or negative.
    """
    flux = as_float_array(flux_ujy)
    mag = np.full(flux.shape, np.nan)
    good = np.isfinite(flux) & (flux > 0)
    mag[good] = -2.5 * np.log10(flux[good]) + 23.9
    return mag


# ---------------------------------------------------------------------------
# Statistics on the SOM grid
# ---------------------------------------------------------------------------
def find_bmu(som, data):
    """Find the best-matching unit (BMU) of each source on a trained SOM.

    The BMU of a source is the SOM cell whose weight vector is closest to the colors of the source.

    Parameters
    ----------
    som : minisom.MiniSom
        Trained SOM.
    data : numpy.ndarray
        Normalized colors, shape (n_sources, n_colors), with no NaN.

    Returns
    -------
    bmu : numpy.ndarray
        Integer array of shape (n_sources, 2) with the (row, column) of the BMU of each source.
    distance : numpy.ndarray
        Distance of each source to the weight vector of its BMU (its quantization error).
    """
    weights = som.get_weights()
    n_rows, n_cols, n_colors = weights.shape
    flat_weights = weights.reshape(-1, n_colors)
    bmu = np.empty((len(data), 2), dtype=int)
    distance = np.empty(len(data))
    # Work in chunks so the (n_sources, n_cells) distance array stays small.
    for start in range(0, len(data), 5000):
        chunk = data[start:start + 5000]
        dist = np.linalg.norm(chunk[:, None, :] - flat_weights[None, :, :], axis=2)
        best = np.argmin(dist, axis=1)
        bmu[start:start + 5000] = np.column_stack(np.unravel_index(best, (n_rows, n_cols)))
        distance[start:start + 5000] = dist[np.arange(len(chunk)), best]
    return bmu, distance


def count_map(bmu, shape):
    """Count the sources in each SOM cell.

    Parameters
    ----------
    bmu : numpy.ndarray
        Integer array of shape (n_sources, 2) with the (row, column) of the BMU of each source.
    shape : tuple of int
        Shape (n_rows, n_columns) of the SOM grid.

    Returns
    -------
    numpy.ndarray
        Array of shape ``shape`` with the number of sources in each cell.
    """
    counts = np.zeros(shape)
    np.add.at(counts, (bmu[:, 0], bmu[:, 1]), 1)
    return counts


def median_map(values, bmu, shape):
    """Median of a quantity in each SOM cell.

    Parameters
    ----------
    values : array-like
        One value per source. NaN values are ignored.
    bmu : numpy.ndarray
        Integer array of shape (n_sources, 2) with the (row, column) of the BMU of each source.
    shape : tuple of int
        Shape (n_rows, n_columns) of the SOM grid.

    Returns
    -------
    numpy.ndarray
        Array of shape ``shape`` with the median of ``values`` in each cell. NaN in the cells with no finite value.
    """
    values = as_float_array(values)
    cell = bmu[:, 0] * shape[1] + bmu[:, 1]
    result = np.full(shape[0] * shape[1], np.nan)
    good = np.isfinite(values)
    for index in np.unique(cell[good]):
        result[index] = np.median(values[good & (cell == index)])
    return result.reshape(shape)


def mean_map(values, bmu, shape):
    """Mean of a quantity in each SOM cell.

    Parameters
    ----------
    values : array-like
        One value per source, with no NaN. Use 0 for the sources without a measurement
        to count them in the mean as non-detections.
    bmu : numpy.ndarray
        Integer array of shape (n_sources, 2) with the (row, column) of the BMU of each source.
    shape : tuple of int
        Shape (n_rows, n_columns) of the SOM grid.

    Returns
    -------
    numpy.ndarray
        Array of shape ``shape`` with the mean of ``values`` in each cell. NaN in the cells with no source.
    """
    counts = count_map(bmu, shape)
    total = np.zeros(shape)
    np.add.at(total, (bmu[:, 0], bmu[:, 1]), np.asarray(values, dtype=float))
    result = np.full(shape, np.nan)
    result[counts > 0] = total[counts > 0] / counts[counts > 0]
    return result


def fill_empty_cells(cell_map):
    """Fill the empty (NaN) cells of a SOM map with the value of the nearest filled cell.

    This is only for display, so that a background map has no holes.

    Parameters
    ----------
    cell_map : numpy.ndarray
        2D map with NaN in the empty cells.

    Returns
    -------
    numpy.ndarray
        Copy of ``cell_map`` with the NaN cells filled.
    """
    cell_map = np.asarray(cell_map, dtype=float)
    filled = np.isfinite(cell_map)
    if filled.all() or not filled.any():
        return cell_map.copy()
    rows, cols = np.indices(cell_map.shape)
    interpolator = NearestNDInterpolator(np.column_stack([rows[filled], cols[filled]]), cell_map[filled])
    return interpolator(rows, cols)


def jitter_bmu(bmu, jitter=0.34, seed=0):
    """Plot coordinates for sources on the SOM, randomly shifted inside their cell.

    All the sources of a cell have the same BMU, so without a shift their markers would overlap.

    Parameters
    ----------
    bmu : numpy.ndarray
        Integer array of shape (n_sources, 2) with the (row, column) of the BMU of each source.
    jitter : float, optional
        Maximum shift in units of the cell size.
    seed : int, optional
        Seed of the random number generator.

    Returns
    -------
    x, y : numpy.ndarray
        Plot coordinates (column and row) of each source.
    """
    rng = np.random.default_rng(seed)
    x = bmu[:, 1] + rng.uniform(-jitter, jitter, len(bmu))
    y = bmu[:, 0] + rng.uniform(-jitter, jitter, len(bmu))
    return x, y


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def format_som_axis(ax, shape):
    """Set the limits of a SOM panel and remove its ticks.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to format.
    shape : tuple of int
        Shape (n_rows, n_columns) of the SOM grid.
    """
    ax.set_xlim(-0.5, shape[1] - 0.5)
    ax.set_ylim(-0.5, shape[0] - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])


def overlay_density(ax, counts, smooth=SMOOTH_SIGMA, low_percentile=18, high_percentile=98):
    """Draw the smoothed number of sources per SOM cell on top of an existing SOM image.

    Only the cells above ``low_percentile`` of the smoothed counts are drawn, so the background
    stays visible where there are few sources.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes that already show a SOM background.
    counts : numpy.ndarray
        Number of sources in each SOM cell (see `count_map`).
    smooth : float, optional
        Width of the Gaussian smoothing, in cells.
    low_percentile, high_percentile : float, optional
        Percentiles of the smoothed counts used as the limits of the color scale.

    Returns
    -------
    matplotlib.image.AxesImage or None
        The overlay image, for a colorbar. None if ``counts`` is empty.
    """
    if counts.sum() == 0:
        return None
    smoothed = gaussian_filter(counts, sigma=smooth)
    vmin, vmax = np.percentile(smoothed[smoothed > 0], [low_percentile, high_percentile])
    vmin = max(vmin, 0.08)
    vmax = max(vmax, 1.35 * vmin)
    image = ax.imshow(np.ma.masked_less(smoothed, vmin), origin='lower', cmap=AGN_CMAP,
                      norm=LogNorm(vmin=vmin, vmax=vmax), interpolation='bilinear')
    ax.contour(smoothed, levels=np.geomspace(vmin, vmax, 8), colors=OVERLAY_COLOR, linewidths=0.55, alpha=0.68)
    return image


def overlay_method_excess(ax, counts_method, counts_all, smooth=SMOOTH_SIGMA, min_excess=1.25):
    """Draw where one AGN selection method is over-represented compared with all AGN candidates.

    The excess of a cell is the smoothed number of candidates of the method in the cell, divided by the
    number expected if the method followed the distribution of all the AGN candidates on the SOM
    (the smoothed counts of all candidates, scaled to the total number of candidates of the method).
    An excess of 2 means the method selects twice as many candidates in that cell as expected.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes that already show a SOM background.
    counts_method : numpy.ndarray
        Number of candidates of the method in each SOM cell.
    counts_all : numpy.ndarray
        Number of all AGN candidates in each SOM cell.
    smooth : float, optional
        Width of the Gaussian smoothing, in cells.
    min_excess : float, optional
        Only the cells with an excess above this value are drawn.

    Returns
    -------
    matplotlib.image.AxesImage or None
        The overlay image, for a colorbar. None if the method has no candidate or no cell is above ``min_excess``.
    """
    if counts_method.sum() == 0 or counts_all.sum() == 0:
        return None
    smoothed_method = gaussian_filter(counts_method, sigma=smooth)
    expected = gaussian_filter(counts_all, sigma=smooth) * counts_method.sum() / counts_all.sum()
    # The small constant keeps the ratio stable in cells with very few candidates;
    # cells where almost none are expected are left out.
    excess = np.where(expected >= 0.03, (smoothed_method + 0.02) / (expected + 0.02), np.nan)
    shown = excess[np.isfinite(excess) & (excess >= min_excess)]
    if shown.size == 0:
        return None
    vmax = max(np.percentile(shown, 98), 1.2 * min_excess)
    hidden = ~np.isfinite(excess) | (excess < min_excess)
    image = ax.imshow(np.ma.masked_where(hidden, excess), origin='lower', cmap=METHOD_CMAP,
                      norm=LogNorm(vmin=min_excess, vmax=vmax), interpolation='bilinear')
    ax.contour(excess, levels=np.geomspace(min_excess, vmax, 4), colors=OVERLAY_COLOR, linewidths=0.35, alpha=0.35)
    return image


def add_panel_colorbar(ax, image, label):
    """Add a small colorbar to the right of a panel.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Panel to add the colorbar to.
    image : matplotlib.image.AxesImage
        Image the colorbar describes.
    label : str
        Label of the colorbar.
    """
    # The color scale is logarithmic: label a few round values instead of powers of ten.
    vmin, vmax = image.get_clim()
    ticks = [tick for tick in (1.25, 1.5, 2, 3, 4, 6, 8, 10, 15, 20, 30) if vmin <= tick <= vmax]
    colorbar = plt.colorbar(image, cax=ax.inset_axes([1.02, 0.16, 0.035, 0.68]), ticks=ticks, format='%g')
    colorbar.ax.minorticks_off()
    colorbar.ax.tick_params(labelsize=7, length=2)
    colorbar.set_label(label, fontsize=7)
