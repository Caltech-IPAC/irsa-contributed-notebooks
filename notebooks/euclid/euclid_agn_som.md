---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: python3
  language: python
  name: python3
authors:
  - name: Shoubaneh Hemmati
---

# Comparing AGN selections on a Euclid DR1 color SOM

## Overview

- Build a galaxy color map with a self-organizing map (SOM), from Euclid DR1 photometric redshifts and photometry queried with IRSA's TAP service.
- Place active galactic nucleus (AGN) candidates selected with different methods on that map, and see where each method finds its candidates.
- Connect positions on the map to Euclid DR1 spectroscopy, using the Hα line measurements of the SPE catalog and the SIR 1D spectra.

## Introduction

AGN can be selected in many ways: from their optical and near-infrared colors, their mid-infrared colors, their spectra, or the shape of their spectral energy distribution (SED).
Each method finds a different part of the AGN population.
This notebook compares several AGN selections in a common space, the colors of galaxies.

We use a self-organizing map (SOM), a method that arranges sources on a two-dimensional grid of cells so that sources with similar colors end up in the same or in neighboring cells.
We train the SOM on the colors of ordinary galaxies, and then find, for each AGN candidate, the cell whose colors are closest to its own.
This cell is called the best-matching unit (BMU) of the candidate.
The approach follows [Sanjaripour et al. (2024)](https://arxiv.org/abs/2410.07354), who compared AGN selections on a SOM in the CANDELS fields; Sanjaripour et al. (in prep.) present the full Euclid analysis.

**Data used.**
- AGN candidates from Euclid Collaboration: Matamoro Zatarain et al. (2026), selected in the Euclid Quick Release 1 (Q1) data with photometric and spectroscopic methods.
- The Euclid DR1 PHZ catalog: photometric redshifts and "unified" fluxes in the `g`, `r`, `i`, `z` bands (from ground-based surveys) and the Euclid `Y`, `J`, `H` bands.
- The Euclid DR1 SPE catalog of emission line measurements, and the SIR 1D spectra.

Euclid observes three Euclid Deep Fields: North (EDF-N), South (EDF-S), and Fornax (EDF-F).
These are three separate regions of the sky.
The ground-based `griz` images come from different telescopes in each field, so the PHZ catalog converts all fluxes to one common filter system (the unified fluxes), and records which set of bands a source was observed in (its `photometric_system`).
To keep this notebook short we use EDF-N, the field with the most AGN selection methods.
Section 1 shows how to use the other two fields.

**Data access.** The AGN candidate list is a small file (less than 1 MB) read over HTTPS.
The DR1 catalogs are queried with IRSA's TAP service, and the spectra are read one at a time from IRSA's cloud-hosted spectrum service.
Nothing needs to be downloaded in advance.

**Workflow.** The notebook has these sections:
1. Load the AGN candidates.
2. Get DR1 redshifts and photometry for the AGN candidates.
3. Get a reference sample of galaxies.
4. Compute colors and select the samples.
5. Check the input colors.
6. Train the SOM.
7. Color the SOM by redshift and $H$ magnitude.
8. Project the AGN candidates onto the SOM.
9. Compare selection methods.
10. Look at Hα emission across the SOM.

**Runtime.** A few minutes. Most of the time is spent in the query of the emission line catalog in Section 10.

## Imports

```{code-cell} ipython3
# Uncomment the next line to install dependencies if needed.
# %pip install numpy==2.5.4 scipy==1.18.1 matplotlib==3.10.5 astropy==8.0.1 pyvo==1.9.1 minisom==2.3.6
```

```{code-cell} ipython3
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pyvo
from astropy import units as u
from astropy.table import QTable, Table, join, vstack
from astropy.units import UnitsWarning
from minisom import MiniSom

from agn_tutorial_utils import (abmag, add_panel_colorbar, any_flag_set, as_float_array, count_map,
                                fill_empty_cells, find_bmu, format_som_axis, jitter_bmu, median_map,
                                overlay_density, overlay_method_excess)

# The spectra in Section 10 are VOTables that write their units as 'Angstrom' and 'erg'.
# astropy reads these units correctly but warns that the VOUnit standard prefers other spellings.
warnings.filterwarnings('ignore', category=UnitsWarning)

%matplotlib inline
```

## Science choices

We work in the redshift range 1.3 < z < 1.8.
In this range the Hα line (6565 Å) falls in the Euclid red grism (1.2 to 1.9 μm), so the galaxies and AGN can have Hα measurements in the DR1 spectroscopic catalog.

```{code-cell} ipython3
Z_MIN, Z_MAX = 1.3, 1.8

# Bands used for the colors, as named in the unified flux columns of the PHZ catalog (flux_<band>_unif).
BANDS = ['g_ext_decam', 'r_ext_decam', 'i_ext_decam', 'z_ext_decam', 'y', 'j', 'h']
# The six colors the SOM is trained on, each from two neighboring bands.
COLORS = {'g-r': ('g_ext_decam', 'r_ext_decam'), 'r-i': ('r_ext_decam', 'i_ext_decam'),
          'i-z': ('i_ext_decam', 'z_ext_decam'), 'z-Y': ('z_ext_decam', 'y'),
          'Y-J': ('y', 'j'), 'J-H': ('j', 'h')}

# Sources fainter than this AB magnitude in any of the seven bands are not used.
MAG_FAINT = 25.5

# PHZ photometric system of the sources in EDF-N that were observed in all bands.
PHOT_SYSTEM = 30

SOM_SHAPE = (30, 30)     # number of rows and columns of the SOM grid
SOM_ITERATIONS = 100_000
N_REFERENCE = 30_000     # number of reference galaxies to ask for
RANDOM_SEED = 0
```

## 1. Load the AGN candidates

The AGN candidates are from Euclid Collaboration: Matamoro Zatarain et al. (2026), *Euclid Quick Data Release (Q1): The active galaxies of Euclid*.
Their catalogs list every Euclid Q1 source in the three Euclid Deep Fields, with a flag for each AGN selection method.
The full catalogs are large (0.8 to 2.8 GB per field), so for this notebook we made a small file per field that keeps only the AGN candidates and the columns we need.
The catalog authors' [README](https://irsa.ipac.caltech.edu/data/demos/notebooks/euclid-agn/README.txt) describes all columns.
For questions about the AGN catalogs, please contact their author, Teresa Matamoro Zatarain.

The small files were made in four steps:

- Keep the sources that pass the cleaning recommended in the README: `good_flags = 1`, a VIS magnitude of 18 < $I_E$ ≤ 24.5, and not a star candidate (`star_candidate_all = 0`).
- Keep the sources flagged by at least one AGN selection method.
- Add the DR1 object ID of each candidate (`object_id_dr1`). All of the Q1 data were reprocessed for DR1 and the sources got new object IDs, so we matched the Q1 positions to the DR1 sources brighter than $I_E$ = 24.9 within 0.3 arcsec, and kept the candidates with a match. About 90% of the candidates of most methods have one. The exception is the random forest (PRF) selection: only about 30% of its candidates have a match, because many of them are faint in Q1 and are fainter or not detected at the same position in DR1.
- Keep the candidates with a redshift between 1.2 and 1.9, a little wider than the range used in this notebook. The redshift is the DESI spectroscopic redshift when the AGN catalog has one, and the DR1 photometric redshift otherwise.

```{code-cell} ipython3
AGN_URL = 'https://irsa.ipac.caltech.edu/data/demos/notebooks/euclid-agn/edfn_agn_candidates.fits'
# The other two fields are in edfs_agn_candidates.fits and edff_agn_candidates.fits.
# They have fewer selection methods (no DESI spectra and no SED fits), and their sources
# have other photometric systems (20 in EDF-S, 20 and 50 in EDF-F): to use them, change PHOT_SYSTEM too.

agn = Table.read(AGN_URL)
print(f'{len(agn)} AGN candidates')
```

Each selection method has one flag column, 1 for the sources it selects.
We group the flags into seven methods:

```{code-cell} ipython3
SELECTION_METHODS = {
    'Euclid colors (JH-IeY, IeH-gz)': ['JH_IeY_qso_candidate', 'IeH_gz_qso_candidate'],
    'WISE colors (C75, R90)': ['C75_agn_candidate', 'R90_agn_candidate'],
    'Euclid+ground colors (B24)': ['B24a_qso_candidate', 'B24b_qso_candidate'],
    'SED fit': ['AGN_sed_candidate'],
    'Random forest (PRF)': ['PRF_qso_candidate'],
    'Gaia DR3': ['GDR3_qso_candidate'],
    'DESI spectra': ['DESI_qso_candidate', 'DESI_broadline_galaxy_candidate', 'DESI_broadline_qso_candidate',
                     'DESI_niibpt_agn_candidate', 'DESI_siibpt_agn_candidate', 'DESI_oibpt_agn_candidate',
                     'DESI_whan_agn_candidate', 'DESI_blue_agn_candidate', 'DESI_kex_agn_candidate'],
}

# A candidate can be selected by more than one method, so the numbers add up to more than the total.
for method, flags in SELECTION_METHODS.items():
    print(f'{method:32s} {any_flag_set(agn, flags).sum():6d}')
```

## 2. Get DR1 redshifts and photometry for the AGN candidates

The DR1 PHZ catalog has the photometric redshift (`phz_median`) and the unified fluxes of each source.
In this notebook we work in a Euclid Deep Field, so we use the PHZ table of the Euclid Deep Survey.

```{code-cell} ipython3
tap = pyvo.dal.TAPService('https://irsa.ipac.caltech.edu/TAP')

PHZ_TABLE = 'euclid_dr1_phz_photo_z_deep_survey'
PHZ_COLUMNS = (['object_id', 'phz_median', 'phz_flags', 'photometric_system', 'flux_vis_unif']
               + [f'flux_{band}_unif' for band in BANDS])
```

<!-- TODO: remove this cell when DR1 is public on the production IRSA services. -->

```{code-cell} ipython3
tap = pyvo.dal.TAPService('https://irsadev.ipac.caltech.edu/TAP')
```

To get the PHZ rows of our candidates, we select on their DR1 object IDs.
The TAP service accepts at most 1000 values in a list, so we send one query per 1000 object IDs.

```{code-cell} ipython3
def query_by_object_id(table_name, columns, object_ids, condition=''):
    """Get the rows of an IRSA catalog table for a list of object IDs.

    Parameters
    ----------
    table_name : str
        Name of the catalog table. It must have an ``object_id`` column.
    columns : list of str
        Columns to return.
    object_ids : array-like
        Object IDs to look up.
    condition : str, optional
        Extra ADQL condition on the rows, for example ``"spe_rank = 0"``.

    Returns
    -------
    astropy.table.Table
        The requested columns of the rows with one of the object IDs.
    """
    results = []
    # One query per 1000 object IDs, the longest list the service accepts.
    for start in range(0, len(object_ids), 1000):
        id_list = ', '.join(str(object_id) for object_id in object_ids[start:start + 1000])
        query = f"SELECT {', '.join(columns)} FROM {table_name} WHERE object_id IN ({id_list})"
        if condition:
            query += f' AND {condition}'
        results.append(tap.run_async(query).to_table())
    return vstack(results)
```

We run it for the AGN candidates and add the PHZ columns to their table.

```{code-cell} ipython3
agn_phz = query_by_object_id(PHZ_TABLE, PHZ_COLUMNS, agn['object_id_dr1'])
agn_phz.rename_column('object_id', 'object_id_dr1')
agn = join(agn, agn_phz, keys='object_id_dr1')
print(f'{len(agn)} AGN candidates found in the DR1 PHZ catalog')
```

Some of the candidates have a spectroscopic redshift from DESI in the AGN catalog (`Z_desi`).
We use it when there is one, and the DR1 photometric redshift otherwise.

```{code-cell} ipython3
z_spec = as_float_array(agn['Z_desi'])
has_z_spec = np.isfinite(z_spec) & (z_spec > 0)
agn['z'] = np.where(has_z_spec, z_spec, as_float_array(agn['phz_median']))
agn['z_source'] = np.where(has_z_spec, 'spec', 'phot')

agn = agn[(agn['z'] > Z_MIN) & (agn['z'] < Z_MAX)]
print(f"{len(agn)} AGN candidates at {Z_MIN} < z < {Z_MAX} ({np.sum(agn['z_source'] == 'spec')} with a spectroscopic redshift)")
```

## 3. Get a reference sample of galaxies

The SOM is trained on ordinary galaxies in the same redshift range, so that the map describes the colors of the galaxy population.
We select the reference galaxies from the same PHZ table with these conditions:

- `phz_median` in our redshift range;
- `phz_flags = 0`, so the PHZ processing had no problems;
- `phz_classification = 2`, so the source is classified as a galaxy and not as a star;
- `photometric_system = 30`, so the source is in EDF-N and was observed in all bands;
- the same VIS magnitude range as the AGN candidates, 18 < $I_E$ ≤ 24.5. Without it, the reference sample would have faint galaxies with colors that no AGN candidate can have, because of how the candidates were selected.

The DR1 fluxes are in μJy, so an AB magnitude $m$ corresponds to a flux of $10^{(23.9 - m)/2.5}$ μJy.

```{code-cell} ipython3
flux_vis_min = 10 ** ((23.9 - 24.5) / 2.5)  # 0.575 uJy, I_E = 24.5
flux_vis_max = 10 ** ((23.9 - 18.0) / 2.5)  # 229 uJy, I_E = 18

reference_query = f"""
    SELECT TOP {N_REFERENCE} {', '.join(PHZ_COLUMNS)}
    FROM {PHZ_TABLE}
    WHERE phz_median BETWEEN {Z_MIN} AND {Z_MAX}
      AND phz_flags = 0
      AND phz_classification = 2
      AND photometric_system = {PHOT_SYSTEM}
      AND flux_vis_unif BETWEEN {flux_vis_min} AND {flux_vis_max}
    """
reference = tap.run_async(reference_query).to_table()
print(f'{len(reference)} reference galaxies')
```

`TOP` returns the first rows that the service finds.
The rows of the PHZ table are not ordered by position, so these galaxies are spread over the whole field.
The service can return different galaxies each time the query runs, so the numbers and figures below change a little from run to run.

Some of the reference galaxies are also AGN candidates.
We keep them, because the SOM should describe all galaxies, but we mark them so that we can leave them out when we compare AGN candidates with other galaxies in Section 10.

```{code-cell} ipython3
reference['is_agn_candidate'] = np.isin(reference['object_id'], agn['object_id_dr1'])
print(f"{reference['is_agn_candidate'].sum()} reference galaxies are AGN candidates")
```

## 4. Compute colors and select the samples

We compute the same quantities for the reference galaxies and for the AGN candidates: the AB magnitude in each band, the six colors, and a flag for good photometry.
The unified fluxes of DR1 have no usable uncertainties, so instead of a signal-to-noise cut we ask for a magnitude brighter than `MAG_FAINT` in all seven bands.

```{code-cell} ipython3
def add_colors(sources):
    """Add colors, the H magnitude and a photometry flag to a table of PHZ sources.

    Parameters
    ----------
    sources : astropy.table.Table
        Table with the unified flux columns of the PHZ catalog (``flux_<band>_unif``) for the bands in ``BANDS``.
        These columns are added to it:
        one column per color in ``COLORS``; ``mag_h``, the H AB magnitude;
        and ``good_photometry``, True for the sources brighter than ``MAG_FAINT`` in all bands.
    """
    mags = {band: abmag(sources[f'flux_{band}_unif']) for band in BANDS}
    for color, (band1, band2) in COLORS.items():
        sources[color] = mags[band1] - mags[band2]
    sources['mag_h'] = mags['h']
    # abmag returns NaN for a missing or negative flux, and a comparison with NaN is False.
    sources['good_photometry'] = np.all([mags[band] < MAG_FAINT for band in BANDS], axis=0)
```

We add these columns to both tables and keep the sources with good photometry.

```{code-cell} ipython3
add_colors(reference)
add_colors(agn)

reference = reference[reference['good_photometry']]
# For the AGN candidates we also ask for the same photometric system as the reference galaxies.
agn = agn[agn['good_photometry'] & (agn['photometric_system'] == PHOT_SYSTEM)]
print(f'{len(reference)} reference galaxies and {len(agn)} AGN candidates with good photometry')
```

## 5. Check the input colors

Before training the SOM, we compare the six colors of the reference galaxies and the AGN candidates.

```{code-cell} ipython3
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, color in zip(axes.flat, COLORS):
    # Same bins for both samples, covering 99% of each.
    low = min(np.percentile(reference[color], 0.5), np.percentile(agn[color], 0.5))
    high = max(np.percentile(reference[color], 99.5), np.percentile(agn[color], 99.5))
    bins = np.linspace(low, high, 60)
    ax.hist(reference[color], bins=bins, density=True, color='0.75', label=f'reference galaxies ({len(reference)})')
    ax.hist(agn[color], bins=bins, density=True, histtype='step', color='black', linewidth=1.5,
            label=f'AGN candidates ({len(agn)})')
    ax.set_xlabel(f'${color}$ (mag)')
    ax.set_ylabel('normalized number')
axes[0, 0].legend(frameon=False, fontsize=8)
fig.tight_layout()
plt.show()
```

**Figure 1.** The six colors used to train the SOM, for the reference galaxies (gray) and the AGN candidates (black line).
Each histogram is normalized to an area of 1.
The AGN candidates form two groups: one has the same colors as the reference galaxies, and the other is redder than most galaxies, most clearly in $r-i$, $Y-J$ and $J-H$.
The SOM is trained on the galaxies, so the candidates of the red group will fall where the map has its reddest cells.

## 6. Train the SOM

Each SOM cell has a weight vector, a set of six colors.
Training moves the weights toward the colors of the reference galaxies, so that after training the cells cover the range of galaxy colors, with similar colors in neighboring cells.

Before training we normalize each color: we subtract the mean and divide by the standard deviation of the reference galaxies, so that all colors have the same importance.

```{code-cell} ipython3
reference_colors = np.column_stack([reference[color] for color in COLORS])
color_mean = reference_colors.mean(axis=0)
color_std = reference_colors.std(axis=0)
reference_normalized = (reference_colors - color_mean) / color_std

som = MiniSom(SOM_SHAPE[0], SOM_SHAPE[1], len(COLORS), sigma=10.0, learning_rate=0.1, random_seed=RANDOM_SEED)
som.random_weights_init(reference_normalized)
som.train_random(reference_normalized, SOM_ITERATIONS)
```

Now we find the BMU of each reference galaxy, the cell whose weights are closest to its colors.
The distance between the colors of a galaxy and the weights of its BMU tells how well the SOM describes that galaxy.

```{code-cell} ipython3
reference_bmu, reference_distance = find_bmu(som, reference_normalized)
print(f'Median distance to the BMU: {np.median(reference_distance):.2f}')
```

## 7. Color the SOM by redshift and $H$ magnitude

To see what the SOM has learned, we plot the median photometric redshift and the median $H$ magnitude of the reference galaxies in each cell.
The $H$ magnitude is a simple measure of how bright the galaxy is in the near-infrared; at one redshift, brighter galaxies in $H$ are on average more massive.

```{code-cell} ipython3
median_z = median_map(reference['phz_median'], reference_bmu, SOM_SHAPE)
# The H magnitude map is the background of the next figures, so we fill its empty cells.
median_mag_h = fill_empty_cells(median_map(reference['mag_h'], reference_bmu, SOM_SHAPE))
mag_h_limits = np.percentile(median_mag_h, [5, 95])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
image = axes[0].imshow(fill_empty_cells(median_z), origin='lower', cmap='viridis', vmin=Z_MIN, vmax=Z_MAX)
fig.colorbar(image, ax=axes[0], fraction=0.046, label='median photometric redshift')
image = axes[1].imshow(median_mag_h, origin='lower', cmap='cividis_r', vmin=mag_h_limits[0], vmax=mag_h_limits[1])
fig.colorbar(image, ax=axes[1], fraction=0.046, label='median $H$ magnitude (AB)')
for ax in axes:
    format_som_axis(ax, SOM_SHAPE)
plt.show()
```

**Figure 2.** The SOM colored by the median photometric redshift (left) and the median $H$ magnitude (right) of the reference galaxies in each cell.
Each small square is one cell.
The axes have no physical meaning: only which cells are close to each other matters.
Both quantities change smoothly across the map, although the SOM was trained on colors only: the redshift increases from one side of the map to the other, and the galaxies that are brightest in $H$ are together in one corner.
The two patterns are different, so the position on the map tells us about both the redshift and the brightness of a galaxy.
Which side or corner this is depends on the reference galaxies returned by the query: the map can come out rotated or mirrored from one run to the next.

## 8. Project the AGN candidates onto the SOM

We normalize the colors of the AGN candidates in the same way as the reference galaxies, and find the BMU of each candidate.

```{code-cell} ipython3
agn_colors = np.column_stack([agn[color] for color in COLORS])
agn_bmu, agn_distance = find_bmu(som, (agn_colors - color_mean) / color_std)
```

A candidate with colors that no reference galaxy has is far from the weights of every cell, and its BMU has little meaning.
We keep the candidates that are as close to their BMU as 95% of the reference galaxies are.

```{code-cell} ipython3
max_distance = np.percentile(reference_distance, 95)
on_map = agn_distance <= max_distance
print(f'{on_map.sum()} of {len(agn)} AGN candidates have colors covered by the SOM')

agn, agn_bmu = agn[on_map], agn_bmu[on_map]
agn_counts = count_map(agn_bmu, SOM_SHAPE)
```

We plot the candidates that are left on the SOM in two ways: one marker per candidate, and the number of candidates per cell.

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax in axes:
    background = ax.imshow(median_mag_h, origin='lower', cmap='cividis_r', vmin=mag_h_limits[0], vmax=mag_h_limits[1])
    format_som_axis(ax, SOM_SHAPE)
fig.colorbar(background, ax=axes[0], fraction=0.046, location='left', label='median $H$ magnitude (AB)')

# Left: one cross per candidate, at a random position inside its cell.
x, y = jitter_bmu(agn_bmu, seed=RANDOM_SEED)
axes[0].scatter(x, y, marker='x', s=8, linewidths=0.4, color='#064d2c', alpha=0.4)
axes[0].set_title('AGN candidates')

# Right: smoothed number of candidates per cell.
density = overlay_density(axes[1], agn_counts)
fig.colorbar(density, ax=axes[1], fraction=0.046, label='AGN candidates per cell (smoothed)')
axes[1].set_title('Density of AGN candidates')
plt.show()
```

**Figure 3.** The AGN candidates on the SOM, over the median $H$ magnitude of the reference galaxies.
Left: each cross is one candidate, at its BMU.
Right: the smoothed number of candidates per cell (green); cells with few candidates are left clear.
The candidates are not spread over the map like the galaxies.
Many are in the corner of the galaxies that are bright in $H$, and many are in the cells along the edges of the map.
The edge cells hold the most extreme colors of the reference galaxies, so a candidate with colors redder or bluer than most galaxies has its BMU there.
The crowding at the edges is therefore expected for sources that differ from ordinary galaxies, not a problem of the map.

## 9. Compare selection methods

Now we look at each selection method separately.
The candidates of all methods together are not spread evenly over the SOM (Figure 3), so for each method we plot its *excess*: the number of its candidates in a cell, divided by the number expected if the method selected the same mix of colors as all methods together.
An excess of 2 means the method finds twice as many candidates in that cell as expected.

```{code-cell} ipython3
fig, axes = plt.subplots(2, 4, figsize=(17, 7.5))
for ax, (method, flags) in zip(axes.flat, SELECTION_METHODS.items()):
    ax.imshow(median_mag_h, origin='lower', cmap='cividis_r', vmin=mag_h_limits[0], vmax=mag_h_limits[1])
    selected = any_flag_set(agn, flags)
    excess = overlay_method_excess(ax, count_map(agn_bmu[selected], SOM_SHAPE), agn_counts)
    if excess is not None:
        add_panel_colorbar(ax, excess, 'excess')
    # For a method with few candidates, also mark each candidate.
    if selected.sum() <= 40:
        x, y = jitter_bmu(agn_bmu[selected], seed=RANDOM_SEED)
        ax.scatter(x, y, marker='x', s=28, linewidths=1, color='#064d2c')
    ax.set_title(f'{method} ({selected.sum()})', fontsize=10)
    format_som_axis(ax, SOM_SHAPE)
# Hide the panels that are not used.
for ax in axes.flat[len(SELECTION_METHODS):]:
    ax.axis('off')
fig.subplots_adjust(wspace=0.45)
plt.show()
```

**Figure 4.** Excess of each AGN selection method on the SOM, over the median $H$ magnitude of the reference galaxies (same background as Figure 3).
Green marks the cells where the method selects more candidates than expected from the distribution of all candidates.
The number of candidates of each method is given in the panel title; each panel has its own color scale.
The methods select different parts of the map, with little overlap between some of them.

To put numbers on the figure, we print the median $H$ magnitude, two colors, and the median redshift of the candidates of each method.

```{code-cell} ipython3
def print_medians(name, sources):
    """Print the number of sources and the medians of their H magnitude, two colors and redshift.

    Parameters
    ----------
    name : str
        Name of the sample, printed at the start of the line.
    sources : astropy.table.Table
        Sources with the columns ``mag_h``, ``g-r``, ``J-H`` and ``z``.
    """
    medians = '  '.join(f'{np.median(as_float_array(sources[column])):7.2f}' for column in ['mag_h', 'g-r', 'J-H', 'z'])
    print(f'{name:32s} {len(sources):6d}  {medians}')
```

We print one line per method, and one line each for all AGN candidates and for the reference galaxies.

```{code-cell} ipython3
print(f"{'':32s} {'number':>6s}  {'H mag':>7s}  {'g-r':>7s}  {'J-H':>7s}  {'z':>7s}")
for method, flags in SELECTION_METHODS.items():
    print_medians(method, agn[any_flag_set(agn, flags)])
print_medians('All AGN candidates', agn)
reference['z'] = reference['phz_median']
print_medians('Reference galaxies', reference)
```

The table shows what the regions of Figure 4 mean:

- The WISE color selections find candidates that are bright in $H$ and red: they are in the corner of the bright galaxies.
- The Euclid color selections (JH-IeY, IeH-gz, and B24) find faint candidates with blue colors, in the edge cells on the faint side of the map.
- The Gaia selection finds the brightest candidates, with blue colors, as expected for unobscured quasars.
- The random forest (PRF) candidates are red in $J-H$ and of average brightness.
- The SED fit and DESI candidates are between these groups. They need a spectroscopic redshift, so they are brighter than the reference galaxies.

## 10. Look at Hα emission across the SOM

The DR1 SPE catalog has emission line measurements from the Euclid spectra.
We use it to ask whether the AGN candidates and the other galaxies differ in their Hα emission, and whether that depends on their position on the SOM.

### 10.1 Query the Hα measurements

The line table of the Euclid Deep Survey is `euclid_dr1_spectro_line_features_spe_line_features_deep`.
It has one row per line and per redshift solution of each source; we take the Hα line of the best redshift solution (`spe_rank = 0`).
We use the Gaussian-fit measurements (the `_gf` columns).

This is the slowest query of the notebook, so we do not query all sources: we take `N_HALPHA` AGN candidates at random and, as a control sample, the same number of reference galaxies that are not AGN candidates.
Increase `N_HALPHA` for less noisy results.

```{code-cell} ipython3
LINE_TABLE = 'euclid_dr1_spectro_line_features_spe_line_features_deep'
LINE_COLUMNS = ['object_id', 'spe_line_flux_gf', 'spe_line_snr_gf', 'spe_line_lum_gf', 'spe_line_central_wl_gf']
LINE_CONDITION = "spe_line_name = 'Halpha' AND spe_rank = 0"

N_HALPHA = 1000

# Random AGN candidates and random reference galaxies that are not AGN candidates.
rng = np.random.default_rng(RANDOM_SEED)
chosen = rng.choice(len(agn), size=min(N_HALPHA, len(agn)), replace=False)
agn_halpha, agn_halpha_bmu = agn[chosen], agn_bmu[chosen]
not_agn = np.flatnonzero(~reference['is_agn_candidate'])
chosen = rng.choice(not_agn, size=min(N_HALPHA, len(not_agn)), replace=False)
control, control_bmu = reference[chosen], reference_bmu[chosen]

agn_lines = query_by_object_id(LINE_TABLE, LINE_COLUMNS, agn_halpha['object_id_dr1'], LINE_CONDITION)
control_lines = query_by_object_id(LINE_TABLE, LINE_COLUMNS, control['object_id'], LINE_CONDITION)
```

The catalog has a row for a line even when the line is not detected.
We count Hα as detected when its signal-to-noise ratio is at least 3.5 and its measured wavelength is where Hα falls in our redshift range.
The second condition removes the sources for which the spectroscopic redshift solution is far from the redshift we used to select them.

```{code-cell} ipython3
HALPHA = 6564.6  # vacuum wavelength in Angstrom


def halpha_luminosity(sources, lines, id_column):
    """Hα luminosity of each source, NaN for the sources with no Hα detection.

    Parameters
    ----------
    sources : astropy.table.Table
        Sources to get the luminosity for.
    lines : astropy.table.Table
        Hα rows of the SPE line table, with the columns in ``LINE_COLUMNS``.
    id_column : str
        Name of the DR1 object ID column in ``sources``.

    Returns
    -------
    numpy.ndarray
        Hα luminosity in erg/s for each row of ``sources``.
    """
    snr = as_float_array(lines['spe_line_snr_gf'])
    wavelength = as_float_array(lines['spe_line_central_wl_gf'])
    luminosity = as_float_array(lines['spe_line_lum_gf'])
    detected = ((snr >= 3.5) & (luminosity > 0)
                & (wavelength > HALPHA * (1 + Z_MIN)) & (wavelength < HALPHA * (1 + Z_MAX)))
    luminosity_by_id = dict(zip(lines['object_id'][detected], luminosity[detected]))
    return np.array([luminosity_by_id.get(object_id, np.nan) for object_id in sources[id_column]])
```

We compute the luminosities of both samples and count the detections.

```{code-cell} ipython3
agn_halpha['halpha_luminosity'] = halpha_luminosity(agn_halpha, agn_lines, 'object_id_dr1')
control['halpha_luminosity'] = halpha_luminosity(control, control_lines, 'object_id')
for name, sample in [('AGN candidates', agn_halpha), ('control galaxies', control)]:
    detected = np.isfinite(sample['halpha_luminosity'])
    print(f'{name}: {detected.sum()} of {len(sample)} have detected Hα ({detected.mean():.0%})')
```

### 10.2 Hα detections on the SOM

With 1000 sources there is about one per SOM cell, too few to measure a fraction.
So we group the cells in blocks of 6 by 6 cells, and compute in each block the fraction of sources with detected Hα.

```{code-cell} ipython3
BLOCK = 6
block_shape = (SOM_SHAPE[0] // BLOCK, SOM_SHAPE[1] // BLOCK)


def detection_fraction(bmu, detected, min_sources=10):
    """Fraction of sources with a detection in each block of SOM cells.

    Parameters
    ----------
    bmu : numpy.ndarray
        Integer array of shape (n_sources, 2) with the BMU of each source.
    detected : numpy.ndarray
        Boolean array, True for the sources with a detection.
    min_sources : int, optional
        Blocks with fewer sources are set to NaN.

    Returns
    -------
    numpy.ndarray
        Array of shape ``block_shape`` with the detected fraction in each block.
    """
    n_all = count_map(bmu // BLOCK, block_shape)
    n_detected = count_map(bmu[detected] // BLOCK, block_shape)
    return np.where(n_all >= min_sources, n_detected / np.maximum(n_all, 1), np.nan)
```

We plot this fraction for the control galaxies and for the AGN candidates, and next to it the Hα luminosities of the detected sources.

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw={'width_ratios': [1, 1, 1.1]}, layout='constrained')
samples = [('Control galaxies', control, control_bmu), ('AGN candidates', agn_halpha, agn_halpha_bmu)]
for ax, (name, sample, bmu) in zip(axes, samples):
    fraction = detection_fraction(bmu, np.isfinite(sample['halpha_luminosity']))
    # extent puts the blocks on the same axes as the SOM cells of the other figures.
    image = ax.imshow(fraction, origin='lower', cmap='magma', vmin=0, vmax=1,
                      extent=(-0.5, SOM_SHAPE[1] - 0.5, -0.5, SOM_SHAPE[0] - 0.5))
    ax.set_facecolor('0.85')
    ax.set_title(name)
    format_som_axis(ax, SOM_SHAPE)
fig.colorbar(image, ax=list(axes[:2]), fraction=0.046, pad=0.02, label='fraction with detected Hα')

bins = np.linspace(41.3, 43.5, 30)
for name, sample, _ in samples:
    axes[2].hist(np.log10(sample['halpha_luminosity'][np.isfinite(sample['halpha_luminosity'])]), bins=bins,
                 density=True, histtype='step', linewidth=1.5, label=name)
axes[2].set_xlabel(r'log$_{10}$ Hα luminosity (erg s$^{-1}$)')
axes[2].set_ylabel('normalized number')
axes[2].legend(frameon=False)
plt.show()
```

**Figure 5.** Hα emission of the control galaxies and the AGN candidates.
Left and middle: fraction of sources with detected Hα in each block of 6 by 6 SOM cells, on the same SOM as the other figures; gray blocks have fewer than 10 sources.
Right: Hα luminosities of the sources with a detection.
Hα is detected in about 40% of the AGN candidates and about 25% of the control galaxies (the exact numbers are printed above).
The fraction is higher for the AGN candidates in most blocks, so the difference is not only due to the two samples being in different parts of the map.
The detected lines of the AGN candidates are also more luminous, by a factor of about 2.
Part of this difference can come from the AGN candidates being brighter than the control galaxies (see the table in Section 9).

### 10.3 Look at some spectra

Finally we plot the spectra of the AGN candidates with the brightest Hα lines.
The table `euclid.objectid_spectrafile_association_dr1` lists the spectrum files of each object ID, and IRSA's spectrumDM service returns one spectrum of a file as a table (see the Euclid DR1 SIR spectra tutorial for details).

```{code-cell} ipython3
N_SPECTRA = 4
# The ten AGN candidates with the brightest Hα lines: not all of them have a spectrum we can read.
brightest = agn_halpha[np.argsort(np.nan_to_num(agn_halpha['halpha_luminosity']))[::-1][:10]]

# The spectrumDM service cannot yet read HDU numbers above 9999, so we leave those spectra out.
spectra_query = f"""
    SELECT objectid, 'https://irsacloud.ipac.caltech.edu/api/spectrumdm/convert/' || path
           || '?dataset_id=euclid_combspec_dr1&hdu=' || hdu AS url
    FROM euclid.objectid_spectrafile_association_dr1
    WHERE objectid IN ({', '.join(str(object_id) for object_id in brightest['object_id_dr1'])})
      AND hdu < 10000
    """
spectra_urls = tap.search(spectra_query).to_table()
```

An object can have several spectra, from the blue and the red grism.
For each object we read its spectra until we find one that covers the wavelength of Hα, and we stop when we have four objects.

```{code-cell} ipython3
def read_halpha_spectrum(urls, halpha_observed):
    """Read the first spectrum of a list that covers the observed wavelength of Hα.

    Parameters
    ----------
    urls : list of str
        spectrumDM URLs of the spectra of one object.
    halpha_observed : astropy.units.Quantity
        Observed wavelength of Hα for this object.

    Returns
    -------
    astropy.table.QTable or None
        The spectrum, or None if no spectrum covers the wavelength.
    """
    for url in urls:
        spectrum = QTable.read(url)
        if spectrum['WAVELENGTH'].min() < halpha_observed < spectrum['WAVELENGTH'].max():
            return spectrum
    return None
```

We plot one spectrum per object, and leave out the flux bins that are flagged as bad.

```{code-cell} ipython3
fig, axes = plt.subplots(1, N_SPECTRA, figsize=(4.2 * N_SPECTRA, 3.6))
n_plotted = 0
for source in brightest:
    halpha_observed = HALPHA * (1 + source['z']) * u.angstrom
    spectrum = read_halpha_spectrum(spectra_urls['url'][spectra_urls['objectid'] == source['object_id_dr1']],
                                    halpha_observed)
    if spectrum is None:
        continue
    ax = axes[n_plotted]
    # Flux bins with an odd MASK value or a MASK value of 64 or more should not be used.
    mask = spectrum['MASK'].value
    bad = (mask % 2 == 1) | (mask >= 64)
    ax.plot(spectrum['WAVELENGTH'].to(u.micron), np.ma.masked_where(bad, spectrum['SIGNAL'].value), linewidth=0.8)
    ax.axvline(halpha_observed.to_value(u.micron), color='crimson', linestyle='--', linewidth=0.8)
    ax.set_title(f"{source['object_id_dr1']}\nz = {source['z']:.2f} ({source['z_source']})", fontsize=9)
    ax.set_xlabel('wavelength (μm)')
    n_plotted += 1
    if n_plotted == N_SPECTRA:
        break
axes[0].set_ylabel(r'flux ($10^{-16}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)')
fig.tight_layout()
plt.show()
```

**Figure 6.** Euclid red grism spectra of the AGN candidates with the brightest Hα lines.
The dashed line marks the wavelength of Hα at the redshift used in this notebook.
For the candidates with a spectroscopic redshift from DESI the line is at the expected wavelength.
For a candidate with only a photometric redshift, the line can be a little off the dashed line, which shows the accuracy of the photometric redshift.

## Summary

- A few thousand AGN candidates and 30,000 reference galaxies are enough to compare AGN selections on a SOM, and the DR1 PHZ catalog gives their redshifts and photometry in under a minute.
- The unified fluxes of the PHZ catalog are on one filter system, so the same code works in all three Euclid Deep Fields.
- The AGN candidates are not spread over the galaxy color map like the galaxies: they are in the cells of the galaxies that are bright in $H$, and in the cells with the most extreme colors.
- Each selection method finds candidates in its own part of the map. The WISE selections find bright red sources and the Euclid color selections faint blue ones, so no single method finds all AGN candidates.
- The AGN candidates have detected Hα more often than control galaxies, and their lines are more luminous.

The $H$ magnitude is only a rough measure of the brightness of the host galaxy, not a stellar mass.
The photometric redshifts of the PHZ catalog are made for galaxies and are less reliable for sources dominated by the AGN.
To go further, you can use the other two fields, a larger Hα sample (`N_HALPHA`), or a different redshift range.

## References

**AGN candidate catalogs**

- Euclid Collaboration: Matamoro Zatarain, T., Fotopoulou, S., Ricci, F., et al. (2026). *Euclid Quick Data Release (Q1): The active galaxies of Euclid*. A&A, 711, A20. Please cite this paper when using the AGN candidate catalogs.

**Method**

- Sanjaripour, S., et al. (2024). [arXiv:2410.07354](https://arxiv.org/abs/2410.07354).
- Kohonen, T. (2001). *Self-Organizing Maps*. Springer Series in Information Sciences, 3rd ed.

**AGN selection methods** (see the README of the AGN catalogs for the full list)

- Assef, R. J., Stern, D., Noirot, G., et al. (2018). *The WISE AGN Catalog* (C75 and R90 selections). ApJS, 234, 23.
- Euclid Collaboration: Bisigello, L., et al. (2024). *Selecting AGN using observed colours* (B24a and B24b selections). A&A, 691, A1.
- Storey-Fisher, K., Hogg, D. W., Rix, H.-W., et al. (2024). *Quaia: the Gaia-unWISE Quasar Catalog*. ApJ, 964, 69.
- DESI Collaboration (2024). *DESI Early Data Release*. AJ, 168, 58.

## About This Notebook

**Last updated:** 10 November 2026

This notebook was written and contributed by its authors.
When last checked, it ran without error in an environment set up with the install instructions above.
However, it is not regularly tested or maintained.

To submit a bug report or ask a question, please open an [issue](https://github.com/Caltech-IPAC/irsa-contributed-notebooks/issues).

### Acknowledgements

The AGN candidate catalogs were made by Teresa Matamoro Zatarain (University of Bristol) and collaborators.

This tutorial uses Euclid DR1 data products distributed by the NASA/IPAC Infrared Science Archive (IRSA), which is operated by the California Institute of Technology under contract with the National Aeronautics and Space Administration.
Euclid is a European Space Agency mission with contributions from NASA.

This notebook uses the Python packages NumPy, SciPy, Matplotlib, Astropy, PyVO and MiniSom.

The notebook was written by the author.
The AI assistant Claude (Anthropic) was used for editing, and the author reviewed all of its edits.
