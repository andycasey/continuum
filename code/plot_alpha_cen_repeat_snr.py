import os
import numpy as np
import pickle
from tqdm import tqdm
from astropy.table import Table

from spectrum import Spectrum, SpectrumCollection

summary = Table.read("alfCenA_stability.csv")

#t = Table(rows=results, names=("path", "snr", "status", "cost", "optimality", "std_norm_flux", "p_90", "p_95", "p_97.5"))


snr_bins = 1 + np.arange(200) # 1 to 200
rectified_flux_min, rectified_flux_max, rectified_flux_step = (0.8, 1.20, 0.001)
rectified_flux_bins = np.linspace(rectified_flux_min, rectified_flux_max, int((rectified_flux_max - rectified_flux_min) / rectified_flux_step) + 1)

percentiles = [95]

H = np.nan * np.ones((snr_bins.size, rectified_flux_bins.size - 1))
percentile_values = np.nan * np.ones((snr_bins.size, len(percentiles)))

fig_all, ax_all = plt.subplots(figsize=(10.5, 4))

fig, ax = plt.subplots()

#
keep_snrs = (104, 105, 106, 107, 108)
keep_snrs = (119, 120, 121, 122)
keep = {}

import matplotlib as mpl
all_colors = mpl.colormaps["plasma"](np.linspace(0, 1, len(snr_bins)))

comparison = []

colors = ("tab:red", "tab:blue", "tab:green", "tab:orange")
summary.sort("snr")
for row in tqdm(summary):
    if int(row["snr"]) > 200 or row["path"].endswith("_e2ds_B.fits"): continue

    fits_path = "../applications/harps/alfCenA/" + row["path"]
    result_path = fits_path + ".pkl"
    if not os.path.exists(result_path):
        continue

    spectra = SpectrumCollection.read(fits_path)
    with open(result_path, "rb") as fp:
        (result, continuum, rectified_model_flux, rectified_telluric_flux, y_pred) = pickle.load(fp)

    # When we initially ran this I excluded the last echelle order
    rectified_flux = (spectra.flux[:-1] / continuum).flatten()

    mask = ((spectra.λ[:-1] > 4500)).flatten() * (rectified_flux > 0.10)
    rectified_flux = rectified_flux[mask]


    # check if it is cen B
    co = 56
    norm_order = spectra.flux[co]/continuum[co]
    index = spectra.λ_rest_vacuum[co].searchsorted(5890.855)
    likely_cen_b = (norm_order[index] > 0.639)

    comparison.append(list(result.x[:32]) + [spectra.meta["v_rel"], likely_cen_b])
    if likely_cen_b:
        print(f"Skipping {row['snr']:.0f} {row['path']} because Cen B")
        continue

    snr_index = int(row["snr"]) - 1
    h, _ = np.histogram(rectified_flux, bins=rectified_flux_bins, density=True)
    H[snr_index] = h
    percentile_values[snr_index] = np.nanpercentile(rectified_flux, percentiles)

    oi = 50
    if snr_index > 10:
        color = all_colors[snr_index]
        for oi in range(50, 71):
            ax_all.plot(spectra.λ_rest_vacuum[oi], spectra.flux[oi] / continuum[oi], c=color, lw=0.5, alpha=0.5, ms=0)
    

    if int(row["snr"]) in keep_snrs:
        color = colors[snr_index - keep_snrs[0]]

        keep[int(row["snr"])] = [rectified_flux, result.x[:32]]

        print(f"{int(row['snr'])} {row['path']}, {spectra.meta['DATE']} {spectra.meta['OBJECT']} {spectra.meta['RA']} {spectra.meta['DEC']}")

        for i in range(71):
            label = f"snr={int(row['snr'])} {spectra.meta['DATE']}" if i == 0 else None
            ax.plot(spectra.λ_rest_vacuum[i], spectra.flux[i] / continuum[i], c=color, lw=0.5, alpha=0.5, label=label, ms=0)

    if spectra.meta["DATE"].startswith("2013-05-29T"):
        print(row["snr"])


'''
bw = np.array([ea[:32] for ea in comparison])
v_rels = np.array([ea[-2] for ea in comparison])
likely_cen_b = np.array([ea[-1] for ea in comparison])

fig, axes = plt.subplots(32, 32)
for i, ax_row in enumerate(axes):
    for j, ax in enumerate(ax_row):
        if j >= i:
            ax.set_visible(False)
            continue
        ax.scatter(bw.T[i][likely_cen_b], bw.T[j][likely_cen_b], c="tab:blue", s=5, alpha=0.5)
        ax.scatter(bw.T[i][~likely_cen_b], bw.T[j][~likely_cen_b], c="tab:red", s=5, alpha=0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            ax.set_ylabel(i, fontsize=6)
        if i == 31:
            ax.set_xlabel(j, fontsize=6)
        #ax.set_title(f"{i} {j}")

'''

fig, ax = plt.subplots()
for k, (v, bw) in keep.items():
    ax.plot(bw, label=k, ms=0, alpha=0.5)


raise a


scat = ax_all.scatter(np.nan * np.ones_like(snr_bins), np.ones_like(snr_bins), c=np.arange(len(snr_bins)), cmap="plasma", vmin=10, vmax=len(snr_bins))
cbar = plt.colorbar(scat)

ax_all.set_xlabel("Wavelength [$\AA$]")
ax_all.set_ylabel("Normalised flux")
ax_all.set_ylim(0, 1.2)
#ax_all.set_xlim((5041.669645611034, 5044.515739915486))
cbar.set_label("S/N")
fig_all.tight_layout()


ax.legend()

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

si = 30
width_inches = 433.62 / 72
fig, ax = plt.subplots(figsize=(width_inches, 3))
image = ax.imshow(
    H[si:None].T,
    interpolation="none",
    origin="lower",
    extent=(snr_bins[si], snr_bins[-1], rectified_flux_bins[0], rectified_flux_bins[-1]),
    cmap="Greys",
    aspect="auto",
    vmin=0,
    vmax=np.nanpercentile(H, 99.9),
)
#for i, percentile in enumerate(percentiles):
#    ax.plot(snr_bins, percentile_values[:, i], ms=0, label=f"{percentile}th percentile", zorder=10)
ax.legend(frameon=False)
ax.set_xlim(30, 200)
cbar = plt.colorbar(image)
cbar.set_label("Density")
cbar.ax.tick_params(width=0)
ax.set_ylim(rectified_flux_min, rectified_flux_max)
ax.set_xlabel("Spectrum signal-to-noise ratio [pixel$^{-1}$]")
ax.set_ylabel("Normalised flux")
fig.tight_layout()
fig.savefig("../paper/alpha-cen-repeat-snr.pdf", dpi=300)