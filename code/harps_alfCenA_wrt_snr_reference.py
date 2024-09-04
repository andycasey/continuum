from glob import glob
from astropy.table import Table
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import pickle

from spectrum import Spectrum
from model import Clam, apply_radial_velocity_shift, get_closest_order
from continuum_basis import Polynomial, Sinusoids

import matplotlib
#from mpl_utils import mpl_style
#matplotlib.style.use(mpl_style)

def load_harps_e2ds(path, rest_frame=True, mask_na_doublet=True, ignore_last_order=True):

    pattern = path.split("_e2ds_")[0] + "*_bis*.fits"
    bis_path = glob(pattern)[0]
    with fits.open(bis_path) as image:
        bis_rv = image[0].header["ESO DRS BIS RV"]
        berv = image[0].header["ESO DRS BERV"]

    v_shift = -bis_rv + berv
    print(v_shift)

    spectra = []
    with fits.open(path) as image:
        i, coeff = (0, [])
        header_coeff_key = "HIERARCH ESO DRS CAL TH COEFF LL{i}"
        while image[0].header.get(header_coeff_key.format(i=i), False):
            coeff.append(image[0].header[header_coeff_key.format(i=i)])
            i += 1

        n_orders, n_pixels = image[0].data.shape
        x = np.arange(n_pixels)# - n_pixels // 2

        coeff = np.array(coeff).reshape((n_orders, -1))
        λ = np.array([np.polyval(c[::-1], x) for c in coeff])

        n_pixels = 0
        K = n_orders - 1 if ignore_last_order else n_orders
        for i in range(K): # ignore last order until we have a good telluric model
            flux = image[0].data[i]
            ivar = 1/image[0].data[i]
            bad_pixel = (flux == 0) | ~np.isfinite(flux) | (flux < 0)
            flux[bad_pixel] = 0
            ivar[bad_pixel] = 0

            if rest_frame:
                λ_order = apply_radial_velocity_shift(λ[i], v_shift)
            else:
                λ_order = λ[i]            

            if mask_na_doublet:
                na_double = (5898 >= λ[i]) * (λ[i] >= 5886)
                ivar[na_double] = 0

            spectra.append(Spectrum(λ_order, flux, ivar, vacuum=False))
            n_pixels += flux.size

    return spectra


with open("../applications/harps-sandbox/20240816_train_harps_model.pkl", "rb") as fp:
    λ, label_names, parameters, W, H = pickle.load(fp)

with open("../applications/harps-sandbox/20240816_train_telfit_model.pkl", "rb") as fp:
    tel_λ, tel_label_names, tel_parameters, tel_W, tel_H = pickle.load(fp)

model = Clam(λ, H)

meta = Table.read("../applications/harps/alfCenA/meta.csv")

#path = "../applications/harps/alfCenA/" + meta["path"][-30]#np.argmin(np.abs(meta["snr"] - 50))]
#path = '../applications/harps/alfCenA/calib/2011-02-27/HARPS.2011-02-28T06:49:35.176_e2ds_A.fits'

# subsample to  only get 1 per snr bin
n_counts = np.zeros(50)
keep = np.zeros(len(meta), dtype=bool)

for i, row in enumerate(meta):
    index = int(row["snr"]) - 1
    if index > (len(n_counts) - 1):
        keep[i] = False
        continue

    if n_counts[index] > 0:
        keep[i] = False
    else:
        keep[i] = True
        n_counts[index] += 1

meta_subset = meta[keep]

# Get the reerence spectrum
reference_index = np.argmin(np.abs(meta["snr"] - 200))
reference_path = "../applications/harps/alfCenA/" + meta["path"][reference_index]
reference_spectra = load_harps_e2ds(reference_path)
initial_order = get_closest_order(reference_spectra, 4861)

from tqdm import tqdm
results = []
for row in tqdm(meta_subset):

    path = "../applications/harps/alfCenA/" + row["path"]
    spectra = load_harps_e2ds(path)

    (result, continuum, rectified_model_flux, rectified_telluric_flux, y_pred) = model.fit(
        reference_spectra + spectra, 
        continuum_basis=Sinusoids(5),
        initial_order=initial_order,
        v_rel=0.0 # already at rest frame.
    )
    R = len(reference_spectra)
    percentiles = np.percentile(np.array([s.flux / c for s, c in zip(spectra, continuum[R:])]).flatten(), [90, 95, 97.5])
    results.append((row["path"], row["snr"], result.status, result.cost, result.optimality, *percentiles))

from astropy.table import Table
t = Table(rows=results, names=("path", "snr", "status", "cost", "optimality", "p_90", "p_95", "p_97.5"))
t.write("alfCenA_stability_with_reference.csv")
