import numpy as np
from sklearn.decomposition import NMF

import h5py as h5

fp = h5.File("/Users/andycasey/Downloads/dr3-rvs-all.hdf5", "r")

np.random.seed(1234)

N = 100_000

try:
    snr
except NameError:
    snr = np.nanmean(fp["flux"][:] / fp["flux_error"][:], axis=1)
    indices = np.sort(np.random.choice(np.where(snr >= 50)[0], N, replace=False))

    flux = fp["flux"][indices]
    source_id = fp["source_id"][indices]


absorption = np.clip(1 - flux, 0, 1)
absorption[~np.isfinite(absorption)] = 0


model = NMF(n_components=8, solver="cd", max_iter=1_000, verbose=True)
W = model.fit_transform(absorption)
H = model.components_

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
for i in range(model.n_components):
    ax.plot(1 - H[i] / H[i].max() + i, label=i)
    
    
from astropy.table import Table

names, data = (["source_id", "gaia_rvs_snr"], [source_id, snr[indices]])
for i in range(W.shape[1]):
    data.append(W[:, i])
    names.append(f"W{i}")
    
t = Table(data=data, names=names)
t.write("20231218_gaia_rvs_nmf.csv")