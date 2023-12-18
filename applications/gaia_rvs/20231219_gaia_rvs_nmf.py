import numpy as np
from sklearn.decomposition import NMF
from astropy.io import fits
import h5py as h5
from astropy.table import Table
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor


EPSILON = np.finfo(np.float32).eps

galah = Table.read("GALAH_DR3_main_allstar_v2.fits")
keep = (
    np.isfinite(galah["teff"]) \
&   np.isfinite(galah["logg"]) \
&   np.isfinite(galah["fe_h"]) \
&   np.isfinite(galah["alpha_fe"])
)
galah = galah[keep]

fp = h5.File("/Users/andycasey/Downloads/dr3-rvs-all.hdf5", "r")


_, A_idx, B_idx = np.intersect1d(fp["source_id"][:], np.array(galah["dr3_source_id"]), assume_unique=False, return_indices=True)

N = 10_000

np.random.seed(1234)


try:
    flux
except NameError:
    A_indices = np.sort(np.random.choice(A_idx, N, replace=False))

    flux = fp["flux"][A_indices]
    ivar = fp["flux_error"][A_indices]**-2

B_indices = B_idx[np.intersect1d(A_idx, A_indices, return_indices=True)[1]]

X = np.clip(1 - flux, 0, 1)
X[~np.isfinite(X)] = 0
V = ivar
is_bad_pixel = (
    (V == 0)
|   (~np.isfinite(V))
|   (X < 0)
|   (~np.isfinite(X))
)
V[is_bad_pixel] = X[is_bad_pixel] = 0


def _multiplicative_update(X, V, W, H, update_H=True, update_W=True, l1_reg_H=0, l2_reg_H=0, l1_reg_W=0, l2_reg_W=0):
    WH = W @ H
    if update_H:
        numerator = ((V.T * X.T) @ W).T
        denominator = ((V.T * WH.T) @ W).T
        if l1_reg_H > 0:
            denominator += l1_reg_H
        if l2_reg_H > 0:
            denominator += l2_reg_H * H
        denominator[denominator == 0] = EPSILON
        H *= numerator / denominator
    
    if update_W:
        numerator = (X * V) @ H.T
        denominator = (V * WH) @ H.T
        if l1_reg_W > 0:
            denominator += l1_reg_W
        if l2_reg_W > 0:
            denominator += l2_reg_W * W
        W *= numerator/denominator
    return None


def transformed_labels(labels):
    min_label, max_label = (np.min(labels), np.max(labels))
    return (labels - min_label) / (max_label - min_label)
    

from sklearn.decomposition._nmf import _initialize_nmf

n_components = 8

W, H = _initialize_nmf(X, n_components, init=None, eps=1e-6, random_state=None)

# initialize W by some normalised labels (not necessary, but helps)
W[:, 1] = 1 - transformed_labels(galah["teff"][B_indices])
W[:, 2] = transformed_labels(galah["logg"][B_indices])
W[:, 3] = transformed_labels(galah["fe_h"][B_indices])
W[:, 4] = transformed_labels((galah["alpha_fe"] + galah["fe_h"])[B_indices])

#for i in range(10):
#    _multiplicative_update(X, V, W, H, update_H=True, update_W=False)

from tqdm import tqdm
for i in tqdm(range(1000)):
    _multiplicative_update(X, V, W, H, update_H=True, update_W=True)



'''
rf = RandomForestRegressor()

L = np.vstack([
    np.array(galah["teff"][B_indices]),
    np.array(galah["logg"][B_indices]),
    np.array(galah["fe_h"][B_indices]),
    np.array((galah["alpha_fe"] + galah["fe_h"])[B_indices])
]).T

rf.fit(W, L)

'''
rf = dict(
    teff=RandomForestRegressor(),
    logg=RandomForestRegressor(),
    fe_h=RandomForestRegressor(),
    alpha_fe=RandomForestRegressor()
)
for k, m in rf.items():
    m.fit(W, galah[k][B_indices])


galah["alpha_h"] = galah["alpha_fe"] + galah["fe_h"]
label_names = ("teff", "logg", "fe_h", "alpha_fe")
y_pred = np.vstack([rf[k].predict(W) for k in label_names]).T

fig, axes = plt.subplots(2, 2)
for i, (ax, label_name) in enumerate(zip(axes.flat, label_names)):
    x, y = (galah[label_name][B_indices], y_pred[:, i])
    ax.scatter(x, y, s=1, alpha=0.5)
    
    lim = (ax.get_xlim(), ax.get_ylim())
    lim = np.min(lim), np.max(lim)
    ax.plot(lim, lim, c="#666666", ls=":")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    
    ax.set_xlabel(f"{label_name} (GALAH)")
    ax.set_ylabel(f"{label_name}")
    
    mu = np.mean(y - x)
    sigma = np.std(y - x)
    ax.set_title(f"{mu:.2f} +/- {sigma:.2f}")


vmin, vmax = (-2.5, 0.5)
fig, (ax_galah, ax_gaia) = plt.subplots(1, 2, figsize=(8, 4))

scat = ax_galah.scatter(
    galah["teff"][B_indices],
    galah["logg"][B_indices],
    c=galah["fe_h"][B_indices],
    vmin=vmin,
    vmax=vmax,
    s=1
)
ax_gaia.scatter(
    rf["teff"].predict(W),
    rf["logg"].predict(W),
    c=rf["fe_h"].predict(W),
    vmin=vmin,
    vmax=vmax,
    s=1  
)

cbar = plt.colorbar(scat, ax=[ax_galah, ax_gaia])
cbar.set_label("[Fe/H]")


for ax in (ax_galah, ax_gaia):
    ax.set_xlim(7500, 3000)
    ax.set_ylim(5, 0)
    ax.set_xlabel("Teff")
    ax.set_ylabel("logg")

ax_galah.set_title("GALAH")
ax_gaia.set_title("Gaia RVS")



fig, ax = plt.subplots()
for i in range(n_components):
    ax.plot(1 - H[i] / H[i].max() + i, label=i)


N = 10_000
np.random.seed(0)

# warning: this test set will not be strictly disjoint from the training set, but the chance of overlap is pretty small
test_indices = np.sort(np.random.choice(fp["source_id"].size, N, replace=False))

test_flux = fp["flux"][test_indices]
test_ivar = fp["flux_error"][test_indices]**-2

X_test = np.clip(1 - test_flux, 0, 1)
X_test[~np.isfinite(X_test)] = 0
V_test = test_ivar
is_bad_pixel = (
    (V_test == 0)
|   (~np.isfinite(V_test))
|   (X_test < 0)
|   (~np.isfinite(X_test))
)
V_test[is_bad_pixel] = X_test[is_bad_pixel] = 0

# compute W given H in a shitty way (should use linear algebra instead so that we use the errors correctly)
# X = WH + noise
model = NMF(n_components=n_components)
model.components_ = H
W_test = model.transform(X_test)

y_pred = np.vstack([rf[k].predict(W) for k in label_names]).T

vmin, vmax = (-2.5, 0.5)
fig, ax = plt.subplots()
scat = ax.scatter(
    y_pred[:, 0],
    y_pred[:, 1],
    c=y_pred[:, 2],
    vmin=vmin,
    vmax=vmax,
    s=1,
)
cbar = plt.colorbar(scat)
cbar.set_label("[Fe/H]")
ax.set_xlabel("Teff")
ax.set_ylabel("logg")
