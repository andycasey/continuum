
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

paths = [
    ("julia_cubic.csv", "Cubic spline and vsini"),
    ("julia_cubic_no_vsini.csv", "Cubic spline and no vsini"),
    ("julia_linear.csv", "Linear interpolation and vsini"),
    ("julia_linear_no_vsini.csv", "Linear interpolation and no vsini")
]

xlim = (8000, 3000)
ylim = (5.5, 0)

fig, axes = plt.subplots(2, 2)

for ax, (path, title) in zip(axes.flat, paths):
    
    names = [f"c{i}" for i in range(25)]
    try:
        table = Table.read(path, names=names)
    except:
        names.append("c26")        
        table = Table.read(path, names=names)
        
    ax.scatter(table["c0"], table["c1"], c=table["c3"], s=5, alpha=0.5)
    
    ax.set_xlabel("Teff")
    ax.set_ylabel("logg")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title)
fig.tight_layout()