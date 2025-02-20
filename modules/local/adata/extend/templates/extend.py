#!/usr/bin/env python3

import platform
import anndata as ad
import pandas as pd
import numpy as np
import os

def format_yaml_like(data: dict, indent: int = 0) -> str:
    """Formats a dictionary to a YAML-like string.

    Args:
        data (dict): The dictionary to format.
        indent (int): The current indentation level.

    Returns:
        str: A string formatted as YAML.
    """
    yaml_str = ""
    for key, value in data.items():
        spaces = "  " * indent
        if isinstance(value, dict):
            yaml_str += f"{spaces}{key}:\\n{format_yaml_like(value, indent + 1)}"
        else:
            yaml_str += f"{spaces}{key}: {value}\\n"
    return yaml_str

adata = ad.read_h5ad("${base}")
prefix = "${prefix}"
obs_paths = "${obs}".split()
obsm_paths = "${obsm}".split()
layers_paths = "${layers}".split()

for path in obs_paths:
    df = pd.read_pickle(path).reindex(adata.obs_names)
    adata.obs = pd.concat([adata.obs, df], axis=1)

for path in obsm_paths:
    df = pd.read_pickle(path).reindex(adata.obs_names)
    name = os.path.basename(path).split(".")[0]
    adata.obsm[name] = np.float32(df.to_numpy())

for path in layers_paths:
    array = np.load(path)
    name = os.path.basename(path).split(".")[0]
    adata.layers[name] = np.float32(array)

adata.write_h5ad(f"{prefix}.h5ad")
adata.obs.to_csv(f"{prefix}_metadata.csv")

# Versions

versions = {
    "${task.process}": {
        "python": platform.python_version(),
        "anndata": ad.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__
    }
}

with open("versions.yml", "w") as f:
    f.write(format_yaml_like(versions))
