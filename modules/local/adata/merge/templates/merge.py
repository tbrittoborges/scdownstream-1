#!/usr/bin/env python3

import scanpy as sc
import anndata as ad
from scipy.sparse import csr_matrix
import platform
import scipy

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

adatas = [sc.read_h5ad(f) for f in "${h5ads}".split()]

base_path = "${base}"
if base_path:
    adata_base = sc.read_h5ad(base_path)
    adatas = [adata_base] + adatas

genes = [adata.var_names for adata in adatas]
obs_col_intersection = set(adatas[0].obs.columns).intersection(*[adata.obs.columns for adata in adatas[1:]])
obs_col_intersection = list(obs_col_intersection.union("${force_obs_cols}".split()))
sorted(obs_col_intersection)

for adata in adatas:
    for col in set(obs_col_intersection).difference(adata.obs.columns):
        adata.obs[col] = None
    adata.obs = adata.obs[obs_col_intersection]

adata_outer = ad.concat(adatas, join="outer")
adata_outer.X = csr_matrix(adata_outer.X)

# Make sure there are no cells and genes without any counts
sc.pp.filter_cells(adata_outer, min_counts=1)
sc.pp.filter_genes(adata_outer, min_cells=1)

genes_intersection = list(set(adata_outer.var_names).intersection(*genes))
sorted(genes_intersection)
adata_inner = adata_outer[:, genes_intersection]

adata_inner.write("${prefix}_inner.h5ad")
adata_outer.write("${prefix}_outer.h5ad")

if base_path:
    adata_transfer = adata_inner[~adata_inner.obs.index.isin(adata_base.obs.index)]

    known_labels = adata_base.obs["label"].unique()
    adata_transfer.obs["label"] = adata_transfer.obs["label"].map(
        lambda x: x if x in known_labels else "unknown"
    )

    adata_transfer.write("${prefix}_transfer.h5ad")

# Versions

versions = {
    "${task.process}": {
        "python": platform.python_version(),
        "scanpy": sc.__version__,
        "anndata": ad.__version__,
        "scipy": scipy.__version__,
    }
}

with open("versions.yml", "w") as f:
    f.write(format_yaml_like(versions))
