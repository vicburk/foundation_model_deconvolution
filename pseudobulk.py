import os
import pickle as pkl
import numpy as np
import pandas as pd
import scanpy as sc
from dataclasses import dataclass, field
from joblib import Parallel, delayed


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PseudoReplicate:
    pseudobulk:  sc.AnnData
    proportions: list
    sub_index:   tuple


@dataclass
class ReplicateSet:
    pseudo: list[PseudoReplicate] = field(default_factory=list)

    def __len__(self):
        return len(self.pseudo)


# ---------------------------------------------------------------------------
# Index containers
# ---------------------------------------------------------------------------
# PseudoIndices is a plain dict rather than a dataclass so that joblib/loky
# can always pickle it without needing to resolve a class name back to a
# module — which fails when code is defined in a Jupyter notebook cell.
#
# Schema:
#   {
#     'sub_indices': tuple[np.ndarray, ...],   # positional indices per sample
#     'obs_names':   np.ndarray,               # cell barcodes for the split
#     'proportions': list[pd.Series],          # one Series per pseudobulk
#   }
#
# Helper so call-sites stay readable:

def make_pseudo_indices(
    sub_indices: tuple,
    obs_names: np.ndarray,
    proportions: list,
) -> dict:
    return {'sub_indices': sub_indices, 'obs_names': obs_names, 'proportions': proportions}


# Type alias for annotations (does not affect pickling)
PseudoIndices = dict


# ---------------------------------------------------------------------------
# Cell sampling helpers
# ---------------------------------------------------------------------------

def cells_to_select(adata: sc.AnnData, cell_column: str, ntype: int) -> np.ndarray:
    cell_types = adata.obs[cell_column].unique()
    n = np.random.randint(ntype) + 2
    return np.random.choice(cell_types, n)


def generate_sample_size(
    adata: sc.AnnData, cell_column: str, ntype: int, num_cells_total: int
) -> tuple[np.ndarray, np.ndarray]:
    cell_types = cells_to_select(adata, cell_column, ntype)
    n = len(cell_types)
    p_int = np.random.randint(10, size=n) + 1
    noise = np.random.choice([0.5, 10, 20], size=n, p=[0.8, 0.15, 0.05])
    pseudo_p = p_int * noise
    pseudo_p /= pseudo_p.sum()
    num_cells_per_type = (pseudo_p * num_cells_total).astype(int)
    return cell_types, num_cells_per_type


def sample_cells(
    adata: sc.AnnData, cell_column: str, ntype: int, num_cells_total: int
) -> tuple[sc.AnnData, list[int]]:
    cell_types, num_cells_per_type = generate_sample_size(adata, cell_column, ntype, num_cells_total)
    sampled_indices = []
    for cell_type, num_cells in zip(cell_types, num_cells_per_type):
        cell_type_indices = np.where(adata.obs[cell_column] == cell_type)[0]
        if len(cell_type_indices) >= num_cells:
            selected = np.random.choice(cell_type_indices, size=num_cells, replace=False)
        else:
            selected = cell_type_indices
        sampled_indices.extend(selected)

    adata_sampled = adata[sampled_indices, :].copy()
    adata_sampled.obs_names_make_unique()
    return adata_sampled, sampled_indices


def pseudobulk_generation(
    adata: sc.AnnData, cell_column: str, ntype: int, num_cells_total: int
) -> tuple[pd.DataFrame, pd.Series, list[int]]:
    adata_sampled, sampled_indices = sample_cells(adata, cell_column, ntype, num_cells_total)
    pseudobulk = pd.DataFrame(
        adata_sampled.X.sum(axis=0),
        columns=['pseudobulk_counts'],
        index=adata_sampled.var['ensembl_id'],
    )
    true_proportions = adata_sampled.obs[cell_column].value_counts(normalize=True)
    return pseudobulk, true_proportions, sampled_indices


def make_adata(data: list[pd.DataFrame], original: sc.AnnData) -> sc.AnnData:
    X = np.array([list(x['pseudobulk_counts']) for x in data])
    adata = sc.AnnData(X)
    adata.var_names         = original.var_names
    adata.var['ensembl_id'] = original.var['ensembl_id']
    adata.var.index.name    = None
    adata.obs['n_counts']   = adata.X.sum(axis=1)
    adata.obs['index_col']  = np.arange(len(data))
    return adata


# ---------------------------------------------------------------------------
# Reference library (no sampling — call once on your test adata)
# ---------------------------------------------------------------------------

def build_reference_library(
    adata: sc.AnnData,
    cell_column: str = 'final_annotation',
) -> sc.AnnData:
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    lib_df = (
        pd.DataFrame(X, index=adata.obs[cell_column])
        .groupby(level=0)
        .sum()
    )
    library = sc.AnnData(
        X=lib_df.values,
        obs=pd.DataFrame({'cell_type': lib_df.index}, index=lib_df.index),
    )
    library.var_names         = adata.var_names
    library.var['ensembl_id'] = adata.var['ensembl_id']
    library.var.index.name    = None
    library.obs['n_counts']   = library.X.sum(axis=1)
    return library


# ---------------------------------------------------------------------------
# Single replicate generator (parallelisable unit of work)
# ---------------------------------------------------------------------------

def generate_replicate(
    adata_split: sc.AnnData,
    N: int,
    ntype: int = 5,
    sample_size: int = 1000,
    cell_column: str = 'final_annotation',
) -> ReplicateSet:
    pseudobulk_samples = [
        pseudobulk_generation(adata_split, cell_column, ntype, sample_size)
        for _ in range(N)
    ]
    pseudo_list, proportions, sub_indices = zip(*pseudobulk_samples)
    pseudobulk = make_adata(list(pseudo_list), adata_split)

    pseudo_rep = PseudoReplicate(
        pseudobulk=pseudobulk,
        proportions=list(proportions),
        sub_index=sub_indices,
    )
    return ReplicateSet(pseudo=[pseudo_rep])


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def load_pseudobulk(session_path: str, split_name: str) -> list[sc.AnnData]:
    split_dir = os.path.join(session_path, split_name)
    rep_dirs = sorted(
        d for d in os.listdir(split_dir)
        if d.startswith('replicate_') and os.path.isdir(os.path.join(split_dir, d))
    )
    adatas = [sc.read_h5ad(os.path.join(split_dir, d, 'temp.h5ad')) for d in rep_dirs]
    print(f'[{split_name}] Loaded {len(adatas)} pseudobulk h5ad files from {split_dir}')
    return adatas


def load_indices(drive_path: str, split_name: str) -> list[PseudoIndices]:
    idx_dir = os.path.join(drive_path, split_name, 'indices')
    files = sorted(
        f for f in os.listdir(idx_dir)
        if f.startswith('replicate_') and f.endswith('.pkl')
    )
    all_indices = []
    for f in files:
        with open(os.path.join(idx_dir, f), 'rb') as fh:
            all_indices.append(pkl.load(fh))
    print(f'[{split_name}] Loaded {len(all_indices)} index files from {idx_dir}')
    return all_indices


def load_proportions(drive_path: str, split_name: str) -> list[pd.DataFrame]:
    prop_dir = os.path.join(drive_path, split_name, 'proportions')
    files = sorted(
        f for f in os.listdir(prop_dir)
        if f.startswith('replicate_') and f.endswith('.csv')
    )
    dfs = [pd.read_csv(os.path.join(prop_dir, f), index_col=0) for f in files]
    print(f'[{split_name}] Loaded {len(dfs)} proportion CSVs from {prop_dir}')
    return dfs


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------

def reconstruct_pseudobulk(
    pseudo_idx: PseudoIndices,
    adata_split: sc.AnnData,
) -> sc.AnnData:
    rows = []
    for sub_idx in pseudo_idx['sub_indices']:
        adata_sub = adata_split[sub_idx, :]
        X_sub = adata_sub.X.toarray() if hasattr(adata_sub.X, 'toarray') else adata_sub.X
        rows.append(X_sub.sum(axis=0))

    X = np.vstack(rows)
    pseudobulk = sc.AnnData(X)
    pseudobulk.var_names         = adata_split.var_names
    pseudobulk.var['ensembl_id'] = adata_split.var['ensembl_id']
    pseudobulk.obs['n_counts']   = pseudobulk.X.sum(axis=1)
    return pseudobulk


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_indices(
    pseudo_idx: PseudoIndices,
    adata_split: sc.AnnData,
    cell_column: str = 'final_annotation',
    tol: float = 1e-6,
) -> bool:
    all_pass = True
    for i, (sub_idx, saved_props) in enumerate(
        zip(pseudo_idx['sub_indices'], pseudo_idx['proportions'])
    ):
        adata_sub = adata_split[sub_idx, :]
        reconstructed = adata_sub.obs[cell_column].value_counts(normalize=True)

        all_types     = saved_props.index.union(reconstructed.index)
        saved_aligned = saved_props.reindex(all_types, fill_value=0)
        recon_aligned = reconstructed.reindex(all_types, fill_value=0)

        diff     = (saved_aligned - recon_aligned).abs()
        max_diff = diff.max()

        if max_diff > tol:
            all_pass = False
            print(f'  Sample {i + 1}: FAIL (max diff = {max_diff:.2e})')
            print(diff[diff > tol].to_string())
        else:
            print(f'  Sample {i + 1}: PASS (max diff = {max_diff:.2e})')

    return all_pass


def validate_all_replicates(
    all_indices: list[PseudoIndices],
    adata_split: sc.AnnData,
    cell_column: str = 'final_annotation',
    tol: float = 1e-6,
) -> None:
    for i, pseudo_idx in enumerate(all_indices):
        print(f'\nReplicate {i + 1}:')
        passed = validate_indices(pseudo_idx, adata_split, cell_column, tol)
        print(f"  → {'ALL PASS' if passed else 'FAILURES DETECTED'}")


# ---------------------------------------------------------------------------
# Module-level worker — must be at module scope to be picklable by loky
# ---------------------------------------------------------------------------

def _replicate_worker(
    i: int,
    adata_split: sc.AnnData,
    split_name: str,
    n: int,
    N: int,
    ntype: int,
    sample_size: int,
    cell_column: str,
    obs_names: np.ndarray,
    idx_dir: str,
    prop_dir: str,
    session_path: str,
) -> None:
    """
    Generate one replicate and write all outputs to disk.
    Defined at module scope so loky can pickle it by name.
    """
    rs  = generate_replicate(adata_split, N, ntype, sample_size, cell_column)
    rep = rs.pseudo[0]

    # pseudobulk h5ad
    rep_dir = os.path.join(session_path, split_name, f'replicate_{i + 1}')
    os.makedirs(rep_dir, exist_ok=True)
    rep.pseudobulk.write_h5ad(os.path.join(rep_dir, 'temp.h5ad'))

    # indices pickle — plain dict, always picklable from any context
    pseudo_idx = make_pseudo_indices(
        sub_indices=rep.sub_index,
        obs_names=obs_names,
        proportions=rep.proportions,
    )
    with open(os.path.join(idx_dir, f'replicate_{i + 1}.pkl'), 'wb') as f:
        pkl.dump(pseudo_idx, f)

    # proportions CSV
    df = pd.DataFrame(rep.proportions).fillna(0)
    df.index = [f'sample_{j + 1}' for j in range(len(df))]
    df.to_csv(os.path.join(prop_dir, f'replicate_{i + 1}.csv'))

    print(f'  [{split_name}] Replicate {i + 1}/{n} saved.')


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def process_split(
    adata_split: sc.AnnData,
    split_name: str,
    drive_path: str,
    session_path: str,
    n: int = 5,
    N: int = 20,
    ntype: int = 5,
    sample_size: int = 1000,
    cell_column: str = 'final_annotation',
    n_jobs: int = -1,
) -> None:
    idx_dir  = os.path.join(drive_path, split_name, 'indices')
    prop_dir = os.path.join(drive_path, split_name, 'proportions')
    for d in (idx_dir, prop_dir):
        os.makedirs(d, exist_ok=True)

    obs_names = np.array(adata_split.obs_names)

    print(f'[{split_name}] Generating {n} replicates × {N} pseudobulks...')
    Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_replicate_worker)(
            i, adata_split, split_name, n, N, ntype, sample_size,
            cell_column, obs_names, idx_dir, prop_dir, session_path,
        )
        for i in range(n)
    )
    print(f'[{split_name}] Done.')


# ---------------------------------------------------------------------------
# Load or generate
# ---------------------------------------------------------------------------

def load_or_generate(
    adata_split: sc.AnnData,
    split_name: str,
    drive_path: str,
    session_path: str,
    n: int = 5,
    N: int = 20,
    ntype: int = 5,
    sample_size: int = 1000,
    cell_column: str = 'final_annotation',
    n_jobs: int = -1,
) -> list[sc.AnnData]:
    idx_dir  = os.path.join(drive_path, split_name, 'indices')
    prop_dir = os.path.join(drive_path, split_name, 'proportions')

    drive_ready = (
        os.path.isdir(idx_dir)
        and os.path.isdir(prop_dir)
        and len([f for f in os.listdir(idx_dir) if f.endswith('.pkl')]) == n
    )

    if drive_ready:
        print(f'[{split_name}] Indices found on Drive — reconstructing pseudobulk.')
        all_indices = load_indices(drive_path, split_name)
        adatas = []
        for i, pseudo_idx in enumerate(all_indices):
            adata_pb = reconstruct_pseudobulk(pseudo_idx, adata_split)
            rep_dir  = os.path.join(session_path, split_name, f'replicate_{i + 1}')
            os.makedirs(rep_dir, exist_ok=True)
            adata_pb.write_h5ad(os.path.join(rep_dir, 'temp.h5ad'))
            adatas.append(adata_pb)
            print(f'  [{split_name}] Replicate {i + 1}/{n} reconstructed and saved to session.')
    else:
        print(f'[{split_name}] No indices on Drive — generating pseudobulk from scratch.')
        process_split(
            adata_split  = adata_split,
            split_name   = split_name,
            drive_path   = drive_path,
            session_path = session_path,
            n            = n,
            N            = N,
            ntype        = ntype,
            sample_size  = sample_size,
            cell_column  = cell_column,
            n_jobs       = n_jobs,
        )
        adatas = load_pseudobulk(session_path, split_name)

    return adatas
