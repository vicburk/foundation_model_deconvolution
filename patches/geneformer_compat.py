# patches/geneformer_compat.py
#
# MONKEY PATCH — remove once upstream fixes positional indexing in tokenize_anndata.
# Tracking issue: <link to Geneformer GitHub issue>
# Applied to Geneformer version: X.Y.Z
#
# Root cause: tokenize_anndata used label-based [] on a positional integer index
# (coding_miRNA_loc). Fixed here with .iloc[].

import geneformer.tokenizer as _tk
import numpy as np
import scipy.sparse as sp


def _patched_tokenize_anndata(self, adata_file_path, target_sum=10_000, file_format="h5ad"):
    from geneformer.tokenizer import sum_ensembl_ids, rank_genes

    adata = sum_ensembl_ids(
        adata_file_path,
        self.collapse_gene_ids,
        self.gene_mapping_dict,
        self.gene_token_dict,
        self.custom_attr_name_dict,
        self.use_h5ad_index,
        file_format=file_format,
        chunk_size=self.chunk_size,
    )

    if self.custom_attr_name_dict is not None:
        file_cell_metadata = {
            attr_key: [] for attr_key in self.custom_attr_name_dict.keys()
        }

    coding_miRNA_loc = np.where(
        [self.genelist_dict.get(i, False) for i in adata.var["ensembl_id_collapsed"]]
    )[0]

    # FIX: use .iloc for positional indexing instead of label-based []
    norm_factor_vector = np.array(
        [self.gene_median_dict[i] for i in adata.var["ensembl_id_collapsed"].iloc[coding_miRNA_loc]]
    )
    coding_miRNA_ids = adata.var["ensembl_id_collapsed"].iloc[coding_miRNA_loc]
    coding_miRNA_tokens = np.array(
        [self.gene_token_dict[i] for i in coding_miRNA_ids]
    )

    try:
        _ = adata.obs["filter_pass"]
        filter_pass_loc = np.where([i == 1 for i in adata.obs["filter_pass"]])[0]
    except KeyError:
        print(f"{adata_file_path} has no column attribute 'filter_pass'; tokenizing all cells.")
        filter_pass_loc = np.array([i for i in range(adata.shape[0])])

    tokenized_cells = []
    tokenized_counts = []

    for i in range(0, len(filter_pass_loc), self.chunk_size):
        idx = filter_pass_loc[i : i + self.chunk_size]
        n_counts = adata[idx].obs["n_counts"].values[:, None]
        X_view = adata[idx, :].X[:, coding_miRNA_loc]
        X_norm = sp.csr_matrix(X_view / n_counts * target_sum / norm_factor_vector)
        X_norm_unscaled = sp.csr_matrix(X_view / n_counts * target_sum)

        tokenized_cells += [
            rank_genes(X_norm[i].data, coding_miRNA_tokens[X_norm[i].indices]).copy()
            for i in range(X_norm.shape[0])
        ]

        if self.keep_counts:
            tokenized_counts += [
                rank_genes(X_norm[i].data, X_norm_unscaled[i].data).copy()
                for i in range(X_norm.shape[0])
            ]

        if self.custom_attr_name_dict is not None:
            for k in file_cell_metadata.keys():
                file_cell_metadata[k] += adata[idx].obs[k].tolist()
        else:
            file_cell_metadata = None

    return tokenized_cells, file_cell_metadata, tokenized_counts

# Monkey-patch before instantiating the tokenizer
_tk.TranscriptomeTokenizer.tokenize_anndata = _patched_tokenize_anndata