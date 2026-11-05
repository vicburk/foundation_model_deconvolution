from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.optimize import nnls
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR


class deconvolution:
    def __init__(self, reference_library, pseudobulk, proportion):
        if type(reference_library) == sc.AnnData:
            self.reference_library = pd.DataFrame(reference_library.X)
            self.reference_library['cell_type'] = reference_library.obs.select_dtypes(include='category').values.squeeze()
        else:
            self.reference_library = reference_library
        if type(pseudobulk) == sc.AnnData:
            self.pseudobulk = pd.DataFrame(pseudobulk.X)
        else:
            self.pseudobulk = pseudobulk
        self.proportion = proportion
        self.estimates = {}

    # ------------------------------------------------------------------
    # Internal per-sample workers (static so joblib can pickle them)
    # ------------------------------------------------------------------
    @staticmethod
    def _make_proportion_map(proportion_row):
        """Return a dict mapping cell_type -> proportion regardless of input shape."""
        if isinstance(proportion_row, pd.Series):
            # Index is cell_type, values are proportions
            return proportion_row.to_dict()
        elif isinstance(proportion_row, pd.DataFrame):
            # Expect columns 'cell_type' and 'proportion'
            return dict(zip(proportion_row['cell_type'], proportion_row['proportion']))
        return {}

    @staticmethod
    def _nnls_one(index, A, b_row, cell_types, proportion_row):
        b = np.array(b_row).flatten()
        x, _ = nnls(A, b)
        if x.sum() > 0:
            x = x / x.sum()
        prop_map = deconvolution._make_proportion_map(proportion_row)
        df = pd.DataFrame({
            'nnls_estimate': x,
            'cell_type': cell_types,
            'proportion': [prop_map.get(ct, 0.0) for ct in cell_types]
        })
        return df

    @staticmethod
    def _svr_one(index, A_fit, b_row, cell_types, proportion_row, scaler_genes, pca_model=None):
        b = np.array(b_row).flatten()
        b_scaled = scaler_genes.transform(b.reshape(1, -1))  # (1, n_genes)
        if pca_model is not None:
            b_fit = pca_model.transform(b_scaled).ravel()    # (n_components,)
        else:
            b_fit = b_scaled.ravel()
        model = NuSVR(kernel='linear', nu=0.5, C=1.0)
        model.fit(A_fit, b_fit)
        raw_x = model.coef_[0].copy()
        raw_x[raw_x < 0] = 0
        x = raw_x / np.sum(raw_x) if raw_x.sum() > 0 else raw_x
        prop_map = deconvolution._make_proportion_map(proportion_row)
        df = pd.DataFrame({
            'svr_estimate': x,
            'cell_type': cell_types,
            'proportion': [prop_map.get(ct, 0.0) for ct in cell_types]
        })
        return df

    # ------------------------------------------------------------------
    # Public fitting methods
    # ------------------------------------------------------------------
    def nnls(self, n_jobs=-1):
        A = self.reference_library.select_dtypes(include='number').T.values
        cell_types = self.reference_library['cell_type'].values
        self.estimates["nnls"] = Parallel(n_jobs=n_jobs)(
            delayed(self._nnls_one)(
                i, A,
                self.pseudobulk.iloc[i],
                cell_types,
                self.proportion.iloc[i]
            )
            for i in range(len(self.pseudobulk))
        )

    def svr(self, pca=True, n_components=50, n_jobs=-1):
        # A_raw: (n_genes, n_cells) — genes are features, cells are samples
        A_raw = self.reference_library.select_dtypes(include='number').values  # (n_cells, n_genes)
        # Fit scaler on gene axis: each gene (column) is a feature across cells
        scaler_genes = StandardScaler()
        A_scaled = scaler_genes.fit_transform(A_raw)  # (n_cells, n_genes)
        pca_model = None
        if pca:
            n_components = min(n_components, A_scaled.shape[0], A_scaled.shape[1])
            pca_model = PCA(n_components=n_components)
            A_fit = pca_model.fit_transform(A_scaled)  # (n_cells, n_components)
            print(f"variance explained: {pca_model.explained_variance_ratio_.sum()}")
        else:
            A_fit = A_scaled.T  # (n_genes, n_cells) — genes as samples, cells as features
        cell_types = self.reference_library['cell_type'].values
        self.estimates["svr"] = Parallel(n_jobs=n_jobs)(
            delayed(self._svr_one)(
                i, A_fit,
                self.pseudobulk.iloc[i],
                cell_types,
                self.proportion.iloc[i],
                scaler_genes,
                pca_model
            )
            for i in range(len(self.pseudobulk))
        )

    # ------------------------------------------------------------------
    # Metrics (unchanged)
    # ------------------------------------------------------------------
    def mask_mse(self, method):
        try:
            mse_list = np.array([])
            estimates = self.estimates[method]
            for i in range(len(estimates)):
                A = estimates[i][f'{method}_estimate']
                B = estimates[i]['proportion']
                mask = (A != 0) | (B != 0)
                mse = ((A[mask] - B[mask]) ** 2).mean()
                mse_list = np.append(mse_list, mse)
            return mse_list
        except AttributeError as e:
            print(f"run {method} before calculating mse: {e}")

    def max_error(self, method):
        try:
            error_list = np.array([])
            estimates = self.estimates[method]
            for i in range(len(estimates)):
                A = estimates[i][f'{method}_estimate']
                B = estimates[i]['proportion']
                error_list = np.append(error_list, abs(A - B).max())
            return error_list
        except AttributeError as e:
            print(f"run {method} before calculating max error: {e}")

    def kl_divergence(self, method, eps=1e-8, mean=True):
        try:
            kl_list = np.array([])
            estimates = self.estimates[method]
            for i in range(len(estimates)):
                A = estimates[i][f'{method}_estimate'] + eps
                B = estimates[i]['proportion'] + eps
                kl = (A * np.log(A / B)).sum()
                if mean:
                    kl /= len(A)
                kl_list = np.append(kl_list, kl)
            return kl_list
        except AttributeError as e:
            print(f"run {method} before calculating KL Divergence: {e}")

    def js_divergence(self, method, eps=1e-8, mean=True):
        try:
            js_list = np.array([])
            estimates = self.estimates[method]
            for i in range(len(estimates)):
                A = estimates[i][f'{method}_estimate'] + eps
                B = estimates[i]['proportion'] + eps
                M = (A + B) / 2
                js = 0.5 * ((A * np.log(A / M)).sum() + (B * np.log(B / M)).sum())
                if mean:
                    js /= len(A)
                js_list = np.append(js_list, js)
            return js_list
        except AttributeError as e:
            print(f"run {method} before calculating JS Divergence: {e}")

    def error_by_cell_type(self, method):
        try:
            estimates = self.estimates[method]
            resid = [abs(est[f'{method}_estimate'] - est['proportion']) for est in estimates]
            residuals = pd.DataFrame(resid).T
            residuals['cell_type'] = estimates[0]['cell_type'].values
            res = residuals.drop(columns='cell_type')
            mask_res_by_cell = [
                np.mean(row[row > 0]) if (row > 0).any() else 0.0
                for _, row in res.iterrows()
            ]
            return pd.DataFrame({'cell_type': residuals['cell_type'], 'error': mask_res_by_cell})
        except AttributeError as e:
            print(f"run {method} before calculating error by cell type: {e}")


