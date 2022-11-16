import io
import math
import numpy as np
import pandas as pd
import pickle
from pycocotools import mask
from scipy import stats
import statsmodels.formula.api as smf
import torch


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def parse_pkl_filename(pkl_path):
    path = str(pkl_path).split('/')
    task = path[-1].split('_')[-2]
    img_id = '_'.join(path[-1].split('_')[:-2])
    return task, img_id


def encode_segmentation(segmentation_arr):
    """
    Encode a binary segmentation (np.array) to RLE format using the pycocotools Mask API.
    Args:
        segmentation_arr (np.array): [h x w] binary segmentation
    Returns:
		Rs (dict): the encoded mask in RLE format
    """
    segmentation = np.asfortranarray(segmentation_arr.astype('uint8'))
    Rs = mask.encode(segmentation)
    Rs['counts'] = Rs['counts'].decode()
    return Rs


def run_linear_regression(regression_df, task, y, x):
    """
    Run linear regression model given a regression dataframe of a single pathology.

    Args:
        task (str): localization task
        y (str): the dependent variable
        x (str): the independent variable
    """
    est = smf.ols(f"{y} ~ {x}", data = regression_df)
    est2 = est.fit()
    ci = est2.conf_int(alpha=0.05, cols=None)
    lower, upper = ci.loc[x]
    mean = est2.params.loc[x]
    pval = est2.pvalues.loc[x]
    corr, corr_pval = stats.spearmanr(regression_df[y].values,regression_df[x].values,nan_policy = 'omit')
    n = len(regression_df[regression_df[y].notnull()])
    stderr = 1.0 / math.sqrt(n - 3)
    delta = 1.96 * stderr
    lower_r = math.tanh(math.atanh(corr) - delta)
    upper_r = math.tanh(math.atanh(corr) + delta)

    results = {'lower': round(lower,3),
               'upper': round(upper,3),
               'mean': round(mean,3),
               'coef_pval': pval,
               'corr_lower': round(lower_r,3),
               'corr_upper': round(upper_r,3),
               'corr': round(corr,3),
               'corr_pval': corr_pval,
               'n': int(n),
               'feature': x,
               'task': task}
    return pd.DataFrame([results])


def format_ci(row, **kwargs):
    """Format confidence interval."""
    def format_stats_sig(p_val):
        """Output *, **, *** based on p-value."""
        stats_sig_level = ''
        if p_val < 0.001 / kwargs['bonferroni_correction']:
            stats_sig_level = '***'
        elif p_val < 0.01 / kwargs['bonferroni_correction']:
            stats_sig_level = '**'
        elif p_val < 0.05 / kwargs['bonferroni_correction']:
            stats_sig_level = '*'
        return stats_sig_level

    # CI on linear regression coefficients
    p_val = row['coef_pval']
    stats_sig_level = format_stats_sig(p_val)
    mean = row['mean']
    upper = row['upper']
    lower = row['lower']
    row['Linear regression coefficients'] =  f'{mean}, ({lower}, {upper}){stats_sig_level}'

    # CI on spearman correlations
    p_val = row['corr_pval']
    stats_sig_level = format_stats_sig(p_val)
    mean = row['corr']
    upper = row['corr_upper']
    lower = row['corr_lower']
    row['Spearman correlations'] =  f'{mean}, ({lower}, {upper}){stats_sig_level}'

    return row
