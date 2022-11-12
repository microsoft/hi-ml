import subprocess
import numpy as np
import os
import pandas as pd
from PIL import Image
import h5py
import matplotlib.pyplot as plt
from typing import List, Callable

import torch
from torch.utils import data
from tqdm.notebook import tqdm
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, Resize

import sklearn
from sklearn.metrics import matthews_corrcoef, confusion_matrix, accuracy_score, auc, roc_auc_score, roc_curve, classification_report
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.metrics import average_precision_score
from sklearn.utils import resample 

import scipy
import scipy.stats

import sys
sys.path.append('../..')


def compute_mean(stats, is_df=True): 
    spec_labels = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
    if is_df: 
        spec_df = stats[spec_labels]
        res = np.mean(spec_df.iloc[0])
    else: 
        # cis is df, within bootstrap
        vals = [stats[spec_label][0] for spec_label in spec_labels]
        res = np.mean(vals)
    return res

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    print('pred: ', pred)
    
    expand = target.expand(-1, max(topk))
    print('expand: ', expand)
    
    correct = pred.eq(expand)
    print('correct: ', correct)
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

def sigmoid(x): 
    z = 1/(1 + np.exp(-x)) 
    return z

''' ROC CURVE '''
def plot_roc(y_pred, y_true, roc_name, plot=False):
    # given the test_ground_truth, and test_predictions 
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    roc_auc = auc(fpr, tpr)

    if plot: 
        plt.figure(dpi=100)
        plt.title(roc_name)
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
    return fpr, tpr, thresholds, roc_auc

# J = TP/(TP+FN) + TN/(TN+FP) - 1 = tpr - fpr
def choose_operating_point(fpr, tpr, thresholds):
    sens = 0
    spec = 0
    J = 0
    for _fpr, _tpr in zip(fpr, tpr):
        if _tpr - _fpr > J:
            sens = _tpr
            spec = 1-_fpr
            J = _tpr - _fpr
    return sens, spec

''' PRECISION-RECALL CURVE '''
def plot_pr(y_pred, y_true, pr_name, plot=False):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)
    # plot the precision-recall curves
    baseline = len(y_true[y_true==1]) / len(y_true)
    
    if plot: 
        plt.figure(dpi=20)
        plt.title(pr_name)
        plt.plot(recall, precision, 'b', label='AUC = %0.2f' % pr_auc)
        # axis labels
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [baseline, baseline],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        # show the plot
        plt.show()
    return precision, recall, thresholds

def evaluate(y_pred, y_true, cxr_labels, 
                   roc_name='Receiver Operating Characteristic', pr_name='Precision-Recall Curve', label_idx_map=None):
    
    '''
    We expect `y_pred` and `y_true` to be numpy arrays, both of shape (num_samples, num_classes)
    
    `y_pred` is a numpy array consisting of probability scores with all values in range 0-1. 
    
    `y_true` is a numpy array consisting of binary values representing if a class is present in
    the cxr. 
    
    This function provides all relevant evaluation information, ROC, AUROC, Sensitivity, Specificity, 
    PR-Curve, Precision, Recall for each class. 
    '''
    import warnings
    warnings.filterwarnings('ignore')

    num_classes = y_pred.shape[-1] # number of total labels
    
    dataframes = []
    for i in range(num_classes): 
#         print('{}.'.format(cxr_labels[i]))

        if label_idx_map is None: 
            y_pred_i = y_pred[:, i] # (num_samples,)
            y_true_i = y_true[:, i] # (num_samples,)
            
        else: 
            y_pred_i = y_pred[:, i] # (num_samples,)
            
            true_index = label_idx_map[cxr_labels[i]]
            y_true_i = y_true[:, true_index] # (num_samples,)
            
        cxr_label = cxr_labels[i]
        
        ''' ROC CURVE '''
        roc_name = cxr_label + ' ROC Curve'
        fpr, tpr, thresholds, roc_auc = plot_roc(y_pred_i, y_true_i, roc_name)
        
        sens, spec = choose_operating_point(fpr, tpr, thresholds)

        results = [[roc_auc]]
        df = pd.DataFrame(results, columns=[cxr_label+'_auc'])
        dataframes.append(df)
        
        ''' PRECISION-RECALL CURVE '''
        pr_name = cxr_label + ' Precision-Recall Curve'
        precision, recall, thresholds = plot_pr(y_pred_i, y_true_i, pr_name)
        
    dfs = pd.concat(dataframes, axis=1)
    return dfs

''' Bootstrap and Confidence Intervals '''
def compute_cis(data, confidence_level=0.05):
    """
    FUNCTION: compute_cis
    ------------------------------------------------------
    Given a Pandas dataframe of (n, labels), return another
    Pandas dataframe that is (3, labels). 
    
    Each row is lower bound, mean, upper bound of a confidence 
    interval with `confidence`. 
    
    Args: 
        * data - Pandas Dataframe, of shape (num_bootstrap_samples, num_labels)
        * confidence_level (optional) - confidence level of interval
        
    Returns: 
        * Pandas Dataframe, of shape (3, labels), representing mean, lower, upper
    """
    data_columns = list(data)
    intervals = []
    for i in data_columns: 
        series = data[i]
        sorted_perfs = series.sort_values()
        lower_index = int(confidence_level/2 * len(sorted_perfs)) - 1
        upper_index = int((1 - confidence_level/2) * len(sorted_perfs)) - 1
        lower = sorted_perfs.iloc[lower_index].round(4)
        upper = sorted_perfs.iloc[upper_index].round(4)
        mean = round(sorted_perfs.mean(), 4)
        interval = pd.DataFrame({i : [mean, lower, upper]})
        intervals.append(interval)
    intervals_df = pd.concat(intervals, axis=1)
    intervals_df.index = ['mean', 'lower', 'upper']
    return intervals_df
    
def bootstrap(y_pred, y_true, cxr_labels, n_samples=1000, label_idx_map=None): 
    '''
    This function will randomly sample with replacement 
    from y_pred and y_true then evaluate `n` times
    and obtain AUROC scores for each. 
    
    You can specify the number of samples that should be
    used with the `n_samples` parameter. 
    
    Confidence intervals will be generated from each 
    of the samples. 
    
    Note: 
    * n_total_labels >= n_cxr_labels
        `n_total_labels` is greater iff alternative labels are being tested
    '''
    np.random.seed(97)
    y_pred # (500, n_total_labels)
    y_true # (500, n_cxr_labels) 
    
    idx = np.arange(len(y_true))
    
    boot_stats = []
    for i in tqdm(range(n_samples)): 
        sample = resample(idx, replace=True, random_state=i)
        y_pred_sample = y_pred[sample]
        y_true_sample = y_true[sample]
        
        sample_stats = evaluate(y_pred_sample, y_true_sample, cxr_labels, label_idx_map=label_idx_map)
        boot_stats.append(sample_stats)

    boot_stats = pd.concat(boot_stats) # pandas array of evaluations for each sample
    return boot_stats, compute_cis(boot_stats)
