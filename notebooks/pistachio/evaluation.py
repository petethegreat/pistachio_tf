'''
evaluation.py
utils for evaluating models
'''
import sys
from typing import List, Dict, Callable, Tuple

from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import scipy
import statsmodels
from statsmodels.stats.proportion import proportion_confint


def plot_metric(history_df: pd.DataFrame, metric_name:str):
    '''plot metric vs epoch'''
    fig = plt.figure()
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    # train data value 
    if metric_name == 'learning_rate':
        ax.plot(history_df.index, history_df[metric_name], color=sns.xkcd_rgb['merlot'], label=metric_name)
    else:    
        ax.plot(history_df.index, history_df[metric_name], color=sns.xkcd_rgb['merlot'], label=f'train_{metric_name}')
        ax.plot(history_df.index, history_df[f'val_{metric_name}'], color=sns.xkcd_rgb['blurple'], label=f'val_{metric_name}')
    ax.legend()
    ax.set_title(f'{metric_name} vs epoch')
    return fig, ax

def get_roc_results(predicted_probs: List[float], actual_classes: List[float]) -> Tuple[List[float],List[float],List[float]]:
    """get roc curve definition

    Args:
        predicted_probs (List[float]): predicted probabilities
        actual_classes (List[float]): actual binary labels

    Returns:
        Tuple[List,List,List]: fpr, tpr, thresholds
    """
    fpr, tpr, thresholds = roc_curve(actual_classes, predicted_probs)
    if thresholds[0] == float('inf'):
        thresholds[0] = sys.float_info.max

    return fpr, tpr, thresholds

#################################################################

def plot_roc_curve(fpr, tpr, thresholds, title: str="ROC curve", xlabel='False Positive Rate', ylabel: str='True Positive Rate') -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """_summary_

    Args:
        fpr (_type_): _description_
        tpr (_type_): _description_
        thresholds (_type_): _description_
        title (str, optional): _description_. Defaults to "ROC curve".
        xlabel (str, optional): _description_. Defaults to 'False Positive Rate'.
        ylabel (str, optional): _description_. Defaults to 'True Positive Rate'.

    Returns:
        Tuple[mpl.Figure, mpl.Axis]: _description_
    """
    fig = plt.figure()
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    ax.plot(fpr, tpr, color=sns.xkcd_rgb['blurple'], label='roc curve')
    ax.plot([0.0, 1.0],[0.0, 1.0], color=sns.xkcd_rgb['merlot'], linestyle='--', label='random')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    # fig.show()
    return fig, ax
#################################################################
def get_confusion_matrix(predicted_classes, actual_classes, normalise=None):
    """get confusion matrix
    computes confusion matrix for binary classification

    Args:
        predicted_classes (_type_): _description_
        actual_classes (_type_): _description_
        normalise (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    matrix = confusion_matrix(actual_classes,predicted_classes, normalize=normalise)
    return matrix
#################################################################

def make_confusion_matrix_plot(
    predicted_classes,
    actual_classes,
    title:str = 'confusion matrix',
    xlabel: str='predicted class',
    ylabel: str='actual class',
    class_names: List[str] = None,
    normalise:str=None
    ):
    """ generate confusion matrix plot"""
    fig = plt.figure()
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    ax.grid(False)
    # cmap = sns.color_palette("magma_r", as_cmap=True)
    cmap = sns.light_palette("indigo", as_cmap=True)

    # cmap = 'viridis'

    matrix = confusion_matrix(actual_classes,predicted_classes, normalize=normalise)
    ax.imshow(matrix, cmap=cmap)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(i,j,f'{matrix[i,j]}')
    # ax.plot([0.0, 1.0],[0.0, 1.0], color=sns.xkcd_rgb['merlot'], linestyle='--', label='random')
    labels = class_names if class_names else ['0','1']

    ax.set_xlim([-0.5, matrix.shape[0]- 0.5])
    ax.set_ylim([matrix.shape[0]- 0.5, -0.5])
    ax.set_xticks(np.arange(matrix.shape[0]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # ax.legend()
    # fig.show()
    return fig, ax
#################################################################
def make_precision_recall_plot(predicted_probs, actual_classes, title: str="Precision-Recall Curve", xlabel='Recall',ylabel: str='Precision',
                              positive_rate:float=None):
    """make a roc curve"""
    precision, recall, _ = precision_recall_curve(actual_classes, predicted_probs)
    fig = plt.figure()
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    classifier_average_precision = average_precision_score(actual_classes, predicted_probs)
    ax.plot(recall, precision, color=sns.xkcd_rgb['blurple'], label=f'precision recall curve (average precision = {classifier_average_precision:0.3f}')
    if positive_rate:
        ax.plot([0.0, 1.0],[positive_rate, positive_rate], color=sns.xkcd_rgb['merlot'], linestyle='--', label=f'positive response rate = {positive_rate:0.3f}')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend()
    # fig.show()
    return fig, ax
###########################################################

def make_prob_calibration_plot(predicted_probs, actual_classes, n_bins: int=20, alpha: float = 0.05, title:str = 'probability calibration'):
    """bin records, check that proportion of labels in each bin matches mean probability of that bin"""
    bins = pd.qcut(predicted_probs, n_bins, labels=False)
    df = pd.DataFrame({'probability':predicted_probs, 'class':actual_classes, 'bin':bins})
    df = df.sort_values(by='probability',ascending=False).reset_index(drop=True)
    agged = df.groupby('bin').agg(
        pred_prob=pd.NamedAgg('probability','mean'),
        pred_std=pd.NamedAgg('probability','std'),
        class_prob=pd.NamedAgg('class','mean'),
        class_sum=pd.NamedAgg('class','sum'),
        bin_size=pd.NamedAgg('class','count')
    )
    act_err_low, act_err_high = proportion_confint(agged.class_sum, agged.bin_size, method='wilson', alpha = alpha)
    z_low = scipy.stats.norm.ppf(alpha/2)
    z_high = scipy.stats.norm.ppf(1.0 - alpha/2)
    agged['pred_low'] =  -z_low*agged['pred_std']/np.sqrt(agged['bin_size']) # agged['pred_prob'] +
    agged['pred_high'] = z_high*agged['pred_std']/np.sqrt(agged['bin_size']) #+ agged['pred_prob'] +
    agged['actual_error_high'] = np.maximum(act_err_high  - agged.class_prob,0)
    agged['actual_error_low'] =  np.maximum(agged.class_prob - act_err_low,0)
    agged.loc[np.abs(agged.actual_error_high) < 1e-10, 'actual_error_high'] = 0


    # print(agged)
    # print(agged)
    fig = plt.figure()
    ax = fig.add_axes([0.1,0.1, 0.8, 0.8])
    ax.errorbar(
        agged.pred_prob,
        agged.class_prob,
        yerr=[agged.actual_error_low, agged.actual_error_high],
        xerr=[agged.pred_low,agged.pred_high],
        fmt='.',
        color=sns.xkcd_rgb['blurple'])
    ax.plot([0.0,1.0],[0.0,1.0],'--',label='ideal', color=sns.xkcd_rgb['dark blue'])
    ax.set_title(title)
    ax.set_xlabel('predicted probability')
    ax.set_ylabel('observed probabiilty')
    return fig, ax