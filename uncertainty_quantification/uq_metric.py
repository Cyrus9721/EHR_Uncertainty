from torch_uncertainty.metrics.classification import BrierScore, CategoricalNLL
from torch_uncertainty.metrics.classification.adaptive_calibration_error import BinaryAdaptiveCalibrationError

from typing import List, Tuple
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os

results_dir = '../uncertainty_quantification/single_model_results/'
single_task_results_dir = '../uncertainty_quantification/single_task_results/'
labeling_functions_path = '../uncertainty_quantification/labeling_function_names.txt'

def read_labeling_functions(path: str) -> List[str]:
    with open(path, 'r') as f:
        labeling_functions = [line.rstrip('\n') for line in f]
    return labeling_functions

def convert_string_to_probs(s):
    s = s.strip('[]')
    numbers = s.split()
    pro_list = [float(n) for n in numbers]
    return pro_list

def read_results(results_dir: str, task_name: str):
    task_results_file = os.path.join(results_dir, task_name + '_results.csv')
    df_results = pd.read_csv(task_results_file)
    gt = df_results['0'].values
    pred = df_results['1'].values
    predicted_prob= df_results['2'].values
    binary_prob = [convert_string_to_probs(s) for s in df_results['3'].values]
    return gt, pred, predicted_prob, binary_prob

def calcualte_brier_score(gt, binary_prob):
    metric_brier = BrierScore(num_classes=2, top_class=False)
    metric_brier.update(torch.tensor(binary_prob), torch.tensor(gt))
    brierScore = metric_brier.compute()
    return np.round(brierScore.item(), 3)

# calculate the categorical nll
def calcualte_categorical_nll(gt, binary_prob):
    metric_categorical_nll = CategoricalNLL()
    metric_categorical_nll.update(torch.tensor(binary_prob), torch.tensor(gt))
    categorical_nll = metric_categorical_nll.compute()
    return np.round(categorical_nll.item(), 3)

def calculate_calibration_error(ge, pred):
    metric_calibration = BinaryAdaptiveCalibrationError(n_bins=10, norm='l1')
    metric_calibration.update(torch.tensor(predicted_prob), torch.tensor(gt))
    calibration_error = metric_calibration.compute()
    return np.round(calibration_error.item(), 3)