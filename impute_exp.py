import os

import click
import pandas as pd
from fancyimpute import *
from models.OTimputer import *
from models.gain import *
from concurrent.futures import ThreadPoolExecutor
import tensorflow as tf
import torch
import random


def set_seed(seed: int):
    """
    Set the random seed for reproducibility across numpy, TensorFlow, and PyTorch.

    Parameters:
    seed (int): The seed value to set.
    """
    # Set the seed for Python's built-in random module
    random.seed(seed)

    # Set the seed for NumPy
    np.random.seed(seed)

    # Set the seed for TensorFlow
    tf.random.set_seed(seed)

    # Set the seed for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variables for other potential sources of randomness
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


def parallelize(func, iterable):
    """
    :param func: a function that takes a single element to apply some operation to it
    :param iterable: a list of elements
    :return: a list with the same size of iterable containing all the elements after applying func
    """
    output = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(func, element) for element in iterable]
    for future in futures:
        output.append(future.result())
    return output


def load_data(target_data, missing_data_dir):
    target = pd.read_csv(target_data)
    all_missing_data = []
    for missing_data_name in os.listdir(missing_data_dir):
        path = os.path.join(missing_data_dir, missing_data_name)
        missing_data = pd.read_csv(path)
        all_missing_data.append(missing_data)
    return target, all_missing_data


def load_model(k_nearest_value):
    gain = Gain(k_nearest_value)  # Missing Data Imputation using Generative Adversarial Nets
    bsi = BatchSinkhornImputation(
        batchsize=k_nearest_value)  # first algorithm in Missing Data Imputation using Optimal Transport
    simple_fill_mean = SimpleFill(fill_method='mean')
    simple_fill_median = SimpleFill(fill_method='median')
    simple_fill_random = SimpleFill(fill_method='random')
    knn = KNN(verbose=False, k=k_nearest_value)
    soft_impute = SoftImpute(verbose=False)
    iterative_impute = IterativeImputer(verbose=False)
    iterative_SVD = IterativeSVD(verbose=False)

    models = [gain, bsi, simple_fill_mean, simple_fill_median, simple_fill_random, knn, soft_impute, iterative_impute,
              iterative_SVD]

    return models


def run_single_model(missing_data):
    return lambda model: model.fit_transform(missing_data)


def impute_single_data(missing_data):
    k = missing_data.shape[1]
    missing_data = np.array(missing_data)
    models = load_model(k)
    results = parallelize(run_single_model(missing_data), models)
    return results


def print_results(results, all_missing_data, exp_name):
    models_name = ['gain', 'bsi',
                   'simple_fill_mean', 'simple_fill_median', 'simple_fill_random', 'knn', 'soft_impute',
                   'iterative_impute', 'iterative_SVD']

    os.makedirs(f'results_{exp_name}', exist_ok=True)
    for i in range(len(results)):
        k = all_missing_data[i].shape[1] - 1
        os.makedirs(f'results_{exp_name}/results_k_{k}', exist_ok=True)
        for j in range(len(results[i])):
            result = results[i][j]
            pd.DataFrame(result).to_csv(f'results_{exp_name}/results_k_{k}/{models_name[j]}.csv', index=False)


def get_loss(target, imputed_column, mask, model_name):
    MSE = ((target[mask] - imputed_column[mask]) ** 2).mean()
    MAE = np.abs(target[mask] - imputed_column[mask]).mean()
    std = np.std(target[mask] - imputed_column[mask])

    output = {'model': model_name, 'MSE': MSE, 'MAE': MAE, 'std': std}
    return output


def calculate_losses(results, target, target_column, all_missing_data, exp_name):
    models_name = ['gain', 'bsi',
                   'simple_fill_mean', 'simple_fill_median', 'simple_fill_random', 'knn', 'soft_impute',
                   'iterative_impute', 'iterative_SVD']

    target = np.array(target).reshape(-1)
    os.makedirs(f'losses_{exp_name}', exist_ok=True)
    # loop over dataframe
    for i in range(len(results)):
        missing_column = all_missing_data[i].columns.get_loc(target_column)
        mask = np.array(all_missing_data[i].isna())
        k = results[i][0].shape[1] - 1
        # loop over models
        models_losses = []
        for j in range(len(results[i])):
            result = results[i][j]
            imputed_column = result[:, missing_column]
            models_losses.append(get_loss(target, imputed_column, mask[:, missing_column], models_name[j]))

        df = pd.DataFrame(models_losses)
        df.to_csv(f'losses_{exp_name}/losses_k_{k}.csv', index=False)


@click.command()
@click.option('--target_data_path', help='path to csv file that contains the true date')
@click.option('--missing_data_dir', help='path to the directory that contains csv files for the missing data')
@click.option('--exp_name', help='path to the directory that contains csv files for the missing data')
def run_full_imputation(target_data_path, missing_data_dir, exp_name):
    target, all_missing_data = load_data(target_data_path, missing_data_dir)
    results = parallelize(impute_single_data, all_missing_data)
    target_column = target.columns[0]
    calculate_losses(results, target, target_column, all_missing_data, exp_name)
    print_results(results, all_missing_data, exp_name)


if __name__ == '__main__':
    set_seed(42)
    run_full_imputation()
