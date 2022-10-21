import os
import math
import logging

import pandas as pd
import numpy as np

import keras.backend as K


# Formats Position
def format_position(price): return (
    '-$' if price < 0 else '+$') + f'{abs(price):.2f}'


# Formats Currency
def format_currency(price): return f'${abs(price):.2f}'


def show_train_result(result, val_position, initial_offset):
    """ Displays training results
    """
    if val_position == initial_offset or val_position == 0.0:
        logging.info(
            f'Episode {result[0]}/{result[1]} - Train Position: {format_position(result[2])}  Val Position: USELESS  Train Loss: {result[3]:.4f}')
    else:
        logging.info(
            f'Episode {result[0]}/{result[1]} - Train Position: {format_position(result[2])}  Val Position: {format_position(val_position)}  Train Loss: {result[3]:.4f})')


def show_eval_result(model_name, profit, initial_offset):
    """ Displays eval results
    """
    if profit == initial_offset or profit == 0.0:
        logging.info(f'{model_name}: USELESS\n')
    else:
        logging.info(f'{model_name}: {format_position(profit)}\n')


def get_stock_data(stock_file):
    """Reads stock data from csv file
    """
    df = pd.read_csv(stock_file)
    return list(df['Adj Close'])


def switch_k_backend_device():
    """ Switches `keras` backend from GPU to CPU if required.

    Faster computation on CPU (if using tensorflow-gpu).
    """
    if K.backend() == "tensorflow":
        logging.debug("switching to TensorFlow for CPU")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
