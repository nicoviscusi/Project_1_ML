# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------------------------------------


def load_data(file_path):
    """
    Load the csv. file into 3 arrays for the ids, predictions and measurements
    Args:
        file_path: path of the csv. file

    Returns:
       ids: ndarray of size (N, )
       predictions: ndarray of size (N, )
       data: ndarray of size (N, D)
    """

    df = load_dataframe(file_path)
    # calling of pandas.Dataframe.to_numpy() method to concert a dataframe into a array
    ids = df["Id"].to_numpy(dtype=np.int64)
    predictions = np.where(df["Prediction"] == "s", 1, 0)
    data = df.drop(["Id", "Prediction"], axis=1).to_numpy()
    return ids, predictions, data


# -----------------------------------------------------------------------------------------------------------


def load_dataframe(file_path):
    """
    Load the csv. file into 3 arrays for the ids, predictions and measurements
    Args:
        file_path: path of the csv. file

    Returns:
       whole_dataframe: dataframe of the whole csv. file
    """

    whole_dataframe = pd.read_csv(file_path)
    return whole_dataframe
