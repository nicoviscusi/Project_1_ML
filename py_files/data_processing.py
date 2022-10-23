import numpy as np

# -----------------------------------------------------------------------------------------------------------


def one_hot_encoding(data):
    """
    performs one-hot encoding on the categorical variable PRI_JET_NUM that stores either 0,1,2,3
    (22th. variable in data array) by creating 3 new variables each containing 1 for samples
    that have PRI_JET_NUM set to 0,1,2 respectively and 0 for the rest.
    performs one-hot encoding on the first variable by creating another variable that contains 1 for samples
    that are undefined (-999) and 0 for the rest

    Args:
        data: ndarray

    Returns:
        data: ndarray after preprocessing
    """
    for i in range(3):
        data = np.c_[data, np.where(data[:, 22] == i, 1, 0)]
    data = np.c_[data, np.where(data[:, 0] == -999, 1, 0)]
    # deletion of the 22th. variable (PRI_JET_NUM) after performing one-hot encoding
    data = np.delete(data, 22, 1)
    return data


# -----------------------------------------------------------------------------------------------------------


def handle_outliers(data):
    """
    Fore each feature, this function is looking for outliers.
    Outliers in the distribution are defined as such:
    - We first compute the interquartile range defined as IQR = (3rd. quartile - 1st. quartile)
    - We then define the upper and lower threshold:
      - Lower threshold: 1st. quartile - 3*(IQR)
      - Upper threshold: 3rd. quartile + 3*(IQR)
    - Values are defined as outliers if they are above the upper or below the lower threshold
    After identifying the outlier values, we replace them with the median of the distribution (without
    the undefined values)
    Finally, a vector containing 1 for samples that were defined as outliers and 0 for the rest is added to the
    data, because we assume that being an outlier might be relevant for determining whether we get a signal or
    background noise in the classification task

    Args:
        data: ndarray

    Returns:
        data: ndarray after preprocessing
    """

    for i in range(data.shape[1] - 5):
        data_test = data[data[:, i] > -999, i]
        median = np.median(data_test)
        IQR = np.quantile(data_test, 0.75) - np.quantile(data_test, 0.25)
        lower_bridge = np.quantile(data_test, 0.25) - (IQR * 3)
        upper_bridge = np.quantile(data_test, 0.75) + (IQR * 3)
        cond = np.where(
            np.logical_and(
                np.logical_or(data[:, i] < lower_bridge, data[:, i] > upper_bridge),
                data[:, i] > -999,
            ),
            1,
            0,
        )
        if np.any(cond == 1):
            data = np.c_[data, cond]
        data[data[:, i] > -999, i] = np.where(
            np.logical_or(data_test < lower_bridge, data_test > upper_bridge),
            median,
            data_test,
        )
    return data


# -----------------------------------------------------------------------------------------------------------


def handle_undefined_values(data):
    """
    For each feature containing undefined values (-999), the function replaces the undefined values with
    the median of the distribution.
    For variable TOTAL_JET_PT (last variable of the data before preprocessing), we replace the values equal
    to 0 with the median of the distribution

    Args:
        data: ndarray

    Returns:
        data: ndarray after preprocessing
    """

    for i in range(data.shape[1] - 6):
        median = np.median(data[data[:, i] > -999, i])
        data[data[:, i] < -998, i] = median
    median = np.median(data[data[:, data.shape[1] - 6] != 0, data.shape[1] - 6])
    data[data[:, data.shape[1] - 6] == 0, data.shape[1] - 6] = median
    return data


# -----------------------------------------------------------------------------------------------------------


def polynomial_2(data):
    """
    Polynomial feature expansion of degree 2
    For each feature of the data except the categorical variables, the function will compute its square only.
    We didn't multiply features with each other, because it is computationally expensive

    Args:
        data: ndarray

    Returns:
        data: ndarray after preprocessing
    """

    # in the original data before processing we had 30 features, but we deleted the PRI_JET_NUM (22th.)
    # feature during the one-hot encoding step, explaining why we iterate over the first 29 features
    for i in range(29):
        data = np.c_[data, data[:, i] ** 2]
    return data


# -----------------------------------------------------------------------------------------------------------


def standardize(data):
    """
    Normalization of the data by subtracting each sample value of a non-categoraical feature with its mean
    and diving by the variance

    Args:
        data: ndarray

    Returns:
        data: ndarray after preprocessing
    """

    # in the original data before processing we had 30 features, but we deleted the PRI_JET_NUM (22th.)
    # feature during the one-hot encoding step, explaining we we iterate over the first 29 features. We
    # also standardize the squares of the features, explaining why we iterate a second time from the 49th.
    # features, because the other ones are categorical (see one_hot_encoding and handle_outliers)
    for i in range(29):
        data[:, i] = (data[:, i] - data[:, i].mean()) / data[:, i].var()
    for i in range(48, data.shape[1]):
        data[:, i] = (data[:, i] - data[:, i].mean()) / data[:, i].var()
    return data


# -----------------------------------------------------------------------------------------------------------


def ones_concatenate(data):
    """
    Add a column vector of size (N,) with 1 as values (shift term)

    Args:
        data: ndarray

    Returns:
        data: ndarray after preprocessing
    """
    data = np.c_[data, np.ones(data.shape[0])]
    return data
