import numpy as np

# -----------------------------------------------------------------------------------------------------------


def one_hot_encoding(data):
    for i in range(3):
        data = np.c_[data, np.where(data[:, 22] == i, 1, 0)]
    # data = np.c_[data, np.where(np.logical_and(data[:,22]==0,data[:,22]==1),1,0)]
    data = np.c_[data, np.where(data[:, 0] == -999, 1, 0)]
    data = np.delete(data, 22, 1)
    return data


# -----------------------------------------------------------------------------------------------------------


def handle_outliers(data):
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
    for i in range(data.shape[1] - 6):
        median = np.median(data[data[:, i] > -999, i])
        data[data[:, i] < -998, i] = median
    median = np.median(data[data[:, data.shape[1] - 6] != 0, data.shape[1] - 6])
    data[data[:, data.shape[1] - 6] == 0, data.shape[1] - 6] = median
    return data


# -----------------------------------------------------------------------------------------------------------


def polynomial_2(data):
    for i in range(29):
        for j in range(i,29):
            np.c_[data, data[:,i]*data[:,j]]
    return data


# -----------------------------------------------------------------------------------------------------------


def standardize(data):
    for i in range(29):
        data[:, i] = (data[:, i] - data[:, i].mean()) / data[:, i].var()
    return data


# -----------------------------------------------------------------------------------------------------------


def ones_concatenate(data):
    data = np.c_[data, np.ones(data.shape[0])]
    return data
