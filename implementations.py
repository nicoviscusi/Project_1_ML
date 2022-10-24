import numpy as np

# -----------------------------------------------------------------------------------------------------------
"""Some useful functions"""
# -----------------------------------------------------------------------------------------------------------


def compute_MSE(y, tx, w):
    """
    Calculate the loss using MSE
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        w: numpy array of shape=(D,). The vector of model parameters.
    Returns:
       the value of the loss (a scalar), corresponding to the input parameters w.
    """

    # error vector y - y', where y' = Xw (predictions)
    e = y - tx.dot(w)
    return 1 / 2 * np.mean(e**2)


# -----------------------------------------------------------------------------------------------------------


def compute_gradient(y, tx, w):
    """
    Computes the gradient at w.
    Args:
        y: numpy array of shape=(N,)
        tx: numpy array of shape=(N,D)
        w: numpy array of shape=(D, ). The vector of model parameters.
    Returns:
        gradient = an numpy array of shape (D, ) (same shape as w), containing the gradient of
        the loss at w.
    """

    N = y.shape[0]
    # error vector y - y', where y' = Xw (predictions)
    e_vect = y - tx.dot(w)
    gradient = -1 / N * tx.T.dot(e_vect)
    return gradient


# -----------------------------------------------------------------------------------------------------------


def compute_stoch_gradient(y, tx, w):
    """
    Compute a stochastic gradient at w from just few examples n and their corresponding y_n labels.
    Args:
        y: scalar
        tx: numpy array of shape=(D, )
        w: numpy array of shape=(D, ). The vector of model parameters.
    Returns:
        stoch_gradient = a numpy array of shape (D, ) (same shape as w), containing the stochastic gradient of
        the loss at w.
    """

    # error vector y - y', where y' = Xw (predictions)
    e_vect = y - tx.dot(w)
    stoch_gradient = -tx.T.dot(e_vect)
    return stoch_gradient


# -----------------------------------------------------------------------------------------------------------


def sigmoid(t):
    """
    apply the sigmoid function on t.
    Args:
        t: scalar
    Returns:
       The value of the sigmoid function defined at t.
    """

    return (1 + np.exp(-t)) ** (-1)


# -----------------------------------------------------------------------------------------------------------


def compute_log_loss(y, tx, w):
    """
    compute the 'logistic' loss: negative log likelihood.
    Args:
        y: numpy array of shape=(N,)
        tx: numpy array of shape=(N,D)
        w: numpy array of shape=(D,). The vector of model parameters.
    Returns:
       loss: a scalar, the value of negative log likelihood for logistic regression.
    """

    N = y.shape[0]
    pred = sigmoid(tx.dot(w))
    loss = -1 / N * (y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred)))
    return np.squeeze(loss)


# -----------------------------------------------------------------------------------------------------------


def compute_ridge_log_loss(y, tx, w, lambda_):
    """
    compute the 'logistic' loss with l2-regularization: negative log likelihood.
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        w: numpy array of shape=(D,). The vector of model parameters.
        lambda_: L2-regularization constant
    Returns:
       the value of the loss (a scalar), the value of negative log likelihood for logistic regression
       with regularization term lambda_*w.T.dot(w)
    """

    N = y.shape[0]
    pred = sigmoid(tx.dot(w))
    loss = -1 / N * (y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred)))
    return np.squeeze(loss + lambda_ * w.T.dot(w))


# -----------------------------------------------------------------------------------------------------------


def compute_log_gradient(y, tx, w):
    """
    compute the gradient of negative log likelihood.
    Args:
        y: numpy array of shape=(N,)
        tx: numpy array of shape=(N,D)
        w: numpy array of shape=(D,). The vector of model parameters.
    Returns:
        the gradient, a numpy array of shape (D,).
    """

    N = y.shape[0]
    return 1 / N * tx.T.dot(sigmoid(tx.dot(w)) - y)


# -----------------------------------------------------------------------------------------------------------


def compute_ridge_log_gradient(y, tx, w, lambda_):
    """
    compute the gradient of negative log likelihood with l2-regularization.
    Args:
        y: numpy array of shape=(N,)
        tx: numpy array of shape=(N,D)
        w: numpy array of shape=(D,). The vector of model parameters.
        lambda_: L2-regularization constant
    Returns:
        the gradient, a numpy array of shape (D,).
    """

    N = y.shape[0]
    return 1 / N * tx.T.dot(sigmoid(tx.dot(w)) - y) + 2 * lambda_ * w


# -----------------------------------------------------------------------------------------------------------


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with
    the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """

    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


# -----------------------------------------------------------------------------------------------------------
"""Implemented methods"""
# -----------------------------------------------------------------------------------------------------------


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    Gradient descent algorithm for MSE.
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization)
        for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
    Returns:
        loss: the loss value (scalar) for the best w found
        w: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ),
        for each iteration of GD.
    """

    w = initial_w
    loss = compute_MSE(y, tx, w)
    for n_iter in range(max_iters):
        grad = compute_gradient(y, tx, w)
        w = w - gamma * grad
        loss = compute_MSE(y, tx, w)
    return w, loss


# -----------------------------------------------------------------------------------------------------------


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """
    Stochastic gradient descent algorithm.
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization)
        for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize
    Returns:
        loss: the loss value (scalar) at w
        w: the model parameters as a numpy array of shape (D, ).
    """

    w = initial_w
    loss = compute_MSE(y, tx, w)
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1):
            # compute a stochastic gradient and loss
            stoch_grad = compute_stoch_gradient(y_batch, tx_batch, w)
            w = w - gamma * stoch_grad
            loss = compute_MSE(y, tx, w)
            # store w and loss
    return w, loss


# -----------------------------------------------------------------------------------------------------------


def least_squares(y, tx):
    """
    Calculate the least squares solution through normal equations.
       returns loss mse, and optimal weights.
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss (mse): scalar.
    """

    gram = tx.T.dot(tx)  # Gram's matrix, for later use
    a = gram
    b = tx.T.dot(y)
    w_opt = np.linalg.lstsq(a, b, rcond=None)[0]
    loss = compute_MSE(y, tx, w_opt)
    return w_opt, loss


# -----------------------------------------------------------------------------------------------------------


def ridge_regression(y, tx, lambda_):
    """
    Calculate the ridge regression.
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar, hyperparameter.
    Returns:
        w_ridge: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar, MSE loss function computed with y, tx and w.
    """

    N = y.shape[0]
    D = tx.shape[1]

    # We will first create the Gram Matrix
    gram = tx.T.dot(tx)

    # Now we asssemble the rest of the linear system
    # First, a recomputation of lambda_ is needed
    lambdap = 2 * N * lambda_
    lmatrix = lambdap * np.identity(D)
    a = gram + lmatrix
    b = tx.T.dot(y)

    # Now we can solve the linear system
    w_ridge = np.linalg.lstsq(a, b, rcond=None)[0]
    loss = compute_MSE(y, tx, w_ridge)

    return w_ridge, loss


# -----------------------------------------------------------------------------------------------------------


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Implement logistic regression.
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        initial_w: numpy array of shape (D,), D is the number of features.
        max_iters: scalar
        gamma: scalar (learning rate)
    Returns:
        w_ridge: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar, negative log likelihood function computed with y, tx and w.
    """

    w = initial_w
    loss = compute_log_loss(y, tx, w)
    for n_iter in range(max_iters):
        gradient = compute_log_gradient(y, tx, w)
        w = w - gamma * gradient
        loss = compute_log_loss(y, tx, w)
    return w, loss


# -----------------------------------------------------------------------------------------------------------


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Implement l2-regularized logistic regression.
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar, hyperparameter (L2-regularization constant)
        initial_w: numpy array of shape (D,), D is the number of features.
        max_iters: scalar
        gamma: scalar (learning rate)
    Returns:
        w_ridge: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar, negative log likelihood function computed with y, tx and w.
    """

    w = initial_w
    loss = compute_ridge_log_loss(y, tx, w, lambda_)
    for n_iter in range(max_iters):
        gradient = compute_ridge_log_gradient(y, tx, w, lambda_)
        w = w - gamma * gradient
        loss = compute_ridge_log_loss(y, tx, w, lambda_)
    return w, loss


# -----------------------------------------------------------------------------------------------------------
"""Cross-validation algorithm"""
# -----------------------------------------------------------------------------------------------------------


def split_data(x, y, ratio, seed):
    """
    split the dataset based on the split ratio. If ratio is 0.8
    you will have 80% of your data set dedicated to training
    and the rest dedicated to testing. In the end, you will get your training
    and test predictions and data.
    Args:
        x: data, a numpy array of shape (N,D)
        y: predictions, a numpy array of shape (N,)
        ratio: scalar (between 0 and 1)
        seed: scalar to fix the seed and have reproducible results
    Returns:
        x_tr: a numpy array of shape (K,D)
        x_te: a numpy array of shape (N-K,D)
        y_tr: a numpy array of shape (K,)
        y_te: a numpy array of shape (N-K,)
    """

    # set seed
    np.random.seed(seed)
    # generate random indices
    num_row = len(y)
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_tr = indices[:index_split]
    index_te = indices[index_split:]
    # create split
    x_tr = x[index_tr, :]
    x_te = x[index_te, :]
    y_tr = y[index_tr]
    y_te = y[index_te]
    return y_tr, x_tr, y_te, x_te


# -----------------------------------------------------------------------------------------------------------


def build_k_indices(predictions, k_fold, seed):
    """
    build k indices for k-fold.
    Args:
        predictions: a numpy array of shape (N,)
        k_fold: scalar, indicating in how many subsets diving the data for cross-validation
        seed: scalar to fix the seed and have reproducible results
    Returns:
       k_indices: an index array
    """

    num_row = predictions.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]

    return np.array(k_indices)


# -----------------------------------------------------------------------------------------------------------


def cross_validation(predictions, data, k_indices, k):
    """
    return kth test and training sets for cross-validation.
    Args:
        predictions: a numpy array of shape (N,)
        data: a numpy array of shape (N,D)
        k_indices: an index array
        k: index of desiredindices set
    Returns:
       test_predictions: a numpy array of shape (K,)
       test_data: a numpy array of shape (K,D)
       train_predictions: a numpy array of shape (N-K,)
       train_data: a numpy array of shape (N-K,)
    """

    # create indices
    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)

    # split the data
    test_predictions = predictions[te_indice]
    train_predictions = predictions[tr_indice]
    test_data = data[te_indice]
    train_data = data[tr_indice]

    return test_predictions, test_data, train_predictions, train_data
