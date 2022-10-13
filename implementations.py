import numpy as np

def compute_MSE(y, tx, w):                                       
    """Calculate the loss using MSE

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        w: numpy array of shape=(D,). The vector of model parameters.

    Returns:
       L = the value of the loss (a scalar), corresponding to the input parameters w."""
    
    N = y.shape[0]
    e_vect = y - np.dot(tx,w)
    L = 1/(2*N)*np.dot(e_vect,e_vect) #expression of the MSE cost function with the error vector
    return L

def compute_gradient(y, tx, w):
    """Computes the gradient at w.
        
    Args:
        y: numpy array of shape=(N,)
        tx: numpy array of shape=(N,D)
        w: numpy array of shape=(D, ). The vector of model parameters.
        
    Returns:
        gradient = an numpy array of shape (D, ) (same shape as w), containing the gradient of the loss at w."""
    
    N = y.shape[0]
    e_vect = y - np.dot(tx,w) 
    gradient = -1/N*np.dot(np.transpose(tx),e_vect)
    return gradient

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient at w from just few examples n and their corresponding y_n labels.
        
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        w: numpy array of shape=(D, ). The vector of model parameters.
        
    Returns:
        A numpy array of shape (D, ) (same shape as w), containing the stochastic gradient of the loss at w."""
    
    N = y.shape[0]
    e_vect = y - np.dot(tx,w)
    stoch_grad = -1/N*np.dot(np.transpose(tx),e_vect)
    return stoch_grad
    
def sigmoid(t):
    """apply the sigmoid function on t."""
    
    return (1 + np.exp(-t))**(-1)

def compute_log_loss(y, tx, w):
    """compute the 'logistic' loss: negative log likelihood."""

    return np.sum(np.log(1+np.exp(tx.dot(w)))) - (y.T).dot(tx.dot(w))
    
def compute_ridge_log_loss(y, tx, w, lambda_):
    """compute the 'logistic' loss with l2-regularization: negative log likelihood."""
    
    return np.sum(np.log(1+np.exp(tx.dot(w)))) - (y.T).dot(tx.dot(w)) + lambda_*w.T.dot(w)

def compute_log_gradient(y, tx, w):
    """compute the gradient of negative log likelihood."""

    return tx.T.dot(sigmoid(tx.dot(w))-y)
    
def compute_ridge_log_gradient(y, tx, w, lambda_):
    """compute the gradient of negative log likelihood with l2-regularization."""

    return tx.T.dot(sigmoid(tx.dot(w))-y) + 2*lambda_*w

def least_squares_GD(y, tx, initial_w, max_iters, gamma):              
    """Gradient descent algorithm for MSE.
    
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
        
    Returns:
        loss: the loss value (scalar) for the best w found
        w: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of GD."""
    
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_gradient(y, tx, w)
        w = w - gamma*grad;
        loss = compute_MSE(y,tx,w)
    return loss, w

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm.
            
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize
        
    Returns:
        loss: the loss value (scalar) at w
        w: the model parameters as a numpy array of shape (D, )."""    
   
    w = initial_w
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # compute a stochastic gradient and loss
            stoch_grad = compute_stoch_gradient(y_batch, tx_batch, w)
            w = w- gamma*stoch_grad;
            loss = compute_MSE(y, tx, w)
            # store w and loss     
    return loss, w

def least_squares(y, tx):
    """Calculate the least squares solution.
       returns loss mse, and optimal weights.
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss (mse): scalar."""
    
    N = y.shape[0]
    gram = np.dot(np.transpose(tx), tx);    # Gram's matrix, for later use
    w_opt = np.linalg.solve(gram, np.dot(np.transpose(tx), y))
    e_vect = y - np.dot(tx, w_opt)
    loss = 1/(2*N)*np.dot(e_vect, e_vect)
    return loss, w


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    
    return loss, w


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """implement logistic regression."""

    w = initial_w
    loss = compute_log_loss(y, tx, w)
    for n_iter in range(max_iters):
        gradient = compute_log_gradient(y, tx, w)
        w = w - gamma*gradient
        loss = compute_log_loss(y, tx, w)
    return loss, w

def reg_logistic_regression(y, tx, lambda_ , initial_w, max_iters, gamma):
    """implement logistic regression with l2-regularization."""

    w = initial_w
    loss = compute_ridge_log_loss(y, tx, w, lambda_)
    for n_iter in range(max_iters):
        gradient = compute_ridge_log_gradient(y, tx, w, lambda_)
        w = w - gamma*gradient
        loss = compute_ridge_log_loss(y, tx, w)
    return loss, w
    
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING> """
    
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