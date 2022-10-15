import numpy as np
import warnings
warnings.filterwarnings('ignore')

#-----------------------------------------------------------------------------------------------------------
"""Some useful functions"""
#-----------------------------------------------------------------------------------------------------------

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
    #expression of the MSE cost function with the error vector
    L = 1/(2*N)*np.dot(e_vect,e_vect)
    return L

#-----------------------------------------------------------------------------------------------------------

def compute_gradient(y, tx, w):
    """Computes the gradient at w.
        
    Args:
        y: numpy array of shape=(N,)
        tx: numpy array of shape=(N,D)
        w: numpy array of shape=(D, ). The vector of model parameters.
        
    Returns:
        gradient = an numpy array of shape (D, ) (same shape as w), containing the gradient of
        the loss at w."""
    
    N = y.shape[0]
    e_vect = y - np.dot(tx,w) 
    gradient = -1/N*np.dot(np.transpose(tx),e_vect)
    return gradient

#-----------------------------------------------------------------------------------------------------------

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient at w from just few examples n and their corresponding y_n labels.
        
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        w: numpy array of shape=(D, ). The vector of model parameters.
        
    Returns:
        A numpy array of shape (D, ) (same shape as w), containing the stochastic gradient of
        the loss at w."""
    
    N = y.shape[0]
    e_vect = y - np.dot(tx,w)
    stoch_grad = -1/N*np.dot(np.transpose(tx),e_vect)
    return stoch_grad

#-----------------------------------------------------------------------------------------------------------
    
def sigmoid(t):
    """apply the sigmoid function on t."""
    
    return (1 + np.exp(-t))**(-1)

#-----------------------------------------------------------------------------------------------------------

def compute_log_loss(y, tx, w):
    """compute the 'logistic' loss: negative log likelihood."""

    return np.sum(np.log(1+np.exp(tx.dot(w)))) - (y.T).dot(tx.dot(w))

#-----------------------------------------------------------------------------------------------------------
    
def compute_ridge_log_loss(y, tx, w, lambda_):
    """compute the 'logistic' loss with l2-regularization: negative log likelihood."""
    
    return np.sum(np.log(1+np.exp(tx.dot(w)))) - (y.T).dot(tx.dot(w)) + lambda_*w.T.dot(w)

#-----------------------------------------------------------------------------------------------------------

def compute_log_gradient(y, tx, w):
    """compute the gradient of negative log likelihood."""

    return tx.T.dot(sigmoid(tx.dot(w))-y)
    
#-----------------------------------------------------------------------------------------------------------
    
def compute_ridge_log_gradient(y, tx, w, lambda_):
    """compute the gradient of negative log likelihood with l2-regularization."""

    return tx.T.dot(sigmoid(tx.dot(w))-y) + 2*lambda_*w

#-----------------------------------------------------------------------------------------------------------

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with
    the randomness of the minibatches.
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

#-----------------------------------------------------------------------------------------------------------
"""Implemented methods"""
#-----------------------------------------------------------------------------------------------------------

def least_squares_GD(y, tx, initial_w, max_iters, gamma):              
    """Gradient descent algorithm for MSE.
    
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
        for each iteration of GD."""
    
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_gradient(y, tx, w)
        w = w - gamma*grad;
        loss = compute_MSE(y,tx,w)
    return loss, w

#-----------------------------------------------------------------------------------------------------------

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm.
            
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization)
        for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing
        the stochastic gradient
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

#-----------------------------------------------------------------------------------------------------------

def least_squares(y, tx):
    """Calculate the least squares solution through normal equations.
       returns loss mse, and optimal weights.
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss (mse): scalar."""
    from numpy.linalg import matrix_rank
    N = y.shape[0]
    if matrix_rank(tx) != tx.shape[1]:
        print("X matrix is rank-deficient!")
        
    gram = np.dot(np.transpose(tx), tx);    # Gram's matrix, for later use
    print(gram)
    print(np.dot(np.transpose(tx), y))
    
    w_opt = np.linalg.solve(gram, np.dot(np.transpose(tx), y))
    e_vect = y - np.dot(tx, w_opt)
    loss = 1/(2*N)*np.dot(e_vect, e_vect)
    return loss, w

#-----------------------------------------------------------------------------------------------------------

def ridge_regression(y, tx, lambda_):
    """Calculate the ridge regression.
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar, hyperparameter.
    
    Returns:
        w_ridge: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar, MSE loss function computed with y, tx and w."""
    
    N = y.shape[0]
    D = tx.shape[1]
    
    # We will first create the Gram Matrix
    gram = np.dot(np.transpose(tx), tx)
    
    # Now we asssemble the rest of the linear system
    # First, a recomputation of lambda_ is needed
    lambdap = 2 * N * lambda_
    lmatrix = lambdap * np.identity(D)
    A = np.add(gram, lmatrix)
    b = np.dot(np.transpose(tx), y)
    
    
    # Now we can solve the linear system
    w_ridge = np.linalg.solve(A, b)
    
    
    return w_ridge

#-----------------------------------------------------------------------------------------------------------

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """implement logistic regression."""

    losses = []
    threshold = 1e-18
    w = initial_w
    loss = compute_log_loss(y, tx, w)
    for n_iter in range(max_iters):
        gradient = compute_log_gradient(y, tx, w)
        w = w - gamma*gradient
        loss = compute_log_loss(y, tx, w)
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return loss, w

#-----------------------------------------------------------------------------------------------------------

def reg_logistic_regression(y, tx, lambda_ , initial_w, max_iters, gamma):
    """implement logistic regression with l2-regularization."""

    losses = []
    threshold = 1e-18
    w = initial_w
    loss = compute_ridge_log_loss(y, tx, w, lambda_)
    for n_iter in range(max_iters):
        gradient = compute_ridge_log_gradient(y, tx, w, lambda_)
        w = w - gamma*gradient
        loss = compute_ridge_log_loss(y, tx, w,lambda_)
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return loss, w

#-----------------------------------------------------------------------------------------------------------
"""Cross-validation algorithm"""
#-----------------------------------------------------------------------------------------------------------

def split_data(data, predictions, ratio, seed):
    """split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing."""
    training_y = []
    training_x = []
    test_y = []
    test_x = []
    np.random.seed(seed)
    for i in range(0,len(predictions)):
        j = np.random.uniform(0,1)
        if j < ratio:
            training_y.append(predictions[i])
            training_x.append(data[i,:])
        else:
            test_y.append(predictions[i])
            test_x.append(data[i,:])
    return np.array(training_y), np.array(training_x), np.array(test_y), np.array(test_x)

#-----------------------------------------------------------------------------------------------------------

def build_k_indices(predictions, k_fold, seed):
    """build k indices for k-fold."""
    
    num_row = predictions.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

#-----------------------------------------------------------------------------------------------------------

def cross_validation(predictions, data, k_indices, k):
    """return kth test and training sets."""

    k_fold = k_indices.shape[0]
    test_predictions = predictions[k_indices[k]]
    test_data = data[k_indices[k],:]
    training_predictions = np.delete(predictions,k_indices[k],axis=0)
    training_data = np.delete(data,k_indices[k],axis=0)
    return test_predictions, test_data, training_predictions, training_data

