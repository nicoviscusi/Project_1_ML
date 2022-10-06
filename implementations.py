import numpy as np

def compute_MSE(y, tx, w):
    """Calculate the loss using MAE."""
    # ***************************************************
    
    # ***************************************************

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    # ***************************************************
    
    # ***************************************************

def compute_stoch_gradient(y, tx, w):
    """Compute the stochastic gradient."""
    # ***************************************************
    
    # ***************************************************
    
def sigmoid(t):
    """apply the sigmoid function on t."""
    # ***************************************************
    return (1 + np.exp(-t))**(-1)
    # ***************************************************

def compute_log_loss(y, tx, w):
    """compute the 'logistic' loss: negative log likelihood."""
    # ***************************************************
    return np.sum(np.log(1+np.exp(tx.dot(w)))) - (y.T).dot(tx.dot(w))
    # ***************************************************
    
def compute_ridge_log_loss(y, tx, w, lambda_):
    """compute the 'logistic' loss with l2-regularization: negative log likelihood."""
    # ***************************************************
    return np.sum(np.log(1+np.exp(tx.dot(w)))) - (y.T).dot(tx.dot(w)) + lambda_*w.T.dot(w)
    # ***************************************************
    
def compute_log_gradient(y, tx, w):
    """compute the gradient of negative log likelihood."""
    # ***************************************************
    return tx.T.dot(sigmoid(tx.dot(w))-y)
    # ***************************************************
    
def compute_ridge_log_gradient(y, tx, w, lambda_):
    """compute the gradient of negative log likelihood with l2-regularization."""
    # ***************************************************
    return tx.T.dot(sigmoid(tx.dot(w))-y) + 2*lambda_*w
    # ***************************************************

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # ***************************************************
    
    return loss, w
    # ***************************************************

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    # ***************************************************
    
    return loss, w
    # ***************************************************

def least_squares(y, tx):
    """calculate the least squares solution."""
    # ***************************************************
    
    return loss, w
    # ***************************************************

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    # ***************************************************
    
    return loss, w
    # ***************************************************

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """implement logistic regression."""
    # ***************************************************
    w = initial_w
    loss = compute_log_loss(y, tx, w)
    for n_iter in range(max_iters):
        gradient = compute_log_gradient(y, tx, w)
        w = w - gamma*gradient
        loss = compute_log_loss(y, tx, w)
    return loss, w
    # ***************************************************

def reg_logistic_regression(y, tx, lambda_ , initial_w, max_iters, gamma):
    """implement logistic regression with l2-regularization."""
    # ***************************************************
    w = initial_w
    loss = compute_ridge_log_loss(y, tx, w, lambda_)
    for n_iter in range(max_iters):
        gradient = compute_ridge_log_gradient(y, tx, w, lambda_)
        w = w - gamma*gradient
        loss = compute_ridge_log_loss(y, tx, w)
    return loss, w
    # ***************************************************