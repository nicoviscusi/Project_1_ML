import numpy as np

def compute_MSE(y, tx, w):
    """Calculate the loss using MSE."""
    # ***************************************************
    """Calculate the loss using MSE

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        w: numpy array of shape=(D,). The vector of model parameters.

    Returns:
       L = the value of the loss (a scalar), corresponding to the input parameters w.
    """
    N = y.shape[0]
    e_vect = y - np.dot(tx,w)
    
    L = 1/(2*N)*np.dot(e_vect,e_vect)   #expression of the MSE cost function with the error vector
    return L
    
    # ***************************************************

def compute_gradient(y, tx, w):    # is it obviously only for the MSE?
    """Compute the gradient."""
    # ***************************************************
    """Computes the gradient at w.
        
    Args:
        y: numpy array of shape=(N,)
        tx: numpy array of shape=(N,D)
        w: numpy array of shape=(D, ). The vector of model parameters.
        
    Returns:
        An numpy array of shape (D, ) (same shape as w), containing the gradient of the loss at w.
    """
    N = y.shape[0]
    e_vect = y - np.dot(tx,w)
    
    gradient = -1/N*np.dot(np.transpose(tx),e_vect)
    
    
    return gradient
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
    """Gradient descent algorithm.""" # for MSE
    # ***************************************************
    """
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
        
    Returns:
        loss: the loss value (scalar) for the best w found
        w: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of GD 
    """
    
    w = initial_w
    for n_iter in range(max_iters):
        
        grad = compute_gradient(y, tx, w)
        
        w = w - gamma*grad;
        loss = compute_MSE(y,tx,w)
        #print("GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
             # bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
       # print("GD norm:", np.linalg.norm(grad))

    if n_iter == max_iters-1:
        print("The method was stopped by the maximum number of iterations!\n")
        
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