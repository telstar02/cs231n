from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = X.shape[0]

    for i in range(N):
      f = X[i] @ W # (1,C)
      # numeric stability
      f -= np.max(f) # minus the max of y
      exp_y_normal = np.exp(f) / np.sum(np.exp(f)) # (1,C)
      loss += -np.log(exp_y_normal[y[i]])

      # backward
      dloss_f = exp_y_normal.copy() # (1,C)
      dloss_f[y[i]] -= 1
      #print(X[i].shape,dloss_f.T.shape)
      dW += X[i].reshape(-1,1) @ dloss_f.reshape(1,-1) # (D,1)@(1,C)

    dW /= N
    dW += 2*reg*W
    

    loss /= N
    loss += reg * np.sum(W*W)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = X.shape[0]
    
    f = X @ W   # f@(N,C)
    f -= np.max(f, axis=1).reshape(-1,1) # per sample stability
    exp_y_normal = np.exp(f) / np.sum(np.exp(f), axis=1).reshape(-1,1) # (N,C)/(N,1) = (N,C)
    loss = np.sum(-np.log(exp_y_normal[np.arange(N), [y]])) / N   # sum(-log(N,1)) /N

    # backward
    dloss_f = exp_y_normal.copy()  # (N,C)
    dloss_f[np.arange(N),y] -= 1  # label dloss_fyi
    dW = X.T @ dloss_f  # (D,N)@(N,C) = (D,C)

    dW /= N

    # regulation
    loss += reg * np.sum(W*W)
    dW += 2*reg*W
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
