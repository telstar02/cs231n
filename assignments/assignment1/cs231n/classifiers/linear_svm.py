from builtins import range
from matplotlib.pyplot import margins
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    #print(W.shape) # W  @(3073,10)
    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        dS = 0
        scores = X[i].dot(W) # s = X*W @(10,)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:  # right score
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                ds = 1 / num_train
                dsy = -1 / num_train
                # ds_dw = x[i]
                dW[:, j] += ds * X[i] 
                # every time margin > 0 makes a -1 to y[i]
                dW[:, y[i]] += dsy * X[i]
                loss += margin

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += 2* reg * W
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = X.shape[0] # N=500
    # W@（3073，10） X@(500, 3073) y@（500，）(1-d)
    scores = X @ W # @(500,10)
    scores_y = scores[range(N),[y]] # (1,500) numpy broadcast-> N = y.shape
    margin = scores - scores_y.T + 1
    #print(margin.shape)
    # now scores[y] = 1, make it =0
    margin[range(N),[y]] = 0
    _loss = np.maximum(margin,0, out=margin) # (500,10)
    loss = np.sum(_loss)/N + reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # 梯度计算
    binary = (margin > 0).astype(float)  # (N, C)
    row_sum = np.sum(binary, axis=1)       # 每个样本正 margin 的个数, (N,)
    binary[np.arange(N), y] = -row_sum     # 正确类别的梯度贡献为负计数
    dW = X.T @ binary / N + 2 * reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
