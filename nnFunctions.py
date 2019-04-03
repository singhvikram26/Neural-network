import numpy as np
from scipy.optimize import minimize
from math import sqrt
import pickle
import  gc
'''
You need to modify the functions except for initializeWeights() and preprocess()
'''

def initializeWeights(n_in, n_out):
    '''
    initializeWeights return the random weights for Neural Network given the
    number of node in the input layer and output layer

    Input:
    n_in: number of nodes of the input layer
    n_out: number of nodes of the output layer

    Output:
    W: matrix of random initial weights with size (n_out x (n_in + 1))
    '''
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def preprocess(filename,scale=True):
    '''
     Input:
     filename: pickle file containing the data_size
     scale: scale data to [0,1] (default = True)
     Output:
     train_data: matrix of training set. Each row of train_data contains
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     test_data: matrix of training set. Each row of test_data contains
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    '''
    with open(filename, 'rb') as f:
        train_data = pickle.load(f)
        train_label = pickle.load(f)
        test_data = pickle.load(f)
        test_label = pickle.load(f)
    # convert data to double
    train_data = train_data.astype(float)
    test_data = test_data.astype(float)

    # scale data to [0,1]
    if scale:
        train_data = train_data/255
        test_data = test_data/255

    return train_data, train_label, test_data, test_label

def sigmoid(z):
    '''
    Notice that z can be a scalar, a vector or a matrix
    return the sigmoid of input z (same dimensions as z)
    '''

    # your code here - remove the next four lines

    if np.isscalar(z):
        s = 1/(1+np.exp(-z))
        print(s)
        print(np.e)
    else:

        a = np.power(np.e, np.negative(z))
        d = 1 + a
        s = 1/d


    return s

def nnObjFunction(params, *args):
    '''
    % nnObjFunction computes the value of objective function (cross-entropy
    % with regularization) given the weights and the training data and lambda
    % - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices W1 (weights of connections from
    %     input layer to hidden layer) and W2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not including the bias node)
    % n_hidden: number of node in hidden layer (not including the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % train_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % train_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.

    % Output:
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector (not a matrix) of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.
    '''
    n_input, n_hidden, n_class, train_data, train_label, lambdaval = args
    # First reshape 'params' vector into 2 matrices of weights W1 and W2

    W1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    W2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here
    #
    #
    x = np.ones((1,train_data.shape[0]))

    train_data= np.concatenate((train_data,x.T ),axis=1)
    a= np.matmul(train_data, W1.T)
    z= sigmoid(a)
    z= np.concatenate((z,x.T),axis=1)
    b= np.matmul(z,W2.T)
    o= sigmoid(b)
    #Backpropogation

    #1 of k coding
    Y= np.zeros((train_label.shape[0],n_class))
    Y[range(0,train_label.shape[0]),(train_label.astype(int))]=1
    ###############

    #error calculation
    r= 1- o
    p= np.log(o)


    Ji=np.matmul(Y.T,p) + np.matmul((1-Y).T,np.log(r))
    Ji= Ji.flatten()
    J=-(np.add.reduce(Ji)/ Ji.size)

    #### regularize J #####

    J+= (lambdaval/(2*train_data.shape[0])) * (np.sum(W1**2) + (np.sum(W2**2)))

    obj_val= J


    #########################


    #gradient w.r.t weights
    q= o-Y
    grad_W2= np.matmul(q.T,z)

    ###########regularize

    grad_W2= (grad_W2 + lambdaval*W2)/train_data.shape[0]


    ##############Gradienec W1

    Z= np.delete(z,n_hidden , 1)
    w2= np.delete(W2,n_hidden,1)
    r= np.multiply((1-Z), Z)
    v= np.matmul(o-Y, w2)


    u= np.multiply(r,v)
    grad_W1= np.matmul(u.T,train_data)

    #############regularize  grad_W1

    grad_W1= (grad_W1 + lambdaval*W1)/train_data.shape[0]




    obj_grad = np.concatenate((grad_W1.flatten(), grad_W2.flatten()),0)
    ##########################
    ##g= np.add.reduce(obj_grad)
    ##grad= g/obj_grad.size

    #params= params - 0.1*grad

    #k= np.add.reduce(params**2)

    #Jr= J + ((lambdaval/(2*params.size))*k)


    



    #
    #
    #



    # Make sure you reshape the gradient matrices to a 1D array. for instance if
    # your gradient matrices are grad_W1 and grad_W2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_W1.flatten(), grad_W2.flatten()),0)

    gc.collect()
    return (obj_val, obj_grad)


def nnPredict(W1, W2, data):
    '''
    % nnPredict predicts the label of data given the parameter W1, W2 of Neural
    % Network.

    % Input:
    % W1: matrix of weights for hidden layer units
    % W2: matrix of weights for output layer units
    % data: matrix of data. Each row of this matrix represents the feature
    %       vector of a particular image

    % Output:
    % label: a column vector of predicted labels
    '''

    labels = np.zeros((data.shape[0],))
    # Your code here

    x = np.ones((1,data.shape[0]))

    train_data= np.concatenate((data,x.T),axis=1)
    a= np.matmul(train_data, W1.T)
    z= sigmoid(a)
    z= np.concatenate((z,x.T),axis=1)
    b= np.matmul(z,W2.T)
    labels= sigmoid(b)
    lab= np.ones((labels.shape[0],))
    for i in range(0,labels.shape[0]):
        lab[i]= np.nanargmax(labels[i])

    #print("---",labels[0])
    #print(np.argmax(labels[0]))

    return lab
