import numpy as np
import pandas as pd
import os

def get_multivariate_normal(nSamples, nFeatures):
    param_var, param_cov = [1, 0.9]
    mu = np.zeros(nFeatures)
    cov = np.full((nFeatures,nFeatures), param_cov)
    for i in range(nFeatures):
        cov[i,i] = param_var
    return np.random.multivariate_normal(mu, cov, nSamples)

def generate_data_0():
    Xtrain = np.array([-1, 2, 3]).reshape(-1, 1)
    ytrain = np.array([0, 3, 4])
    Xtest = np.array([0, 1.5]).reshape(-1, 1)
    ytest = np.array([1, 2.5])
    coef = np.array([1])
    intercept = 1
    return Xtrain, ytrain, Xtest, ytest, coef, intercept
    
def generate_data_1():
    """
    y = X * coef + noise
    """
    nSamples = 100
    nFeatures = 100

    # design matrix
    param_var = 1
    X = np.random.randn(nSamples, nFeatures) * np.sqrt(param_var)
    X[:, 20:30] = get_multivariate_normal(nSamples, 10)
    X[:, 60:70] = get_multivariate_normal(nSamples, 10)
    X = X - np.mean(X, 0) # centered
    X = X / np.linalg.norm(X, ord=2, axis=0) # normalized
    X = container.ndarray(X)
    
    # noise: variance 0.01
    noise = np.random.randn(nSamples) * 0.1

    # coef
    coef = np.zeros(nFeatures)
    coef[20:30] = -1
    coef[60:70] = 1
    
    y = X.dot(coef) + noise
    y = container.ndarray(y)
    return coef, X, y

def generate_data_2():
    """
    # not this one: Zou, Hastie. Elastic Net, 2005
    Zhong, Kwok. Efficient sparse modeling with automatic features grouping, IEEE Signal Processing Letters, 2012
    
    p = 40, 15 nonzeros and 25 zeros
    """
    nFeatures = 40
    nSamples_split = [1000, 200] #ElasticNet: [50, 50, 400] # train/val/test
    nSamples = sum(nSamples_split)

    coef = np.zeros(nFeatures)
    coef[:15] = 3
    intercept = 0.0

    np.random.seed(42)
    z = np.random.randn(3) # z is not drawn for every sample i
    # z = array([ 0.49671415, -0.1382643 ,  0.64768854])
    eps_std = 0.4
    eps = np.random.randn(nSamples, 15) * eps_std
    X = np.zeros((nSamples, nFeatures))
    for i in range(3):
        X[:,i*5:(i+1)*5] = z[i] + eps[:,i*5:(i+1)*5]
    X[:,15:] = np.random.randn(nSamples, 25)
    X = X - np.mean(X, 0) # centered
    X = X / np.linalg.norm(X, ord=2, axis=0) # normalized

    noise_std = 1
    noise = np.random.randn(nSamples) * noise_std
    y = X.dot(coef) + noise
    
    idx_test = np.random.choice(nSamples, nSamples_split[-1], replace=False)
    mask_test = np.array([False] * nSamples)
    mask_test[idx_test] = True
    Xtrain = X[np.logical_not(mask_test)]
    ytrain = y[np.logical_not(mask_test)]
    Xtest = X[mask_test]
    ytest = y[mask_test]
    
    return Xtrain, ytrain, Xtest, ytest, coef, intercept

def get_prostate_cancer_data():
    if not os.path.isfile('prostate.data'):
        url = 'https://web.stanford.edu/~hastie/ElemStatLearn/datasets/prostate.data'
        df = pd.read_csv(url, sep='\t', header=0)
    else:
        df = pd.read_csv('prostate.data', sep='\t', header=0)
    Xtrain = df[df.train=='T'].iloc[:, 1:-2].values
    ytrain = df[df.train=='T'].iloc[:, -2].values
    Xtest = df[df.train=='F'].iloc[:, 1:-2].values
    ytest = df[df.train=='F'].iloc[:, -2].values
    coef, intercept = [None, None]
    return Xtrain, ytrain, Xtest, ytest, coef, intercept