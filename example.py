import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models import LinearRegression, OWL, EN, Lasso, Ridge
import ipdb

def get_multivariate_normal(nSamples, nFeatures):
    param_var, param_cov = [1, 0.9]
    mu = np.zeros(nFeatures)
    cov = np.full((nFeatures,nFeatures), param_cov)
    for i in range(nFeatures):
        cov[i,i] = param_var
    X = np.random.multivariate_normal(mu, cov, nSamples)
    X = X - np.mean(X, 0) # centered
    X = X / np.linalg.norm(X, ord=2, axis=0) # normalized
    return X

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
    Zhong, Kwok. Efficient sparse modeling with automatic features grouping, IEEE Signal Processing Letters, 2012
    
    p = 40, 15 nonzeros and 25 zeros
    """
    nFeatures = 40
    nSamples = 20

    coef = np.zeros(nFeatures)
    coef[:15] = 3

    z = np.random.randn(nSamples, 3)
    eps_std = 0.4
    eps = np.random.randn(nSamples, 15) * eps_std
    X = np.zeros((nSamples, nFeatures))
    for i in range(3):
        X[:,i*5:(i+1)*5] = z[:,i:i+1] + eps[:,i*5:(i+1)*5]
    X[:,15:] = np.random.randn(nSamples, 25)
    X = X - np.mean(X, 0) # centered
    X = X / np.linalg.norm(X, ord=2, axis=0) # normalized
    X = container.ndarray(X)

    y = X.dot(coef) # + noise
    y = container.ndarray(y)
    return coef, X, y

def get_prostate_cancer_data():
    url = 'https://web.stanford.edu/~hastie/ElemStatLearn/datasets/prostate.data'
    df = pd.read_csv(url, sep='\t', header=0)
    Xtrain = df[df.train=='T'].iloc[:, 1:-2].values
    ytrain = df[df.train=='T'].iloc[:, -2].values
    Xtest = df[df.train=='F'].iloc[:, 1:-2].values
    ytest = df[df.train=='F'].iloc[:, -2].values
    return Xtrain, ytrain, Xtest, ytest

def owl_regression(inputs, outputs, hp, fname):
    plt.figure()
    plt.plot(inputs[:3,:].T)
    plt.title("First 3 features of the design matrix")
    plt.savefig(fname + "_features.png")
    plt.show()

    primitive = OWLRegression(hyperparams=hp)
    primitive.set_training_data(inputs=inputs, outputs=outputs)
    primitive.fit(iterations=1000)

    plt.figure()
    plt.plot(primitive._weight)
    plt.title("OWL weights")
    plt.savefig(fname + "_OWL_weights.png")
    plt.show()

    plt.figure()
    plt.plot(primitive._loss_history)
    plt.title("Training loss history")
    plt.savefig(fname + "_loss.png")
    plt.show()

    plt.figure()
    plt.plot(true_param, label='True')
    plt.plot(primitive._coef, label='Estimated')
    plt.title('Coeffcients, MSE={}'.format(np.mean((true_param - primitive._coef)**2)))
    plt.legend()
    plt.savefig(fname + "_coef.png")
    plt.show()
    
    return primitive._coef

def tuneModel(Model, Xtrain, ytrain, tuning_hp_name, tuning_hp_list, other_hp_dict):

    # Hold out validation set
    num_features = Xtrain.shape[1]
    num_train = Xtrain.shape[0]

    np.random.seed(42)
    val_idx = np.random.choice(num_train, size=int(num_train/5), replace=False)
    val_mask = np.array([False] * num_train)
    val_mask[val_idx] = True

    X_train = Xtrain[np.logical_not(val_mask)]
    y_train = ytrain[np.logical_not(val_mask)]
    X_val = Xtrain[val_mask]
    y_val = ytrain[val_mask]
    
    # Training and validation
    mse_list = np.zeros(len(tuning_hp_list), dtype=float)
    for idx_hp, hp in enumerate(tuning_hp_list):
        model = Model(**{tuning_hp_name:hp}, **other_hp_dict)
        model.set_training_data(inputs=X_train, outputs=y_train)
        model.fit()
        ypred = model.produce(inputs=X_val)
        mse_list[idx_hp] = np.mean((ypred-y_val)**2)

    # Plot
    plt.figure()
    plt.plot(mse_list, '+-')
    plt.title(Model.__name__)
    plt.savefig(Model.__name__+'_CV.png')
    plt.show()
    plt.close()

    # Return the best one in the tuning_hp_list
    idx = np.argmin(mse_list)
    return tuning_hp_list[idx]

def testModel(Model, Xtrain, ytrain, Xtest, ytest, tuning_hp_name, tuned_hp, other_hp_dict):
    model = Model(**{tuning_hp_name:tuned_hp}, **other_hp_dict)
    model.set_training_data(inputs=Xtrain, outputs=ytrain)
    model.fit()
    ypred = model.produce(inputs=Xtest)
    mse_test = np.mean((ypred-ytest)**2)
    return mse_test, model.coef

if __name__ == "__main__":
    # Get dataset
    Xtrain, ytrain, Xtest, ytest = get_prostate_cancer_data()
    num_features = Xtrain.shape[1]
    num_train = Xtrain.shape[0]
    num_test = Xtest.shape[0]

    # Hyperparameter tuning
    kwargs = {
            'fit_intercept': True,
            'normalize': True,
            'learning_rate': 1e-3,
            'tol': 1e-3,
            'max_iter': 1000,
            'verbose': 1
            }
    hp_list = [0.0] #[0, 1e-3, 1e-2, 1e-1, 1]
    hp_list.extend(np.logspace(-5, 0, 4).tolist())
    mse_test = {}
    coef = {}

    owl_weight_list = [np.linspace(hp, 0, num_features) for hp in hp_list]
    owl_weight = tuneModel(OWL, Xtrain, ytrain, 'weight', owl_weight_list, kwargs)
    mse_test['OWL'], coef['OWL'] = testModel(OWL, Xtrain, ytrain, Xtest, ytest, 'weight', owl_weight, kwargs)

    best_lam = tuneModel(EN, Xtrain, ytrain, 'lam', hp_list, {**{'l1_ratio':0.5}, **kwargs})
    mse_test['EN'], coef['EN'] = testModel(EN, Xtrain, ytrain, Xtest, ytest, 'lam', best_lam, {**{'l1_ratio':0.5}, **kwargs})

    best_lam = tuneModel(Lasso, Xtrain, ytrain, 'lam', hp_list, kwargs)
    mse_test['Lasso'], coef['Lasso'] = testModel(Lasso, Xtrain, ytrain, Xtest, ytest, 'lam', best_lam, kwargs)

    best_lam = tuneModel(Ridge, Xtrain, ytrain, 'lam', hp_list, kwargs)
    mse_test['Ridge'], coef['Ridge'] = testModel(Ridge, Xtrain, ytrain, Xtest, ytest, 'lam', best_lam, kwargs)

    #best_lam = tuneModel(LinearRegression, Xtrain, ytrain, None, hp_list, kwargs)
    mse_test['LinearRegression'], coef['LinearRegression'] = testModel(Ridge, Xtrain, ytrain, Xtest, ytest, 'lam', 0, kwargs)

    # Model comparison
    plt.figure()
    model_names = ['OWL', 'EN', 'Lasso', 'Ridge', 'LinearRegression']
    plt.bar(range(5), [mse_test[key] for key in model_names])
    plt.xticks(range(5), model_names)
    plt.savefig('model_test_mse.png')
    plt.close()

    plt.figure()
    for key in coef:
        plt.plot(coef[key], label=key)
    plt.legend()
    plt.savefig('coef.png')
    plt.close()
