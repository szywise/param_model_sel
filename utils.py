import numpy as np
import matplotlib.pyplot as plt
import ipdb

def rel_error(x, y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
    
def tuneModel(Model, Xtrain, ytrain, tuning_hp_name, tuning_hp_list, other_hp_dict,
        method='cv'):

    num_features = Xtrain.shape[1]
    num_train = Xtrain.shape[0]

    np.random.seed(42)
    if method == 'holdout':
        # Hold out validation set
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

        # Return the best one in the tuning_hp_list
        mse_list_mean = mse_list
        idx = np.argmin(mse_list_mean)
        mse_list_stat = None
        return tuning_hp_list[idx], idx, mse_list_mean, mse_list_stat
    elif method == 'cv':
        perm = np.random.permutation(num_train)
        cv_fold = 5 # since there are only 67 training samples, use small fold
        num_per = num_train // cv_fold
        perm = perm[:num_train - (num_train % cv_fold)]
        mse_list = np.zeros((len(tuning_hp_list), cv_fold), dtype=float)
        for idx_hp, hp in enumerate(tuning_hp_list):
            model = Model(**{tuning_hp_name:hp}, **other_hp_dict)
            for idx_cv in range(cv_fold):
                val_mask = np.array([False] * num_train)
                val_mask[idx_cv*num_per:(idx_cv+1)*num_per] = True
                X_train = Xtrain[np.logical_not(val_mask)]
                y_train = ytrain[np.logical_not(val_mask)]
                X_val = Xtrain[val_mask]
                y_val = ytrain[val_mask]

                model.set_training_data(inputs=X_train, outputs=y_train)
                model.fit()
                ypred = model.produce(inputs=X_val)
                mse_list[idx_hp, idx_cv] = np.mean((ypred-y_val)**2)
        # Return the best one in the tuning_hp_list
        mse_list_mean = np.mean(mse_list, axis=1)
        mse_list_min = np.min(mse_list, axis=1)
        mse_list_max = np.max(mse_list, axis=1)
        mse_list_std = np.std(mse_list, axis=1)
        mse_list_stat = [mse_list_min, mse_list_max, mse_list_std]
        idx = np.argmin(mse_list_mean)
        return tuning_hp_list[idx], idx, mse_list_mean, mse_list_stat
    else:
        raise ValueError("Unknown method")

def testModel(Model, Xtrain, ytrain, Xtest, ytest, hp_dict):
    model = Model(**hp_dict)
    model.set_training_data(inputs=Xtrain, outputs=ytrain)
    model.fit()
    ypred = model.produce(inputs=Xtest)
    mse_test = np.mean((ypred-ytest)**2)
    return mse_test, model.coef, ypred

def viewPerformanceComp(ypred, ytest, coef, coef_true):
    plt.plot(ytest, label='Truth')
    for key in ypred:
        plt.plot(ypred[key], 'o:', label=key)
    plt.grid()
    plt.title("Predictions on test set")
    plt.legend()
    plt.show()

    if coef_true is not None:
        plt.plot(coef_true, label='Truth')
    for key in coef:
        plt.plot(coef[key], 'o--', label=key)
    plt.grid()
    plt.title("Learned coefficients")
    plt.legend()
    plt.show()
    
def viewCV(mse_list_mean, mse_list_param, best_idx, model_name):
    plt.figure()
    plt.plot(mse_list_mean, '+-')
    plt.plot(best_idx, mse_list_mean[best_idx], 'ro')
    method = 'holdout'
    if mse_list_param is not None:
        mse_list_min, mse_list_max, mse_list_std = mse_list_param
        xx = range(len(mse_list_mean))
        plt.fill_between(xx, mse_list_min, mse_list_max,
            alpha=0.2, edgecolor=None, facecolor='#FF9848')
        plt.fill_between(xx, mse_list_mean-mse_list_std, mse_list_mean+mse_list_std,
            alpha=0.2, edgecolor=None, facecolor='#98FF48')
        method = 'CV'
    plt.title(model_name + ' cross validation')
    plt.grid()
    plt.show()
    #plt.savefig(Model.__name__+'_CV.png')
    plt.close()

def viewPred(ytest, ypred, model_name):
    plt.figure()
    plt.plot(ytest, 'o-', label='ytest')
    plt.plot(ypred, '+', label='ypred')
    plt.legend()
    plt.title(model_name + " test mse: {:f}".format(
        np.mean((ytest - ypred)**2)))
    plt.grid()
    plt.show()
    plt.close()
    
def tune_test(Model, Xtrain, ytrain, Xtest, ytest, tuning_hp_name, tuning_hp_list, hp_dict):
    # Hyperparameter tuning
    print("### CROSS VAL ###")
    tuned_hp, tuned_hp_idx, mse_list_mean, mse_list_param = tuneModel(Model, Xtrain, ytrain, tuning_hp_name, tuning_hp_list, hp_dict)
    viewCV(mse_list_mean, mse_list_param, tuned_hp_idx, Model.__name__)
    hp_dict[tuning_hp_name] = tuned_hp

    # Give test performance
    print("### TEST ###")
    mse_test, coef, ypred = testModel(Model, Xtrain, ytrain, Xtest, ytest, hp_dict)
    print("test mse = {:f}".format(mse_test))
    viewPred(ytest, ypred, Model.__name__)
    
    return mse_test, coef, ypred