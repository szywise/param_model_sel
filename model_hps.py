# Best setting for Zhong's paper
Model_hps = {
    'OWL': {
        'weight': np.linspace(1e-2, 1e-3, num_features),
        'learning_rate': 1e-3,
        'tol': 1e-5,
        'max_iter': 10000,
    },
    'EN': {
        'lam': 3e-3,
        'l1_ratio': 0.8,
        'learning_rate': 1e-2,
        'tol': 1e-5,
        'max_iter': 5000,
    },
    'Lasso': {
        'lam': 4e-2,
        'learning_rate': 1e-2,
        'tol': 1e-5,
        'max_iter': 5000,
    },
    'Ridge': {
        'lam': 4e-2,
        'learning_rate': 1e-2,
        'tol': 1e-5,
        'max_iter': 5000,
    },
    'LinearRegression': {
        'learning_rate': 1e-3,
        'tol': 1e-5,
        'max_iter': 10000,
    },
}
