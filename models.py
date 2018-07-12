import os
import numpy as np
import ipdb

def fista(*, x_init, loss_fun, grad_fun, prox_fun, max_iter=1000, tol=1e-3, learning_rate=1e-3, verbose=2):
    x_old = x_init.copy()
    z = x_init.copy()
    t_old = 1
    loss_history = [loss_fun(x_init)]

    if verbose >= 1:
        print("FISTA training begins:")

    for i in range(max_iter):
        z = z - learning_rate * grad_fun(z)
        x = prox_fun(z)
        
        t = (1 + np.sqrt(1+4*t_old**2)) / 2
        z = x + ((t_old-1)/t) * (x - x_old)

        eps = np.linalg.norm(x - x_old) / (np.linalg.norm(x_old) + 1e-10)
        if eps < tol:
            break
        x_old = x
        t_old = t

        loss_history.append(loss_fun(x))

        if (verbose == 2 and i % max(1, max_iter//20) == 0
            or verbose >= 3):
            print("iter = {:4d}, loss = {:f}".format(i, loss_history[-1]))

    n_iter = len(loss_history) - 1
    if verbose >= 1:
        print("FISTA training exits after {} iterations, with loss={:.2E} and eps={:.2E}\n".format(
            n_iter, loss_history[-1], eps))

    return x, loss_history, n_iter

class LinearRegression():
    def __init__(self, *, fit_intercept=True, normalize=False, learning_rate=1e-3, tol=1e-3, max_iter=1000, verbose=2):
        # Hyperparameters
        self._fit_intercept = fit_intercept
        self._normalize = normalize
        self._learning_rate = learning_rate
        self._tol = tol
        self._max_iter = max_iter
        self._verbose = verbose
        
        # Parameters
        self.coef = None
        self.intercept = None

        # Training data
        self.X = None
        self.y = None
    
        # Status
        self.fitted = False
        self.loss_history = []
        self.n_iter = 0

    def set_training_data(self, *, inputs, outputs) -> None:
        inputs = np.array(inputs).astype(float)
        outputs = np.array(outputs).astype(float)

        assert inputs.ndim == 2, "X should be 2-dimensional array"
        assert inputs.shape[0] == outputs.shape[0], "Shapes of X and y don't match"
        assert outputs.shape[0] == outputs.flatten().shape[0], "All dims of y, except the 1st one, should be 1"
        assert inputs.shape[0] > 0, "Empty input"
        outputs = outputs.flatten()
        self.X = inputs
        self.y = outputs

        # Clear parameters and status
        self.coef = None
        self.intercept = None
        self.fitted = False
        self.loss_history = []
        self.n_iter = 0

    @staticmethod
    def _preprocessing(X, y, fit_intercept, normalize):
        """
        Static preprocessing method to centerize and normalize the data on demand.
        Note if fit_intercept is False, then normalize is ignored.
        """
        if fit_intercept:
            X_offset = np.mean(X, axis=0)
            X -= X_offset
            if normalize:
                X_scale = np.linalg.norm(X, axis=0)
                X = X / X_scale
            else:
                X_scale = np.ones(X.shape[1], dtype=X.dtype)
            y_offset = np.mean(y, axis=0)
            y -= y_offset
        else:
            X_offset = np.zeros(X.shape[1], dtype=X.dtype)
            X_scale = np.ones(X.shape[1], dtype=X.dtype)
            y_offset = X.dtype.type(0)

        return X, y, X_offset, y_offset, X_scale

    def _loss_fun(self, X, y, coef):
        """
            Cost =  1/N * (y - X * coef)^2      --------> MSE loss in data-fitting
        """
        return np.mean((y - X.dot(coef))**2)

    def _grad_fun(self, X, y, coef):
        """
        grad of the smooth term in loss
        """
        return 2 * X.T.dot(X.dot(coef) - y) / X.shape[0]

    def _prox_fun(self, coef):
        """
        prox of the non-smooth term in loss
        """
        return coef

    def fit(self) -> None:
        # State/input checking
        if self.fitted:
            return None
        if not hasattr(self, 'X') or not hasattr(self, 'y'):
            raise ValueError("Missing training data.")

        # Preprocessing
        X, y, X_offset, y_offset, X_scale = self._preprocessing(self.X, self.y, self._fit_intercept, self._normalize)

        # Fit with FISTA
        coef_init = np.zeros(X.shape[1], dtype=float)
        coef, self.loss_history, self.n_iter = fista(x_init=coef_init,
                loss_fun=lambda coef: self._loss_fun(X, y, coef),
                grad_fun=lambda coef: self._grad_fun(X, y, coef),
                prox_fun=lambda coef: self._prox_fun(coef),
                max_iter=self._max_iter,
                tol=self._tol,
                learning_rate=self._learning_rate,
                verbose=self._verbose)
        self.coef = coef / X_scale
        if self._fit_intercept:
            self.intercept = float(y_offset - X_offset.dot(self.coef))
        else:
            self.intercept = float(0)
        self.fitted = True

    def produce(self, *, inputs) -> np.ndarray:
        if self.fitted is False:
            raise ValueError("Call produce before model fitted.")
        if inputs.shape[1] != self.coef.shape[0]:
            raise ValueError("Input dimension {:d} does not match the dimension {:d} the model is trained with.".format(
                inputs.shape[1], self.coef.shape[0]))
        return inputs.dot(self.coef) + self.intercept


class OWL(LinearRegression):
    def __init__(self, *, weight, **kwargs) -> None:
        super().__init__(**kwargs)

        weight = np.array(weight).astype(float)
        assert weight.ndim == 1, "OWL weights must be a 1-dim vector"
        dw = np.diff(weight)
        assert np.all(dw <= 0), "OWL weights should be non-increasing"
        assert np.all(weight >= 0), "OWL weights should be all non-negative"
        self._weight = weight

    def set_training_data(self, *, inputs, outputs) -> None:
        super().set_training_data(inputs=inputs, outputs=outputs)
        assert self.X.shape[1] == self._weight.shape[0], "Number of features in X ({:d}) does not match the dimension of OWL weight ({:d})".format(self.X.shape[1], self._weight.shape[0]) 

    def _loss_fun(self, X, y, coef):
        """
            Cost =  1/N * (y - X * coef)^2      --------> MSE loss in data-fitting
                    + (weight * coef_sort)      --------> OWL regularization term
        """
        loss_fit = np.mean((y - X.dot(coef))**2)
        loss_reg = np.sum(self._weight * np.sort(np.abs(coef))[::-1])
        return loss_fit + loss_reg

    #@staticmethod #for LASSO not
    def _grad_fun(self, X, y, coef):
        """
        grad of the smooth term in loss
        """
        return 2 * X.T.dot(X.dot(coef) - y) / X.shape[0]

    def _prox_fun(self, coef):
        """
        prox of the non-smooth term in loss

        Arguments: 
            coef: the input of the proximal operator

        Proximal operator arguments:
            z:  the alias of the coef
                z = x_t - lr * Gradient (f(x_t)) with lr being the learning rate
            mu: mu = lr * w, where lr is the learning rate, and w are the OWL params.
                It must be nonnegative and in non-increasing order. 

        Returns:
            x:  Note both x and z are of the same shape as coef
        """
        #ipdb.set_trace()
        z = coef
        mu = self._learning_rate * self._weight

        # Cache the signs of z
        sgn = np.sign(z)
        z = abs(z)
        idx = z.argsort()[::-1]
        z = z[idx]
        
        # Find the index of the last positive entry in vector z - mu  
        flag = 0
        n = z.size
        x = np.zeros_like(z)
        diff = (z - mu)[::-1]
        indc = np.argmax(diff>0)
        flag = diff[indc]

        # Apply prox on non-negative subsequence of z - mu
        if flag > 0:
            k = n - indc
            v1 = z[:k]
            v2 = mu[:k]
            v = self._prox_owl_part(v1,v2)
            x[idx[:k]] = v
        else:
            pass

        # Restore signs
        x = sgn * x

        return x

    @staticmethod
    def _prox_owl_part(v1, v2):
        """
        Stack-based algorithm for FastProxSL1. c.f. Bogdan's SLOPE paper, Algorithm 4
        """
        # Stack element
        class PTR(object):
            """
            stack element object: (i, j, s, w)
            """
            def __init__(self, i_, j_, s_, w_):
                self.i = i_
                self.j = j_
                self.s = s_
                self.w = w_

        # Stack-based algorithm
        v = v1 - v2
        stk = []

        for i in range(v.shape[0]):
            ptr = PTR(i, i, v[i], v[i])
            stk.append(ptr)
            while True:
                if len(stk) > 1 and stk[-2].w <= stk[-1].w:
                    ptr = stk.pop()
                    stk[-1].j = i
                    stk[-1].s += ptr.s
                    stk[-1].w = stk[-1].s / (i - stk[-1].i + 1)
                else:
                    break
        
        x = np.zeros_like(v)
        for idx, ptr in enumerate(stk):
            for i in range(ptr.i, ptr.j+1):
                x[i] = max(0, ptr.w)

        return x


class EN(LinearRegression):
    def __init__(self, *, lam, l1_ratio, **kwargs) -> None:
        super().__init__(**kwargs)

        assert lam >= 0, "lam should be nonnegetive"
        assert 0 <= l1_ratio <= 1, "l1_ratio should be in [0, 1]"
        self._lam = lam
        self._l1_ratio = l1_ratio

    def _loss_fun(self, X, y, coef):
        """
            Cost =  1/N * (y - X * coef)^2      --------> MSE loss in data-fitting
                 +  lam * (l1_ratio * \|coef\|_1 + (1-l1_ratio * \|coef\|_2^2)
        """
        loss_fit = np.mean((y - X.dot(coef))**2) 
        loss_reg = self._lam * (self._l1_ratio * np.linalg.norm(coef, ord=1)
                        + (1 - self._l1_ratio) * np.linalg.norm(coef, ord=2)**2)
        return loss_fit + loss_reg

    def _grad_fun(self, X, y, coef):
        """
        grad of the smooth term in loss
        """
        return (2 * X.T.dot(X.dot(coef) - y) / X.shape[0]
                + 2 * self._lam * (1 - self._l1_ratio) * coef)

    def _prox_fun(self, coef):
        """
        prox of the non-smooth term in loss
        """
        z = coef
        thresh = self._learning_rate * self._lam * self._l1_ratio

        # Soft thresholding
        z_sign = np.sign(z)
        z = np.maximum(np.abs(z) - thresh, 0)
        return z_sign * z 


class Lasso(EN):
    def __init__(self, *, lam, **kwargs) -> None:
        super().__init__(lam=lam, l1_ratio=1.0, **kwargs)


class Ridge(LinearRegression):
    def __init__(self, *, lam, **kwargs) -> None:
        super().__init__(**kwargs)
        
        assert lam >= 0, "lam should be nonnegative"
        self._lam = lam

    def _loss_fun(self, X, y, coef):
        """
            Cost =  1/N * (y - X * coef)^2      --------> MSE loss in data-fitting
                 +  lam * \|coef\|_2^2)
        """
        loss_fit = np.mean((y - X.dot(coef))**2) 
        loss_reg = self._lam * np.linalg.norm(coef, ord=2)**2
        return loss_fit + loss_reg

    def _grad_fun(self, X, y, coef):
        """
        grad of the smooth term in loss
        """
        return (2 * X.T.dot(X.dot(coef) - y) / X.shape[0]
                + 2 * self._lam * coef)

    def _prox_fun(self, coef):
        """
        prox of the non-smooth term in loss
        """
        return coef

