import numpy as np
from scipy.sparse import dia_matrix
from scipy.sparse import issparse
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix
from sklearn.metrics import r2_score
#from scipy.sparse import coo_matrix
import datetime

def sparse_outer_multiply(X,order='C'):
    if order not in ["C","F"]:
        raise ValueError('Invalid order: %s' %order)
    from scipy.sparse import issparse
    from scipy.sparse import csr_matrix
    from scipy.sparse import coo_matrix

    ca = []
    da = []
    ra = []
    s = set(np.arange(X[0].shape[0]))
    for x in X:
        if issparse(x):
            x = x.tocoo()
        else:
            x = coo_matrix(x)
        x.eliminate_zeros()
        s = s.intersection(set(x.row))
        ca.append(x.col)
        da.append(x.data)
        ra.append(x.row)
    
    scol = []
    srow = []
    sdat = []
    sh = 0
    for i in s:
        sh = 1
        dm = np.ones(1)
        cm = np.zeros(1, dtype = "int64")
        for j in range(len(X)):
            c = (ra[j].searchsorted(i),ra[j].searchsorted(i+1))
            dm = np.multiply.outer(dm[None,:],da[j][c[0]:c[1],None]).flatten()
            if order == "F":
                cm = np.add.outer(X[j].shape[1]*cm[None,:],ca[j][c[0]:c[1],None]).flatten()
            else:
                cm = np.add.outer(cm[None,:],sh*ca[j][c[0]:c[1],None]).flatten()
            sh *= X[j].shape[1]
        srow += list(np.ones( len(dm), dtype= "int64")*i)
        scol += list(cm)
        sdat += list(dm)
        
    return  csr_matrix((sdat, (srow, scol)), shape=(X[0].shape[0], sh))

def _center_scale_xy(X, Y, scale="range", center=True):
    """ Center X, Y and scale if the scale parameter==True

    Returns
    -------
        X, Y, x_mean, y_mean, x_std, y_std
    """
    #todo: scale array
    # center
    if center:
        x_mean = np.array(X.mean(axis=0)).flatten()
        y_mean = Y.mean(axis=0)
    else:
        x_mean = np.zeros(X.shape[1])
        y_mean = 0
    
    
    
    # scale
    if scale:
        if scale not in ["std", "range"]:
            raise ValueError('Invalid scale: %s' %
                             str(scale))
        if scale == "std":
#            
            if issparse(X):
                x_std = np.sqrt(np.array((X.power(2)).mean(axis=0)).flatten() - \
                                np.array(X.mean(axis=0)).flatten()**2)
            else:
                x_std = X.std(axis=0, ddof=1)
            x_std[x_std == 0.0] = 1.0
#            X /= x_std
            y_std = Y.std(axis=0, ddof=1)
            if not y_std:
                y_std = 1.0
            Y /= y_std
        else:
            x_std = X.max(axis=0) - X.min(axis=0)
            if issparse(X):
                x_std = x_std.toarray()
            x_std[x_std == 0.0] = 1.0
#            X /= x_std
            y_std = Y.std(axis=0, ddof=1)
            if not y_std:
                y_std = 1.0
            Y /= y_std
    else:
        x_std = np.ones(X.shape[1])
#        y_std = np.ones(Y.shape[1])
        y_std = 1.0 
        
    return X, Y, x_mean, y_mean, x_std, y_std

def _center_scale_xy_sparse(X, Y, center=True):
    """ Center X, Y and scale if the scale parameter==True

    Returns
    -------
        X, Y, x_mean, y_mean, x_std, y_std
    """
    #todo: scale array
    # center
    if center:
        x_mean = X.T.dot(csr_matrix(np.ones((X.shape[0],1)))).T/X.shape[0]
        y_mean = Y.mean(axis=0)
    else:
        x_mean = csr_matrix((1,X.shape[1]))
        y_mean = 0
    
        
    return x_mean, y_mean


class PLS():
    """Partial Least Squares (PLS)
    """

    def __init__(self, n_components=2, scale="range", center=True):
        self.n_components = n_components
        self.scale = scale
        self.center = center

    def fit(self, X, Y):
        """Fit model to data.

        Parameters
        ----------
        X : sparse, shape = [n_samples, n_features]
            Training vectors, where n_samples in the number of samples and
            n_features is the number of predictors.

        Y : array-like of response, shape = [n_samples, ]
            Target vectors, where n_samples in the number of samples and
            n_targets is the number of response variables.
        """
        Y = Y.copy()

        n = X.shape[0]
        p = X.shape[1]

        if self.n_components < 1 or self.n_components > p:
            raise ValueError('Invalid number of components: %d' %
                             self.n_components)
        # Scale (in place)
        X, Y, self.x_mean_, self.y_mean_, self.x_std_, self.y_std_ = (
            _center_scale_xy(X, Y, self.scale, self.center))
        # Results matrices
        self.T_ = np.zeros((n, self.n_components + self.center))
        self.W_ = np.zeros((p, self.n_components))
        self.D_ = np.zeros( self.n_components)
        self.STD_ = dia_matrix((1/self.x_std_, [0]), shape=(p, p))
        
        
        if self.center:
            self.T_[:,self.n_components] = np.ones(n)/np.sqrt(n)

        # NIPALS algo: outer loop, over components
        for k in range(self.n_components):
            y = Y - self.T_.dot(self.T_.T.dot(Y))
            w = self.STD_.dot(X.T.dot(y))
            self.W_[:,k] = w/np.linalg.norm(w)
            xw = X.dot(self.STD_.dot(self.W_[:,k]))
            t = xw - self.T_.dot(self.T_.T.dot(xw))
            d = np.linalg.norm(t)
            self.T_[:,k] = t/d
            self.D_[k] = d

        self.PWi_ = np.linalg.pinv( self.T_[:,:self.n_components].T.dot(X.dot(self.STD_.dot(self.W_))))
        self.D_ = dia_matrix((self.D_, [0]), shape=(self.n_components, self.n_components)).toarray()
        self.R_ = self.STD_.dot(self.W_.dot(self.PWi_.dot(self.D_)))
        self.B_ = self.STD_.dot(self.W_.dot(self.PWi_.dot(self.T_[:,:self.n_components].T.dot(Y))))*self.y_std_
        return self

    def transform(self, X, Y=None, copy=True):
        """Apply the dimension reduction learned on the train data.

        Parameters
        ----------
        X : array-like of predictors, shape = [n_samples, p]
            Training vectors, where n_samples in the number of samples and
            p is the number of predictors.

        Y : array-like of response, shape = [n_samples,], optional
            Training vectors, where n_samples in the number of samples and
            q is the number of response variables.

        copy : boolean, default True
            Whether to copy X and Y, or perform in-place normalization.

        Returns
        -------
        x_scores if Y is not given, (x_scores, y_scores) otherwise.
        """
        # Normalize
        if copy:
            X = X.copy()
#        X /= self.x_std_
#        X -= self.x_mean_
        # Apply rotation
        
        x_scores = X.dot(self.R_) - np.ones((X.shape[0],1))*(self.x_mean_[None,:].dot(self.R_))
        if Y is not None:
            if copy:
                Y = Y.copy()
            Y /= self.y_std_
            Y -= self.y_mean_
            return x_scores, Y

        return x_scores

    def predict(self, X, copy=True):
        """Apply the dimension reduction learned on the train data.

        Parameters
        ----------
        X : array-like of predictors, shape = [n_samples, p]
            Training vectors, where n_samples in the number of samples and
            p is the number of predictors.

        copy : boolean, default True
            Whether to copy X and Y, or perform in-place normalization.

        Notes
        -----
        This call requires the estimation of a p x q matrix, which may
        be an issue in high dimensional space.
        """
        #todo: fix predict with scale std
        if copy:
            X = X.copy()
        # Normalize
        
#        X /= self.x_std_
#        X -= self.x_mean_
        Ypred = X.dot((self.B_)) - np.ones(X.shape[0]).dot(self.x_mean_.dot((self.B_)))
        return (Ypred+ self.y_mean_) 

    def fit_transform(self, X, y=None):
        """Learn and apply the dimension reduction on the train data.

        Parameters
        ----------
        X : array-like of predictors, shape = [n_samples, p]
            Training vectors, where n_samples in the number of samples and
            p is the number of predictors.

        Y : array-like of response, shape = [n_samples, q], optional
            Training vectors, where n_samples in the number of samples and
            q is the number of response variables.

        copy : boolean, default True
            Whether to copy X and Y, or perform in-place normalization.

        Returns
        -------
        x_scores if Y is not given, (x_scores, y_scores) otherwise.
        """
        return self.fit(X, y).transform(X, y)
        
    def score(self, X, y, sample_weight=None):
        """Returns the coefficient of determination R^2 of the prediction.

        The coefficient R^2 is defined as (1 - u/v), where u is the regression
        sum of squares ((y_true - y_pred) ** 2).sum() and v is the residual
        sum of squares ((y_true - y_true.mean()) ** 2).sum().
        Best possible score is 1.0 and it can be negative (because the
        model can be arbitrarily worse). A constant model that always
        predicts the expected value of y, disregarding the input features,
        would get a R^2 score of 0.0.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.

        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True values for X.

        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.

        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y.
        """

        return r2_score(y, self.predict(X), sample_weight=sample_weight,
                        multioutput='variance_weighted')
    
class PLSMulti():
    """Partial Least Squares (PLS)
    """

    def __init__(self, n_components=2, center=True, roe = 1.,
                 max_iter=500, tol=1e-06, shuffle = True, mode = "include"):
        #todo scale
        self.n_components = n_components
        self.center = center
        self.roe = roe
        self.max_iter = max_iter
        self.tol = tol
        self.shuffle = shuffle
        self.mode = mode

    def fit(self, X, Y):
        """Fit model to data.

        Parameters
        ----------
        X : sparse, shape = [n_samples, n_features]
            Training vectors, where n_samples in the number of samples and
            n_features is the number of predictors.

        Y : array-like of response, shape = [n_samples, ]
            Target vectors, where n_samples in the number of samples and
            n_targets is the number of response variables.
        """ 
        from sklearn.model_selection import KFold
#        ddd = datetime.datetime.now()
        XX =  sparse_outer_multiply(X)
        
        Y = Y.copy()

        n = XX.shape[0]
        p = XX.shape[1]
        sh = np.cumprod([1] + [x.shape[1] for x in X])

        if self.n_components < 1 or self.n_components > p:
            raise ValueError('Invalid number of components: %d' %
                             self.n_components)
#        print(datetime.datetime.now() - ddd)
#        ddd = datetime.datetime.now()
        # Scale (in place)
        self.x_mean_, self.y_mean_= (_center_scale_xy_sparse(XX, Y, self.center)) #todo: center X, chack scale
        self.xm_mean_ = [_center_scale_xy_sparse(csc_matrix(x), Y, self.center)[0] for x in X]
        # Results matrices
        self.T_ = np.zeros((n, self.n_components + self.center))
        self.W_ = lil_matrix((self.n_components,p))
#        self.WM_ = [np.random.random((x.shape[1], self.n_components)) for x in X]
        self.WM_ = [np.ones((x.shape[1], self.n_components)) for x in X]
        self.D_ = np.zeros( self.n_components)
        
        
        if self.center:
            self.T_[:,self.n_components] = np.ones(n)/np.sqrt(n)

        # NIPALS algo: outer loop, over components
        if self.n_components == 1:
            kff = [([],[])]
        else:
            kf = KFold(n_splits=self.n_components, shuffle = self.shuffle)
            kff = kf.split(XX)
        k = 0
        for inc, ex in kff:
            y = Y - self.T_.dot(self.T_.T.dot(Y))
            if self.mode == "include": 
                y[ex] = 0
                y = y - self.T_.dot(self.T_.T.dot(y))
            if self.mode == "exlude":
                y[inc] = 0
                y = y - self.T_.dot(self.T_.T.dot(y))
            w = XX.T.dot(csr_matrix(y).T)
            w = w/ np.linalg.norm(w.data)
            v = np.ones((len(X),n))
            for i in range(self.max_iter):
#                print(i)
                for j in range(len(X)):
                    a = y.copy()
                    for l in range(len(X)):
                        if j ==l:
                            continue
                        a *= v[l]
                    wm = X[j].T.dot(a)
                    if np.linalg.norm(self.WM_[j][:,k] - wm/np.linalg.norm(wm)) < self.tol:
                        break
                    self.WM_[j][:,k] = wm/np.linalg.norm(wm)
                    v[j,:] = X[j].dot(self.WM_[j][:,k]) 
                    #todo: verbose
            wc = w.tocoo()
            wmw = np.prod([self.WM_[j][(wc.row % sh[j+1]) // sh[j],k] for j in range(len(X))], axis = 0)
            wn = np.sqrt((1 - (wmw**2).sum())*((1- self.roe)**2) + (((1- self.roe) *wmw + wc.data*self.roe)**2).sum())
            self.W_.data[k] = list(w.data/wn)
            self.W_.rows[k] = list(w.indices)
            self.WM_[0][:,k] = self.WM_[0][:,k]/wn
            xwm = np.prod([X[j].dot(self.WM_[j][:,k]) for j in range(len(X))], axis = 0)
#            w = self.roe* w + (1- self.roe) * a #todo: smart norm wip
            xw = self.roe* XX.dot(w/wn).T + (1- self.roe) * xwm
            t = (xw.T - self.T_.dot(self.T_.T.dot(xw.T))).flatten()
            d = np.linalg.norm(t)
            self.T_[:,k] = t/d
            self.D_[k] = d
            k+= 1
        
        self.W_ = self.W_.tocsc().T
        xWm = np.prod([X[j].dot(self.WM_[j]) for j in range(len(X))], axis = 0)
        self.PWi_ = np.linalg.pinv(self.roe *  XX.dot(self.W_).T.dot(self.T_[:,:self.n_components]).T +
                                   (1- self.roe) * xWm.T.dot(self.T_[:,:self.n_components]).T)
        self.D_ = dia_matrix((self.D_, [0]), shape=(self.n_components, self.n_components)).toarray()
        self.B_ = (self.PWi_.dot(self.T_[:,:self.n_components].T.dot(Y)))
        
        self.x_con = (1- self.roe)*np.ones((1,n)).dot(xWm.dot(self.PWi_.dot(self.D_)))/n
        self.x_con += self.roe * self.x_mean_.dot(self.W_.dot(self.PWi_.dot(self.D_)))
        
        self.b_con = (1- self.roe)*np.ones((1,n)).dot(xWm.dot(self.B_))/n
#        self.b_con = (1- self.roe)
#        for j in range(len(X)): 
#            self.b_con *= self.xm_mean_[j].dot(self.WM_[j].dot(self.B_))
        self.b_con += self.roe * self.x_mean_.dot(self.W_.dot(self.B_))
#        print(datetime.datetime.now() - ddd)
        return self

    def transform(self, X, Y=None, copy=True):
        """Apply the dimension reduction learned on the train data.

        Parameters
        ----------
        X : array-like of predictors, shape = [n_samples, p]
            Training vectors, where n_samples in the number of samples and
            p is the number of predictors.

        Y : array-like of response, shape = [n_samples,], optional
            Training vectors, where n_samples in the number of samples and
            q is the number of response variables.

        copy : boolean, default True
            Whether to copy X and Y, or perform in-place normalization.

        Returns
        -------
        x_scores if Y is not given, (x_scores, y_scores) otherwise.
        """
        # Normalize
        XX =  sparse_outer_multiply(X)
#        X /= self.x_std_
#        X -= self.x_mean_
        # Apply rotation
        
        xWm = np.ones((XX.shape[0],self.W_.shape[1]))
        for j in range(len(X)): 
            xWm *= X[j].dot(self.WM_[j]) 
        
        x_scores = self.roe * XX.dot(self.W_.dot(self.PWi_.dot(self.D_))) + \
            (1- self.roe) * xWm.dot(self.PWi_.dot(self.D_)) - np.ones((XX.shape[0],1)) * self.x_con 
        if Y is not None:
            if copy:
                Y = Y.copy()
#            Y /= self.y_std_
            Y -= self.y_mean_
            return x_scores, Y

        return x_scores

    def predict(self, X, copy=True):
        """Apply the dimension reduction learned on the train data.

        Parameters
        ----------
        X : array-like of predictors, shape = [n_samples, p]
            Training vectors, where n_samples in the number of samples and
            p is the number of predictors.

        copy : boolean, default True
            Whether to copy X and Y, or perform in-place normalization.

        Notes
        -----
        This call requires the estimation of a p x q matrix, which may
        be an issue in high dimensional space.
        """
        #todo: fix predict with scale std
        XX =  sparse_outer_multiply(X)
        # Normalize
        
#        X /= self.x_std_
#        X -= self.x_mean_
        xWm = np.ones((XX.shape[0],self.W_.shape[1]))
        for j in range(len(X)): 
            xWm *= X[j].dot(self.WM_[j]) 
            
        Ypred = self.roe * XX.dot((self.W_.dot(self.B_))) + (1- self.roe) * xWm.dot(self.B_)\
            - np.ones(XX.shape[0])*self.b_con
        return (Ypred+ self.y_mean_) 

    def fit_transform(self, X, y=None):
        """Learn and apply the dimension reduction on the train data.

        Parameters
        ----------
        X : array-like of predictors, shape = [n_samples, p]
            Training vectors, where n_samples in the number of samples and
            p is the number of predictors.

        Y : array-like of response, shape = [n_samples, q], optional
            Training vectors, where n_samples in the number of samples and
            q is the number of response variables.

        copy : boolean, default True
            Whether to copy X and Y, or perform in-place normalization.

        Returns
        -------
        x_scores if Y is not given, (x_scores, y_scores) otherwise.
        """
        return self.fit(X, y).transform(X, y)
        
    def score(self, X, y, sample_weight=None):
        """Returns the coefficient of determination R^2 of the prediction.

        The coefficient R^2 is defined as (1 - u/v), where u is the regression
        sum of squares ((y_true - y_pred) ** 2).sum() and v is the residual
        sum of squares ((y_true - y_true.mean()) ** 2).sum().
        Best possible score is 1.0 and it can be negative (because the
        model can be arbitrarily worse). A constant model that always
        predicts the expected value of y, disregarding the input features,
        would get a R^2 score of 0.0.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.

        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True values for X.

        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.

        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y.
        """

        return r2_score(y, self.predict(X), sample_weight=sample_weight,
                        multioutput='variance_weighted')

class PLSMultiOld():
    """Partial Least Squares (PLS)
    """

    def __init__(self, n_components=2, scale="range", center=True, roe = 1.,
                 max_iter=500, tol=1e-06, shuffle = True, mode = "include"):
        self.n_components = n_components
        self.scale = scale
        self.center = center
        self.roe = roe
        self.max_iter = max_iter
        self.tol = tol
        self.shuffle = shuffle
        self.mode = mode

    def fit(self, X, Y):
        """Fit model to data.

        Parameters
        ----------
        X : sparse, shape = [n_samples, n_features]
            Training vectors, where n_samples in the number of samples and
            n_features is the number of predictors.

        Y : array-like of response, shape = [n_samples, ]
            Target vectors, where n_samples in the number of samples and
            n_targets is the number of response variables.
        """ 
        from sklearn.model_selection import KFold
        ddd = datetime.datetime.now()
        XX =  sparse_outer_multiply(X)
        
        Y = Y.copy()

        n = XX.shape[0]
        p = XX.shape[1]

        if self.n_components < 1 or self.n_components > p:
            raise ValueError('Invalid number of components: %d' %
                             self.n_components)
        print(datetime.datetime.now() - ddd)
        ddd = datetime.datetime.now()
        # Scale (in place)
        XX, Y, self.x_mean_, self.y_mean_, self.x_std_, self.y_std_ = (
            _center_scale_xy(XX, Y, self.scale, self.center)) #todo: center X, chack scale
        # Results matrices
        self.T_ = np.zeros((n, self.n_components + self.center))
        self.W_ = np.zeros((p, self.n_components))
        self.WM_ = [np.ones((x.shape[1], self.n_components)) for x in X]
        self.D_ = np.zeros( self.n_components)
        self.STD_ = dia_matrix((1/self.x_std_, [0]), shape=(p, p))
        
        
        if self.center:
            self.T_[:,self.n_components] = np.ones(n)/np.sqrt(n)

        # NIPALS algo: outer loop, over components
        kf = KFold(n_splits=self.n_components, shuffle = self.shuffle)
#        for k in range(self.n_components): #todo: kfold (wip)
        k = 0
        for inc, ex in kf.split(XX):
            y = Y - self.T_.dot(self.T_.T.dot(Y))
#            if self.mode == "include": <------------------ wip
#                y = y[inc]
#            if self.mode == "exlude":
#                y = y[ex]
            w = XX.T.dot(y) #todo: sparse w
            w = w/np.linalg.norm(w)
            v = np.ones((len(X),n))
            for i in range(self.max_iter):
#                print(i)
                for j in range(len(X)):
                    a = y.copy()
                    for l in range(len(X)):
                        if j ==l:
                            continue
                        a *= v[l]
                    wm = X[j].T.dot(a)
#                    print(np.linalg.norm(self.WM_[j][:,k] - wm/np.linalg.norm(wm)))
                    self.WM_[j][:,k] = wm/np.linalg.norm(wm)
                    v[j,:] = X[j].dot(self.WM_[j][:,k]) #todo: tolarens chack
                    #todo: verbose
            a = np.ones(1)
            for j in range(len(X)): 
                a = (np.array([a]).transpose()*np.array([self.WM_[j][:,k]])).flatten()
            w = self.STD_.dot(self.roe* w + (1- self.roe) * a) #todo: smart norm
            self.W_[:,k] = w/np.linalg.norm(w)
            xw = XX.dot(self.STD_.dot(self.W_[:,k]))
            t = xw - self.T_.dot(self.T_.T.dot(xw))
            d = np.linalg.norm(t)
            self.T_[:,k] = t/d
            self.D_[k] = d
            k+= 1

        self.PWi_ = np.linalg.pinv( self.T_[:,:self.n_components].T.dot(XX.dot(self.STD_.dot(self.W_))))
        self.D_ = dia_matrix((self.D_, [0]), shape=(self.n_components, self.n_components)).toarray()
        self.R_ = self.W_.dot(self.PWi_.dot(self.D_))
        self.B_ = self.STD_.dot(self.W_.dot(self.PWi_.dot(self.T_[:,:self.n_components].T.dot(Y))))*self.y_std_
        print(datetime.datetime.now() - ddd)
        return self

    def transform(self, X, Y=None, copy=True):
        """Apply the dimension reduction learned on the train data.

        Parameters
        ----------
        X : array-like of predictors, shape = [n_samples, p]
            Training vectors, where n_samples in the number of samples and
            p is the number of predictors.

        Y : array-like of response, shape = [n_samples,], optional
            Training vectors, where n_samples in the number of samples and
            q is the number of response variables.

        copy : boolean, default True
            Whether to copy X and Y, or perform in-place normalization.

        Returns
        -------
        x_scores if Y is not given, (x_scores, y_scores) otherwise.
        """
        # Normalize
        XX =  sparse_outer_multiply(X)
#        X /= self.x_std_
#        X -= self.x_mean_
        # Apply rotation
        
        x_scores = XX.dot(self.STD_.dot(self.R_)) - np.ones((XX.shape[0],1))*(self.x_mean_[None,:].dot(self.STD_.dot(self.R_)))
        if Y is not None:
            if copy:
                Y = Y.copy()
            Y /= self.y_std_
            Y -= self.y_mean_
            return x_scores, Y

        return x_scores

    def predict(self, X, copy=True):
        """Apply the dimension reduction learned on the train data.

        Parameters
        ----------
        X : array-like of predictors, shape = [n_samples, p]
            Training vectors, where n_samples in the number of samples and
            p is the number of predictors.

        copy : boolean, default True
            Whether to copy X and Y, or perform in-place normalization.

        Notes
        -----
        This call requires the estimation of a p x q matrix, which may
        be an issue in high dimensional space.
        """
        #todo: fix predict with scale std
        XX =  sparse_outer_multiply(X)
        # Normalize
        
#        X /= self.x_std_
#        X -= self.x_mean_
        Ypred = XX.dot((self.B_)) - np.ones(XX.shape[0]).dot(self.x_mean_.dot((self.B_)))
        return (Ypred+ self.y_mean_) 

    def fit_transform(self, X, y=None):
        """Learn and apply the dimension reduction on the train data.

        Parameters
        ----------
        X : array-like of predictors, shape = [n_samples, p]
            Training vectors, where n_samples in the number of samples and
            p is the number of predictors.

        Y : array-like of response, shape = [n_samples, q], optional
            Training vectors, where n_samples in the number of samples and
            q is the number of response variables.

        copy : boolean, default True
            Whether to copy X and Y, or perform in-place normalization.

        Returns
        -------
        x_scores if Y is not given, (x_scores, y_scores) otherwise.
        """
        return self.fit(X, y).transform(X, y)

#X = np.random.rand(1000,99)
#Y = np.random.rand(1000)
#from sklearn.cross_decomposition import PLSRegression
##from scipy.sparse import csr_matrix
#
#X[X > 0.1] = 0
#X*= 10
#x = csr_matrix(X)
#x.eliminate_zeros()
#
##pp = PLSMulti(mode = "all", n_components= 2, roe= 0.3)
##p = PLSRegression(scale=False)
##p.fit(X,Y)
##pp.fit([x],Y)
##pd = np.abs(p.transform(X)*(p.transform(X)[0]/np.abs(p.transform(X)[0]))- pp.transform([x])*(pp.transform([x])[0]/np.abs(pp.transform([x])[0])))
##print(pd)
##print(p.predict(X).T - pp.predict([x]))
##print("hi")
#
#pp = PLSMulti(mode = "all", n_components= 2, roe= 0.5, center=True, tol=1e-15)
#p = PLSMultiOld(scale=False,mode = "all", n_components= 2, roe= 0.5, center=True)
#xl = [x,x]#[x,x,x]
#p.fit(xl,Y)
#pp.fit(xl,Y)
#pd = np.abs(p.transform(xl)*(p.transform(xl)[0]/np.abs(p.transform(xl)[0]))- pp.transform(xl)*(pp.transform(xl)[0]/np.abs(pp.transform(xl)[0])))
#print(pd)
#print(p.predict(xl).T - pp.predict(xl))
#print("hi")
##pp.fit([x,x,x],Y)
##print(np.abs(p.predict(x)- pp.predict([x])).max())