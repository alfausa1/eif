""" 
Extended Isolation forest functions

This is the implementation of the Extended Isolation Forest anomaly detection algorithm. This extension, improves the consistency and reliability of the anomaly score produced by standard Isolation Forest represented by Liu et al.
Our method allows for the slicing of the data to be done using hyperplanes with random slopes which results in improved score maps. The consistency and reliability of the algorithm is much improved using this extension.

This fork of https://github.com/lpryszcz/eif is a slight modification that aims to refactor some pieces of code and further improve performance with the use of numba.  
The functionality to calculate the anomaly threshold based on samples with a contamination parameter is also added.
"""

__author__ = 'Leszek Pryszcz and Alvaro Faubel Sanchis (relying on method developed by Matias Carrasco Kind & Sahand Hariri)'
import numpy as np
import os
from numba import jit

@jit
def c_factor(n):
    """
    Return average path length of unsuccesful search in a binary search tree given n points.
    """
    return 2.0*(np.log(n-1)+0.5772156649) - (2.0*(n-1.)/(n*1.0))

@jit
def minmax(x):
    """
    np.min(x, axis=0), np.max(x, axis=0) for 2D array but faster
    """
    m, n = len(x), len(x[0])
    mi, ma = np.empty(n), np.empty(n)
    mi[:] = ma[:] = x[0]
    for i in range(1, m):
        for j in range(n):
            if x[i, j]>ma[j]: ma[j] = x[i, j]
            elif x[i, j]<mi[j]: mi[j] = x[i, j]
    return mi, ma

@jit
def split(x, w):
    """
    x[w], x[~w] but faster
    """
    k = l = 0
    a, b = np.empty_like(x), np.empty_like(x)
    for i in range(len(x)):
        if w[i]: 
            a[k] = x[i]
            k += 1
        else:
            b[l] = x[i]
            l += 1
    return a[:k], b[:l]

@jit
def scale_minmax(a, mi, ma):
    """
    Return a scaled by mi-ma
    """
    return a * (ma-mi) + mi

@jit
def dot(p, n):
    """
    Return p.dot(n)
    """
    return p.dot(n)

@jit
def update_nodes(ni, w, limit, e):
    """
    Update node index base on w, limit and e
    """
    r = 2**(limit-e)
    for i in range(len(ni)):
        if w[i]: ni[i] += 1
        else: ni[i] += r
    return ni

@jit
def score_false(e, sel):
    """
    Return scores for internal-terminal nodes
    """
    return e*(~sel).sum()

@jit
def score_terminal(limit, ni, size):
    """
    Return limit*len(ni) + c_factor(size[size>1]).sum()
    """
    return limit*len(ni) + c_factor(size[size>1]).sum()

@jit
def _update_threshold(scores, contamination):
        """
        Return the quantile set by the contamination of the scores.
        """
        return np.quantile(scores, 1 - contamination)

class iForest(object):
    """
    Create an iForest object holding trees (iTree_array objects) trained on provided data (X).

    Parameters
    ----------
    ntrees : int, default=200
        Number of trees to be used.
    sample_size : int, default=min(256, X.shape[1])
        The size of the subsample to be used in creation of each tree. Must be smaller than |X|.
    limit : int, default=int(np.ceil(np.log2(sample)))
        The maximum allowed tree depth. This is by default set to average length of unsucessful search in a binary tree.
    ExtensionLevel: int, default=X.shape[1]-1
        Specifies degree of freedom in choosing the hyperplanes for dividing up data. Must be smaller than the dimension n of the dataset.    
    random_state: int, default=None
        Controls the pseudo-randomness of the selection of the feature
        and split values for each branching step and each tree in the forest.
        Pass an int for reproducible results across multiple function calls.
    calculate_scores : boolean, default=True
        Flag to calculate the scores used to train the model (not needed if the decision threshold is set manually).
    contamination : float, default=None
        The contamination is the quantile of data that is estimated to be anomalous.
    
    Attributes
    ----------
    trees : numpy array 
        Store tree information
    threshold : int, default=0.5
        Represents the anomaly threshold above which a sample is considered anomalous.
    scores : (numpy array, numpy array), default=None
        Store the index and the score of the train samples if the calculate_scores parameter is set to True.
    dim: int
        Represents the dimension n of the dataset.
   
    Methods
    -------
    fit(X)
        Trains ensemble of trees on data X.
    
    score_samples(X)
        Computes the anomaly scores for data X. 
    """
    def __init__(self, ntrees=200, sample_size=256, limit=None, ExtensionLevel=None, random_state=None, calculate_scores=False, contamination=None):

        # Define random seed.
        if random_state is not None:
            np.random.seed(random_state)
        self.ExtensionLevel = ExtensionLevel
        self.sample_size = sample_size
        self.limit = limit
        self.ntrees = ntrees
        self.calculate_scores = calculate_scores
        self.scores = None
        self.contamination = contamination
        # Define default threshold of 0.5
        self.threshold = 0.5
    
    def update_threshold(self, contamination=None):
        """
        Updates the threshold provided that the scores are calculated the new contamination parameter.
        If no contamination is passed, the threshold is set to 0.5 by default.
        """
        if contamination is None:
            self.contamination = contamination
            self.threshold = 0.5
        elif self.scores is not None:
            self.contamination = contamination
            self.threshold = _update_threshold(self.scores[1], contamination)
        else:
            print("Threshold is set to 0.5. Train samples must be calculated to update the threshold based on contamination.")
        
    def fit_predict(self, X):
        """
        Return array with outlier predictions: normal (False) and outlier (True).
        """
        self.fit(X)
        return self.score_samples(X) > self.threshold
    
    def predict(self, X):
        """
        Return normal (0) and outlier (1) predictions
        """
        return self.score_samples(X) > self.threshold

    def fit(self, X):
        """
        Return iForest trained on data from X. 

        Parameters
        ----------
        X: 2D array (samples, features)
            Data to be trained on.
        """

        # Initialise variables based on X.
        self.sample_size = min(self.sample_size, X.shape[0])
        self.dim = X.shape[1]
        if self.ExtensionLevel is None:
            self.ExtensionLevel = self.dim-1
        
        # 0 < ExtensionLevel < X.shape[1]
        if self.ExtensionLevel < 0 or self.ExtensionLevel >= self.dim:
            raise Exception("Extension level has to be an integer between 0 and %s."%(self.dim-1,))        
        
        # Set limit to the default as specified by the paper (average depth of unsuccesful search through a binary tree).
        if not self.limit:
            self.limit = int(np.ceil(np.log2(self.sample_size)))
        
        # Sample from normal distribution in order to save time later
        maxnodes = 2**(self.limit+1)-1
        self._rng = np.random.default_rng()
        self._normal = np.random.normal(0, 1, size=(self.ntrees, maxnodes, self.dim))
        self._uniform = np.random.uniform(size=(self.ntrees, maxnodes, self.dim))
        if self.dim-self.ExtensionLevel-1:
            self._choice = np.random.choice(self.dim, size=(self.ntrees, maxnodes, self.dim-self.ExtensionLevel-1))
        
        # Populate trees.
        dtype = [("n", "%sf2"%self.dim), ("pdotn", "f2"), ("size", "u2")]
        self.trees = np.zeros((self.ntrees, maxnodes), dtype=dtype)
        idx_sampled = set() 
        for treei in range(self.ntrees): 
            idx = np.random.choice(X.shape[0], self.sample_size, replace=False)
            idx_sampled = idx_sampled.union(idx)
            self.populate_nodes(X[idx], treei)
        idx_sampled = list(idx_sampled)
        
        # Clean-up.
        del self._normal, self._uniform
        if self.dim-self.ExtensionLevel-1: del self._choice

        # Score samples used in the train
        if self.calculate_scores:
            self.scores = (idx_sampled, self.score_samples(X[idx_sampled]))
        
        # Propose decision threshold based on contamination
        self.update_threshold(self.contamination)
        
        return self

    def populate_nodes(self, X, treei, nodei=0, e=0):
        """
        Builds the tree recursively from a given node (e). 
        By default starts from root note (e=0) and make all trees symmetrical. 
        """
        # For terminal nodes store only the size of dataset at final,
        if e==self.limit or len(X)<2:
            self.trees["size"][treei, nodei] = len(X)
        # for internal nodes store everything
        else:
            # A random normal vector picked form a uniform n-sphere. Note that in order to pick uniformly from n-sphere, we need to pick a random normal for each component of this vector.
            n = self._normal[treei, nodei]
            # Pick the indices for which the normal vector elements should be set to zero acccording to the extension level.
            if self.dim-self.ExtensionLevel-1:
                n[self._choice[treei, nodei]] = 0
            # Picking a random intercept point for the hyperplane splitting data.
            mi, ma = minmax(X)
            # Calculating pdotn here will make classification faster and take less space to store.
            pdotn = dot(scale_minmax(self._uniform[treei, nodei], mi, ma), n)
            # Criteria that determines if a data point should go to the left or right child node.
            w = X.dot(n) < pdotn # here X.dot(n) uses BLAS so no need to optimise ;)
            # Store current node,
            self.trees[treei, nodei] = n, pdotn, len(X)
            # split data from X in order to populate left & right nodes
            left, right = split(X, w)
            # and add left & right node.
            self.populate_nodes(left, treei, nodei+1, e+1)
            self.populate_nodes(right, treei, nodei+2**(self.limit-e), e+1)

    def score_samples(self, X):
        """
        Compute anomaly scores for all data points in a dataset X. 

        ----------
        X: 2D array (samples, features)
            Data to be scored on. 

        Returns
        -------
        S: 1D array (X.shape[0])
            Anomaly scores calculated for all samples from all trees. 
        """
        # This will store scores
        S = np.zeros(X.shape[0])
        trees = self.trees
        n, pdotn, sizes = trees["n"], trees["pdotn"], trees["size"]
        # Iterate over samples
        for xi in range(X.shape[0]):
            ni = np.zeros(len(trees), dtype='int')
            w = X[xi].dot(n[:, 0].T) < pdotn[:, 0]
            ni = update_nodes(ni, w, self.limit, 0)
            tidx = np.arange(trees.shape[0])
            for e in range(1, self.limit):
                w = X[xi].dot(n[tidx, ni].T) < pdotn[tidx, ni]
                ni = update_nodes(ni, w, self.limit, e)
                sel = sizes[tidx, ni]>0
                S[xi] += score_false(e, sel)
                tidx, ni = tidx[sel], ni[sel]
            # The size matters only at terminal nodes
            size = sizes[tidx, ni]
            S[xi] += score_terminal(self.limit, ni, size)
        # Calculate anomaly scores
        S = np.power(2, -S / len(trees) / c_factor(self.sample_size))
        return S
