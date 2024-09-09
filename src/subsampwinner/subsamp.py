import numpy as np 
import statsmodels.api as sm 
from statsmodels.stats.multitest import multipletests # multiple test: none, bonferroni, fdr_bh
import logging
from mpi4py import MPI

class subsamp:
    def __init__(self, s, m, qnum=None, seed=None):
        """
        Initialize the SWA (Subsampling Winner Algorithm) object.

        Parameters:
        s (int): The subsample size.
        m (int): The number of subsample repetitions.
        qnum (int): The number of features to select as semifinalists.
        seed (int, optional): Random seed for reproducibility.
        """
        self.s = s
        self.m = m
        self.qnum = s if qnum is None else qnum
        self.seed = seed

        # Initialize attributes that will be set during fitting
        self.X = None
        self.y = None

        self.feature_weights = None # an array of dim p x 1
        self.semifinalists = None  # an array of dim qnum x 1 (selected semi-finalist, index of the features)
        self.finalists = None # an array, selected finalists (index of the features)
        self.p_adjusted = None # an array, adjusted p-values
        self.final_model = None # the final model an OLS model

    # use print(SWA) to get the string representation
    def __repr__(self):
        """String representation of the SWA object."""
        return (f"SWA(s={self.s}, m={self.m}, qnum={self.qnum}, "
                f"seed={self.seed})")

    def _subsample_s_features(self, p, s):
        """
        Subsamples s features from the given p, number of candidate features.

        Parameters:
        p (int): The number of all features.
        s (int): The number of features to subsample.

        Returns:
        array-like: The subsampled feature indices.
        """
        return np.random.choice(p, s, replace=False)
    
    def _analyze_subsample(self, X, y, subsample_indices, weights=None):
        """
        Analyzes a subsample of the data by fitting a base model and computing required statistics.

        Parameters:
            X (numpy.ndarray): The input features.
            y (numpy.ndarray): The response variable.
            subsample_indices (list): The indices of the subsample.
            weights (numpy.ndarray, optional): The weights for each sample. Defaults to None.

        Returns:
            dict: A dictionary containing the subsample indices, t-values, and residual sum of squares (rss).
        """
        p = X.shape[1] # number of all features
        X_s = X[:, subsample_indices]

        # model could be make flexible to allow different base models (more than just OLS)
        model = self._fit_base_model(X_s, y, weights)

        # compute required statistics: t-vals and rss
        t_vals, rss, _ = self._compute_model_statistics(model, p, subsample_indices)

        return {
            'indices': subsample_indices,
            't_vals': t_vals,
            'rss': rss,
        }

    def _fit_base_model(self, X, y, weights=None):
        """
        Fits a base model using Ordinary Least Squares (OLS) or Weighted Least Squares (WLS).

        Parameters:
            X (array-like): The input features.
            y (array-like): The target variable.
            weights (array-like, optional): The weights for the WLS. Defaults to None.

        Returns:
            model: The fitted model.

        """
        X = sm.add_constant(X)
        if weights is None:
            model = sm.OLS(y, X)
        else:
            model = sm.WLS(y, X, weights=weights)
        return model.fit()
    
    def _compute_model_statistics(self, model, p, subsample_indices):
        """
        Compute base model statistics: t-values and residual sum of squares (rss).

        Args:
            model (statsmodels.regression.linear_model.RegressionResultsWrapper): The fitted regression model.
                The model should be an instance of `statsmodels.regression.linear_model.RegressionResultsWrapper`.
            p (int): The number of features in the model.
                This should be an integer representing the number of features in the model.
            subsample_indices (numpy.ndarray): The indices of the selected features.
                This should be a numpy array containing the indices of the selected features.

        Returns:
            tuple: A tuple containing the t-values and residual sum of squares (rss).
                The tuple contains three elements:
                - t_vals (numpy.ndarray): An array of t-values.
                    This array has a length of `p` and contains the t-values for each feature.
                - rss (float): The residual sum of squares.
                    This is a scalar value representing the sum of squared residuals.
                - residuals (numpy.ndarray): An array of residuals.
                    This array has the same length as the number of observations in the model.

        """
        residuals = model.resid
        rss = np.sum(residuals**2)
        t_vals = np.zeros(p) # initialize t_vals with zeros, non-selected features will remain zero
        t_vals[subsample_indices] = model.tvalues[1:] # exclude the intercept
        return t_vals, rss, residuals

    def subsample_anlysis(self, X, y, parallel=False, weights=None):
        """
        Perform subsample analysis on the given data.

        Args:
            X (numpy.ndarray): The input data matrix of shape (n_samples, n_features).
            y (numpy.ndarray): The target values of shape (n_samples,).
            parallel (bool, optional): Whether to perform the analysis in parallel using MPI. Defaults to False.
            weights (numpy.ndarray, optional): The weights for each sample. Defaults to None.

        Returns:
            list: A list of results obtained from analyzing each subsample.
        """
        s = self.s
        m = self.m

        if parallel:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()
            local_m = m // size # local_m is the number of subsamples to analyze per process
            if rank < m % size:
                local_m += 1
        else:
            local_m = m

        p = X.shape[1] # define all features
        results = []

        for _ in range(local_m):
            subsample_indices = self._subsample_s_features(p, s)
            # analyze subsample
            subsample_result = self._analyze_subsample(X, y, subsample_indices, weights=weights)
            results.append(subsample_result)

        if parallel:
            all_results = comm.gather(results, root=0)
            if rank == 0:
                results = [result for sublist in all_results for result in sublist]

        return results

    # process results and select semi-finalist based on subsample_analysis
    def _compute_feature_weights(self, results, option_s=None):
        """
        Compute the feature weights based on the results and option_s.
        w_j = (1/n_j) * sum(1/RSS_{(i)} * abs(t_{(i)j})),
        where n_j = sum(I(t_{(i)j} != 0)) for i = 1 to s, and j = 1 to p.

        Parameters:
            results (list): A list of dictionaries containing the results.
            option_s (int, optional): The number of top sub-models to consider. If None, all sub-models are used.

        Returns:
            numpy.ndarray: An array of feature weights.

        """
        t_vals_matrix = np.array([result['t_vals'] for result in results]) # t_vals_matrix is m x p matrix
        rss_array = np.array([result['rss'] for result in results]) # rss_array is m x 1 array

        if option_s is not None and option_s < len(results):
            # this is the original paper's implementation: use top s sub-models results instead of all
            top_s_indices = np.argsort(rss_array)[:option_s] # sort the top option_s sub-models by rss
            t_vals_matrix = t_vals_matrix[top_s_indices]
            rss_array = rss_array[top_s_indices]

        n_j = np.count_nonzero(t_vals_matrix, axis=0) # n_j is p x 1 array

        # compute the feature weights
        weighted_t_vals = np.abs(t_vals_matrix) / np.sqrt(rss_array[:, np.newaxis]) # m x p matrix
        feature_weights = np.sum(weighted_t_vals, axis=0) # p x 1 array
        # avoid division by zero
        feature_weights = np.divide(feature_weights, n_j, out=np.zeros_like(feature_weights), where=n_j!=0)
        # feature_weights is p x 1 array

        return feature_weights

    def _select_semifinalist(self, results, qnum, use_top_s=False):
        """
        Selects the semifinalist indices based on the results and feature weights.

        Args:
            results (list): A list of dictionaries containing the results.
            qnum (int): The number of semifinalist indices to select.
            use_top_s (bool, optional): Whether to use the top 's' features in the subsample. Defaults to False.

        Returns:
            tuple: A tuple containing the semifinalist indices and the feature weights.
        """
        #option_s = len(results[0]['indices']) if use_top_s else None # s represents the number of features in the subsample
        option_s = self.s if use_top_s else None

        feature_weights = self._compute_feature_weights(results, option_s=option_s)
        semifinalist_indices = np.argsort(feature_weights)[-qnum:][::-1] # select qnum features with the highest weights, in descending order

        return semifinalist_indices, feature_weights
    
    def process_subsample_results(self, results, qnum, use_top_s=False):
        """
        Process the results of subsampling and select semifinalists based on the given criteria.

        Args:
            results (list): The results of subsampling.
            qnum (int): The number of questions.
            use_top_s (bool, optional): Whether to use the top 's' results. Defaults to False.

        Returns:
            tuple: A tuple containing the selected semifinalists and their corresponding feature weights.
        """
        semifinalists, feature_weights = self._select_semifinalist(results, qnum, use_top_s=use_top_s) 

        # revisit later 
        self.feature_weights = feature_weights
        self.semifinalists = semifinalists

        return semifinalists, feature_weights

    # fit final model and output
    def _fit_final_model(self, X, y, semifinalists, weights=None):
        """
        Fits the final model using the selected features.

        Parameters:
            X (array-like): The input features.
            y (array-like): The target variable.
            semifinalists (list): The indices of the selected features.
            weights (array-like, optional): The weights for weighted least squares. Defaults to None.

        Returns:
            model (statsmodels.regression.linear_model.RegressionResultsWrapper): The fitted model.
        """
        X_semif = X[:, semifinalists]
        X_semif = sm.add_constant(X_semif)

        if weights is None:
            model = sm.OLS(y, X_semif)
        else:
            model = sm.WLS(y, X_semif, weights=weights)

        # rename the exogenous variables, v1, v2, ..., vqnum, starting from 1
        model.exog_names[1:] = ['v' + str(index + 1) for index in semifinalists] # assign names to the exogenous variables

        return model.fit()
    
    def _select_finalists(self, final_model, semifinalists, alpha=0.05, method="bonferroni"):
        """
        Selects the finalists based on statistical significance.

        Args:
            final_model (model): The final model used for feature selection.
            semifinalists (list): The list of features considered as semifinalists.
            alpha (float, optional): The significance level. Defaults to 0.05.
            method (str, optional): The method used for multiple testing correction.
                Can be "bonferroni" or "fdr_bh". Defaults to "bonferroni".

        Returns:
            list: The list of features selected as finalists.
            array: The adjusted p-values for each feature.

        """
        p_vals = final_model.pvalues[1:]  # exclude the intercept
        if method == "bonferroni":
            reject, p_adjusted, _, _ = multipletests(p_vals, alpha=alpha, method='bonferroni')
        elif method == "fdr_bh":
            reject, p_adjusted, _, _ = multipletests(p_vals, alpha=alpha, method='fdr_bh')
        else:
            reject = p_vals < alpha 
            p_adjusted = p_vals

        finalist_indices = np.where(reject)[0] # select the indices of the features that are selected as finalists
        finalists = [semifinalists[i] for i in finalist_indices]
        return finalists, p_adjusted

    def select_finalists(self, X, y, semifinalists, weights=None, alpha=0.05, method="bonferroni"):
        """
        Selects the finalists based on the given semifinalists using a final model.

        Parameters:
        - X (array-like): The input features.
        - y (array-like): The target variable.
        - semifinalists (list): The list of semifinalists to consider.
        - weights (array-like, optional): The weights for each sample. Defaults to None.
        - alpha (float, optional): The significance level for hypothesis testing. Defaults to 0.05.
        - method (str, optional): The method for adjusting p-values. Defaults to "bonferroni".

        Returns:
        - finalists (list): The selected finalists.
        - p_adjusted (float): The adjusted p-value.
        - final_model: The final model trained on the selected finalists.
        """
        final_model = self._fit_final_model(X, y, semifinalists, weights=weights)
        finalists, p_adjusted = self._select_finalists(final_model, semifinalists, alpha=alpha, method=method)
        
        self.final_model = final_model
        self.finalists = finalists
        self.p_adjusted = p_adjusted
        
        return finalists, p_adjusted, final_model

    def fit(self, X, y, parallel=False, weights=None, use_top_s=False, alpha=0.05, method="bonferroni"):
        """
        Fits the model to the given data.

        Parameters:
        - X: The input features.
        - y: The target variable.
        - parallel: Whether to perform the subsample analysis in parallel. Default is False.
        - weights: Optional weights for the samples. Default is None.
        - use_top_s: Whether to use the top s features from the subsample analysis. Default is False.
        - alpha: The significance level for feature selection. Default is 0.05.
        - method: The method for adjusting p-values. Default is "bonferroni".

        Returns:
        - finalists: The selected features after the feature selection process.
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        self.X = X
        self.y = y
        qnum = self.qnum

        # 1. Subsample analysis
        results = self.subsample_anlysis(X, y, parallel=parallel, weights=weights)
        # 2. Process results and select semifinalists
        self.semifinalists, self.feature_weights = self.process_subsample_results(results, qnum, use_top_s)
        # 3. Select finalists
        self.finalists, self.p_adjusted, self.final_model = self.select_finalists(
            X, y, self.semifinalists, weights=weights, alpha=alpha, method=method)

        #return print(f"SWA selects the following feature indices: {self.finalists}.")
        return logging.info(f"SWA selects the following feature indices: {self.finalists}.")