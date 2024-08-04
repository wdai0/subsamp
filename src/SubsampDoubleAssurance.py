import math
import numpy as np

class SubsampDoubleAssurance:
    def __init__(self, m, qnum=None, seed=None):
        #super().__init__(s=None, m=m, qnum=qnum, seed=seed) # better to use composition than inheritance
        self.m = m
        self.qnum = qnum
        self.seed = seed

    @staticmethod
    def calculate_individual_stability(individual_set, union_set):
        return len(individual_set) / len(union_set)

    def double_assurance(self, X, y, s0, T=0.8, I_max=10, init_range = 0.3, r = 0.75, weights=None, alpha=0.05, method="bonferroni", return_objects=False):
        """
        Perform the Double Assurance Procedure with Adaptive Range and Stability-Based Adjustment.

        This method implements an iterative process to find a stable subsample size for feature selection
        using the Subsampling Winner Algorithm (SWA). It adapts the range of candidate subsample sizes
        and adjusts based on stability scores.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,)
            The target values.
        s0 : int
            Initial subsample size.
        T : float, optional (default=0.8)
            Stability threshold for early stopping.
        I_max : int, optional (default=10)
            Maximum number of iterations.
        init_range : float, optional (default=0.3)
            Initial range factor for generating candidate subsample sizes.
        r : float, optional (default=0.75)
            Shrinkage rate for the range factor.
        weights : array-like of shape (n_samples,), optional (default=None)
            Individual weights for each sample.
        alpha : float, optional (default=0.05)
            Significance level for feature selection.
        method : str, optional (default="bonferroni")
            Method for p-value adjustment in multiple testing.
        return_objects : bool, optional (default=False)
            If True, return a dictionary of objects instead of printing results.

        Returns:
        --------
        dict or None
            If return_objects is True, returns a dictionary containing:
            - 'best_s0': Best subsample size found
            - 'final_range': Final range of subsample sizes [s1, s0, s2]
            - 'stabilized_union_set': Set of all features selected across all iterations
            If return_objects is False, prints the results and returns None.

        Notes:
        ------
        The method adapts the range of candidate subsample sizes in each iteration,
        aiming to find a stable set of features. It terminates when either the stability
        threshold is reached, the maximum number of iterations is exceeded, or the
        range of subsample sizes converges to a single value.

        The stability score is calculated as the intersection of features selected by
        three different subsample sizes, divided by their union.

        Example:
        --------
        >>> sda = SubsampDoubleAssurance(m=1000, qnum=20)
        >>> sda.double_assurance(X, y, s0=50, T=0.85, I_max=15, init_range=0.4, r=0.8)
        """
        def run_swa(s):
            swa = subsamp(s=s, m=self.m, qnum=self.qnum, seed=self.seed)
            swa.fit(X, y, weights=weights, alpha=alpha, method=method)
            return set(swa.finalists)

        best_s0 = s0
        best_stability = 0
        stabilitized_set = set()

        for iteration in range(I_max):
            # update range (range of subsample size)
            alpha_iter = init_range * r**iteration
            s1 = math.floor((1 - alpha_iter) * s0)
            s2 = math.ceil((1 + alpha_iter) * s0)

            # break if all subsample sizes are the same
            if s1 == s2:
                print(f"Convergence reached at iteration {iteration}, with s1 = s0 = s2 = {s0}.")
                break
        
            # evaluate performance of each subsample size
            final_set0 = run_swa(s0)
            final_set1 = run_swa(s1)
            final_set2 = run_swa(s2)

            # create union set
            final_set_union = final_set0.union(final_set1).union(final_set2)
            stabilitized_set.update(final_set_union)

            stability_score = len(final_set0.intersection(final_set1).intersection(final_set2)) / len(final_set_union)

            if stability_score > best_stability:
                best_stability = stability_score
                best_s0 = s0

            if stability_score >= T:
                print(f"Stability threshold reached. Terminating at iteration {iteration + 1}.")
                break

            # calculate individual contributions
            stability_0, stability_1, stability_2 = [
               self.calculate_individual_stability(final_set, final_set_union) 
               for final_set in [final_set0, final_set1, final_set2]
            ]

            # update s0 for the best individual stability for the next iteration
            s0 = [s0, s1, s2][np.argmax([stability_0, stability_1, stability_2])]

        if return_objects:
            return {
                'best_s0': best_s0,
                'final_range': [s1, s0, s2],
                'stabilized_union_set': stabilitized_set,
            }
        else:
            print("\nFinal Results:")
            print(f"Best subsample size (s0): {best_s0}")
            print(f"Best stability score: {best_stability:.4f}")
            print(f"Final range: s1 = {s1}, s0 = {s0}, s2 = {s2}")
            print(f"Number of features in stabilized union set: {len(stabilitized_set)}")
            print(f"Final selected features: {stabilitized_set}")
            return None
