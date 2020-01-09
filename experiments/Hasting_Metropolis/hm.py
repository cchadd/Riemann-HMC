import numpy as np
from scipy.stats import multivariate_normal

class HM(object):

    def __init__(self):
        """
        Metropolis Sampler
        """
        pass



    def sample(self, x0, target_dist, iterations, random_state, warm_start=5000):
        """
        Sampling of a target distribution

        Inputs:
        -------
        x_0 (array): sarting point
        target_dist (func): the distribution we want to sample from
        iterations (int): Number of iterations
        random_state (RandomState)
        warm_start (int): Number of iterations for warm-up (ie before sampling)
        """
        x = x0
        samples = []
        warm = []
        d  = len(x0)

        for i in range(iterations):
            x_star = np.array(x) + np.array(random_state.multivariate_normal(mean=[0 for _ in range(d)], cov=np.eye(d)))
            if random_state.rand() < target_dist(x_star) / target_dist(x):
                x = x_star
                if i > warm_start:
                    samples.append(np.array(x).copy())

                if i < warm_start:
                    warm.append(np.array(x).copy())

        return samples, warm