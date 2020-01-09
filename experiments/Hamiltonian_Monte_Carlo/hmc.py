#%%
import numpy as np
from scipy.stats import multivariate_normal


class HMC(object):


    def __init__(self):
        """
        HMC sampler
        """
        pass

    def sample_multivariate(self, x_0, p_mean, M, mean, sigma, iterations, random_state, step=2, eps=0.1, warm_start=5000):
        """
        Sampling of normal multivariate probability law N(mean, sigma)
        
        Inputs:
        --------
        x_0 (array): sarting point
        p_mean (array): mean of auxiliary random variable \sim N(p_mean, M)
        M (array): mass matrix i.e. covriance of auxiliary random variable \sim N(p_mean, M)
        mean (array): mean of target distribution \sim N(mean, sigma)
        sigma (array): covariance matrix of target distribution
        iterations (int): number of iterations
        random_state (RandomState): for reproductibility
        step (int): number of steps in leapfrog integrator
        eps (float): epsillon in leapfrog integrator
        warm_start (int): Number of iterations for warm-up (ie before sampling)
        
        Outputs:
        --------
        val (array): Samples
        warm (array): Samples throughout warm-up
        """
        val = []
        warm = []
        x = x_0
        d = x.shape[0]

        sigma_inv = np.linalg.inv(sigma)
        M_inv = np.linalg.inv(M)


        for i in range(iterations):
            y = multivariate_normal(mean=p_mean[:, 0], cov=M).rvs().reshape(d, 1)
            x_prec = x
            y_prec = y
            for _ in range(step):

                # Compute (y(tau + eps / 2))
                y_2 = self.__proposal_y_2(x_prec, y_prec, mean, sigma_inv, eps).reshape(d, 1)
                x_prop = self.__proposal_x(x_prec, y_prec, mean, sigma_inv, eps, M_inv).reshape(d, 1)
                y_prop = self.__proposal_y(x_prop, y_2, mean, sigma_inv, eps).reshape(d, 1)

                x_prec = x_prop
                y_prec = y_prop

            alpha = min(1, np.exp(- self.__hamilton(x_prop, y_prop, mean, sigma, M) + self.__hamilton(x, y, mean, sigma, M))[0, 0])
            if random_state.rand() < alpha:
                if i  > warm_start:
                    val.append(x_prop.copy())

                if i < warm_start:
                    warm.append(x_prop.copy())

                x = x_prop.copy()
                y = y_prop.copy()

        return val, warm


    def __log_normal(self, x, mean, sigma):
        """
        Computes log_normal proba
        """

        return np.log(multivariate_normal.pdf(x[:, 0], mean[:, 0], sigma))

    def __hamilton(self, x, y, mean, sigma, M):

        return - self.__log_normal(x, mean, sigma) + 0.5 * np.log((2 * np.pi) ** (x.shape[0]) * np.linalg.det(M)) + 0.5 * np.dot(np.dot(y.T, np.linalg.inv(M)), y)


    def __nabla_log_normal(self, x, mean, sigma_inv):
        return np.dot(sigma_inv, (x - mean))

    def __proposal_x(self, x, y_2, mean, sigma_inv, eps, M_inv):
    
        x = x + eps * np.dot(M_inv, y_2)
        return x

    def __proposal_y_2(self, x, y, mean, sigma_inv, eps):
        y_2 = y + eps * self.__nabla_log_normal(x, mean, sigma_inv) / 2
        return y_2

    def __proposal_y(self, x, y_2, mean, sigma_inv, eps):
    
        y = y_2 + eps * self.__nabla_log_normal(x, mean, sigma_inv) / 2
        return y