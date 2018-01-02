import matplotlib.pyplot as plt
import numpy as np

from polyFeatures import poly_features


def plot_fit(min_x, max_x, mu, sigma, theta, p):
    """
    plot a polynomial fit
    """    
    x = np.arange(min_x-15, max_x+25, 0.05)
    X_poly = poly_features(x, p)
    X_poly -= mu
    X_poly /= sigma
    X_poly = np.c_[np.ones(x.size), X_poly]
    plt.plot(x, np.dot(X_poly, theta))
