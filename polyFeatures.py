import numpy as np


def poly_features(X, p):
    """
    plot a polynomial fit
    """    
    # You need to return the following variable correctly.
    X_poly = np.zeros((X.size, p))

    # ===================== Your Code Here =====================
    # Instructions : Given a vector X, return a matrix X_poly where the p-th
    #                column of X contains the values of X to the p-th power.
    P = np.arange(1, p+1)
    X_poly = X.reshape((X.size, 1))**P

    # ==========================================================

    return X_poly
