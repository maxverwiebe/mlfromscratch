import numpy as np
from itertools import combinations_with_replacement

from ..linear_regression import LinearRegression

class PolynomialRegression(LinearRegression):
    def __init__(self, X, y, degree=2):
        self.degree = degree

        X_poly = self.transform_polynomial(X, degree) # transforms X into polynomial features
        # example: if degree=2 and X has 2 features, X_poly will have 3 features: [1, x1, x2, x1*x2]
        super().__init__(X_poly, y) # calls LinearRegression.__init__

    @staticmethod
    def transform_polynomial(X, degree):
        # transforms X into polynomial features
        # example: if degree=2 and X has 2 features, X_poly will have 3 features: [1, x1, x2, x1*x2]
        X = np.array(X)
        n_samples, n_features = X.shape

        # bias (first col with ones)
        poly_features = [np.ones(n_samples)]

        for deg in range(1, degree + 1):
            # generate combinations of features with replacement for the current degree
            for items in combinations_with_replacement(range(n_features), deg):
                # create polynomial feature by multiplying the selected features
                # e.g., if items=(0, 1) and X has features x1 and x2, then feature = x1 * x2
                feature = np.prod(X[:, items], axis=1)
                poly_features.append(feature)

        return np.vstack(poly_features).T

    def predict(self, X_new):
        # we have to transform the to be predicted X as well
        X_new_poly = self.transform_polynomial(X_new, self.degree)
        return super().predict(X_new_poly)