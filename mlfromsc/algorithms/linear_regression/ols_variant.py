import numpy as np

class LinearRegressionOLSVariant:
    """
    A class representing a variant of linear regression using the OLS method.
    """

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.coefficients = None

    def fit(self):
        """
        Fit the linear regression model to the data.
        
        This method computes the coefficients of the linear regression model using the OLS method.
        """
        X = np.array(self.X)
        y = np.array(self.y)

        X_T = X.T
        XTX = X_T @ X
        XTy = X_T @ y

        self.coefficients = np.linalg.inv(XTX) @ XTy
        return self.coefficients
    
    def predict(self, X_new):
        """Predict the target variable for new data points.
        """
        X_new = np.array(X_new)
        return X_new @ self.coefficients