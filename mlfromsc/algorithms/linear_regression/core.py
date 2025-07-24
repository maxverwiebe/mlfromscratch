import numpy as np

class LinearRegression:
    """
    A class representing a variant of linear regression using the Gradient Descent approach.
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.coefficients = None

        self.loss_history = None

    def use_loss_history(self, use_loss_history=False):
        """Activate the loss history to track the loss during training."""
        if use_loss_history:
            self.loss_history = []
        else:
            self.loss_history = None

    def fit(self, learning_rate=0.01, epochs=1000):
        """
        Fit the linear regression model to the data.
        
        This method computes the coefficients of the linear regression model using the GD method.
        """
        X = np.array(self.X)
        y = np.array(self.y)

        theta = np.zeros(X.shape[1])

        if self.loss_history is not None:
            self.loss_history = []

        for epoch in range(epochs):
            y_pred = X @ theta

            error = y_pred - y.flatten()
            
            gradient = (1 / X.shape[0]) * (X.T @ error)

            theta = theta - learning_rate * gradient

            loss = np.mean(error**2)

            if self.loss_history is not None:
                self.loss_history.append(loss)
                
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")

        self.coefficients = np.array(theta)

    def predict(self, X_new):
        """Predict the target variable for new data points.
        """
        X_new = np.array(X_new)
        return X_new @ self.coefficients
    
    def r2_score(self, X_test, y_test):
        """Calculate the R^2 score for the model."""
        y_test = y_test.flatten()

        y_pred = self.predict(X_test)
        ss_total = np.sum((y_test - np.mean(y_test)) ** 2)
        ss_residual = np.sum((y_test - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)
