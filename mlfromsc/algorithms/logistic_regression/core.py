import numpy as np

class LogisticRegression:
    """
    A class representing a logistic regression model using the Gradient Descent approach.
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
        Fit the logistic regression model to the data.

        This method computes the coefficients of the logistic regression model using the GD method.
        """
        X = np.array(self.X)
        y = np.array(self.y)

        theta = np.zeros(X.shape[1])

        if self.loss_history is not None:
            self.loss_history = []

        def _sigmoid(z):
            return 1 / (1 + np.exp(-z))

        for epoch in range(epochs):
            z = X @ theta
            y_pred = _sigmoid(z)

            eps   = 1e-15
            p     = np.clip(y_pred, eps, 1 - eps)
            loss  = -np.mean(y*np.log(p) + (1-y)*np.log(1-p))
            
            gradient = (1 / X.shape[0]) * (X.T @ (y_pred - y))

            theta = theta - learning_rate * gradient

            loss = np.mean(loss**2)

            if self.loss_history is not None:
                self.loss_history.append(loss)
                
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")

        self.coefficients = np.array(theta)

    def predict_proba(self, X_new):
        """Predict the probability of the positive class for new data points.
        """
        X_new = np.array(X_new)
        z = X_new @ self.coefficients
        return 1 / (1 + np.exp(-z))
    
    def predict(self, X_new):
        """Predict the class labels for new data points."""
        probabilities = self.predict_proba(X_new)
        return (probabilities >= 0.5).astype(int)

    def accuracy(self, X_test, y_test):
        """Calculate the accuracy of the model on the test data."""
        return (self.predict(X_test) == y_test.flatten()).mean()

