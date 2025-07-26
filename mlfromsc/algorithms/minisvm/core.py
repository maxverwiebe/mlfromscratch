import numpy as np

class MiniSVM:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.loss_history = None

    def use_loss_history(self, use_loss_history=False):
        """Activate the loss history to track the loss during training."""
        if use_loss_history:
            self.loss_history = []
        else:
            self.loss_history = None

    # helper function to calculate the loss (static?)
    def _loss(self, X, y, learning_rate=0.01):
        """Primal objective: ½λ‖w‖² + ⟨hinge⟩."""
        y_ = np.where(y <= 0, -1, 1)
        hinge = np.maximum(0, 1 - y_ * (X @ self.w + self.b)).mean()
        l2    = 0.5 * learning_rate * np.dot(self.w, self.w)
        return l2 + hinge

    def fit(self, learning_rate=0.01, epochs=1000):
        # convert labels to -1 and 1
        y_ = np.where(self.y <= 0, -1, 1)
        n, d = self.X.shape
        
        self.w = np.zeros(d)
        self.b = 0.0
        step_count = 0
        rng = np.random.default_rng()
        
        if self.loss_history is not None:
            self.loss_history = []

        for epoch in range(epochs):
            for _ in range(n): # process each sample in random order
                step_count += 1
                i = rng.integers(n)
                x_i, y_i = self.X[i], y_[i] # random sample
                
                current_lr = 1.0 / (learning_rate * step_count) # adaptive learning rate
                
                margin = y_i * (x_i @ self.w + self.b) # calcs the margin to the support vector line / hyperplane
                
                # update rule based on whether the margin is less than 1
                if margin < 1:
                    # if misclassified or within margin, update both w and b
                    self.w = (1 - current_lr * learning_rate) * self.w + current_lr * y_i * x_i
                    self.b += current_lr * y_i
                else:
                    # if correctly classified, only apply weight decay
                    self.w = (1 - current_lr * learning_rate) * self.w

            epoch_loss = self._loss(self.X, self.y, learning_rate=learning_rate)

            # Track loss if history is enabled
            if self.loss_history is not None:
                self.loss_history.append(epoch_loss)

            print(f"Epoch {epoch+1:4d}/{epochs}, Loss: {epoch_loss:.4f}")

    def predict(self, X):
        scores = X @ self.w + self.b
        return np.where(scores >= 0, 1, -1)