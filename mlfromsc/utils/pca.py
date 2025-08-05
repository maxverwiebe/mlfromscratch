import numpy as np

class PCA:
    def __init__(self, target_dimensions = 2):
        self.target_dimensions = target_dimensions
        
    def center_data(self, X):
        # each col - mean value
        return X - np.mean(X, axis=0)
    
    def fit(self, X):
        # calculates the covarian matrices
        # rowvar=False column by column (each column = feature)
        cov_matrix = np.cov(X, rowvar=False)
        
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # sort by descending variance
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]
        
        top_k_eigenvectors = sorted_eigenvectors[:, :self.target_dimensions]
        
        # projection
        result = X @ top_k_eigenvectors
        
        return result
