import numpy as np

class Perceptron:
    """
    Perceptron classifer for binary classifcation
    
    Parameters:
        learning_rate (float): step size for weight updates
        epochs (int): number of iterations over the dataset
        
    Attributes:
        weights (np.ndarray): learned weights for input features 
            where bias is the last term
        errors_: number of misclassifications in an epoch
    """
    def __init__(self, learning_rate=.5, epochs=500):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights_ = None
        self.errors_ = []
        np.random.seed(777)
        
    def train(self, X, y):
        """
        Trains the Perceptron model

        Args:
            X (_type_): np.ndarray of shape (n_samples, n_features) Input features
            y (_type_): np.ndarray of shape (n_samples, ) Binary targets of -1 or 1
            
        Returns:
            self: perceptron object with updated weights
        """
        self.weights_ = np.random.randn(1 + X.shape[1])
        self.errors_ = []
        
        # main training loop
        for epoch in range(self.epochs):
            errors = 0
            
            # Iterate throught he samples one by one
            for xi, target in zip(X, y):
                update = self.learning_rate * (target - self.predict(xi))
                
                # weight update
                self.weights_[:-1] -= update * xi
                self.weights_[-1] -= update
                
                errors += int(update != 0)
                
            self.errors_.append(errors)
            
            if errors == 0:
                print(f"Converged at epoch {epoch + 1}")
                return self

        return self
        
    def net_input(self, X):
        """
        Compute the linear combination of inputs and weights.

        Args:
            X : np.ndarray of shape (n_features,) or (n_samples, n_features)
            Input feature vector(s).

        Returns:
            float or np.ndarray of Weighted sum plus bias term.
        """
        return np.dot(X, self.weights_[:-1]) + self.weights_[-1]
    
    def predict(self, X):
        """
        Predict class labels for input samples.

        Parameters
        ----------
        X : np.ndarray of shape (n_features,) or (n_samples, n_features)
            Input features.

        Returns
        -------
        int or np.ndarray
            Predicted class labels (-1 or 1).
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)