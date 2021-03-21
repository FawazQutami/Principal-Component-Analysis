import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


class PCA:
    """ Principal Component Analysis """

    """ 
    The principal components are eigenvectors of the data's covariance matrix.
    Thus, the principal components are often computed by:
            - EVD: Eigenvalue decomposition of the data covariance matrix or 
            - SVD: Singular Value Decomposition of the data matrix.
    """

    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):

        # Calculate the mean for each feature in X
        #           --> Mean = X_bar = 1/n ∑i=1:n xi
        self.mean = np.mean(X, axis=0)

        # Subtract the mean of each variable (feature) from the dataset to center the data around the origin
        X = X - self.mean

        # Compute the covariance matrix of the data
        """
             Cov(X, X) = 1/N ∑i=1:n (xi - X_bar)((xi - X_bar).T)  
                     
        Covariance provides the a measure of strength of correlation between two variable or more set of variables. 
                If COV = 0 then variables are uncorrelated
                If COV > 0 then variables positively correlated
                If COV > < 0 then variables negatively correlated
        """
        """
        np.cov() function: estimate a covariance matrix, given data and weights.
        Each row of the matrix represents a variable, and each column a single observation of all those variables
        But, X is (n_samples, n_features) matrix that need to be transposed to X.T (n_features, n_samples)
        """
        cov_matrix = np.cov(X.T)

        # Calculate the eigenvalues and corresponding eigenvectors of this covariance matrix
        """ 
            The eigenvectors points in the same direction of the maximum variation between 2 variables,
            and the associated eigenvalues indicates the importance of it's eigenvectors
        
            The eigenvectors of the covariance matrix scaled by the square root of the corresponding 
            eigenvalue, and shifted so their tails are at the mean.
            """

        # linalg.eig(a): Compute the eigenvalues and right eigenvectors of a square array.
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        """
            The normalized (unit “length”) eigenvectors, such that the column v[:,i] is the eigenvector 
            corresponding to the eigenvalue w[i].
            """
        # Transpose eigenvectors for easier transformation
        eigenvectors_transposed = eigenvectors.T

        # Sort eigenvectors according to their respective eigenvalues - in descendant order
        """ 
            Observe that the big elements are coming last in the argsort. 
            So, you can read from the tail of the argsort 
            """
        indices = np.argsort(eigenvalues)[::-1]

        # Choose the new eigenvectors according to the new indices (dimension)
        new_eigenvalues = eigenvalues[indices]
        new_eigenvectors = eigenvectors_transposed[indices]

        # Store n eigenvectors as new components
        self.components = new_eigenvectors[0:self.n_components]

    def transform(self, X):

        # Subtract the mean of each variable (feature) from the dataset to center the data around the origin
        X = X - self.mean
        # Project the data according to the new dimension - note that we have to transpose the components
        # as we already transposed the vector column
        return np.dot(X, self.components.T)


if __name__ == '__main__':
    # data = datasets.load_digits()
    data = datasets.load_iris()
    X = data.data
    y = data.target

    # Project the data onto the 2 primary principal components
    pca = PCA(2)
    pca.fit(X)
    X_projected = pca.transform(X)

    print('Shape of X:', X.shape)
    print('Shape of transformed X:', X_projected.shape)

    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]

    plt.scatter(x1, x2,
                c=y, edgecolor='none', alpha=0.8,
                cmap=plt.cm.get_cmap('viridis', 3))

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar()
    plt.show()
