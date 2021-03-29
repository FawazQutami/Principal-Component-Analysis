# File name: PCA.py

import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


class PCA:
    """ Principal Component Analysis """
    """
     Principal Component Analysis, or PCA, is a dimensionality-reduction method that is often used to 
     reduce the dimensionality of large data sets, by transforming a large set of variables into a 
     smaller one that still contains most of the information in the large set.
    
     PCA is used in exploratory data analysis and for making predictive models. 
     It is commonly used for dimensionality reduction by projecting each data point onto 
     only the first few principal components to obtain lower-dimensional data while preserving as 
     much of the data's variation as possible.
     
     The principal components are eigenvectors of the data's covariance matrix.
     Thus, the principal components are often computed by:
            - EVD: Eigenvalue decomposition of the data covariance matrix or 
            - SVD: Singular Value Decomposition of the data matrix.
    """

    def __init__(self, n_components=2):
        self.n_components = n_components
        self.Feature_vector = None
        self.eigenvalues = None
        #self.mean = None
        #self.sta = None

    def fit_transform(self, X):
        # STEP 1: STANDARDIZATION OF FEATURE DATA:
        scaled_features = scaling_data(X)  # X = (X - self.mean)/self.sta

        # STEP 2: COMPUTE THE COVARIANCE MATRIX
        """        
            Cov(X, X) = 1/N ∑i=1:n (xi - X_bar)((xi - X_bar).T)  
                     
            Covariance provides the a measure of strength of correlation between two variable 
            or more set of variables. 
                If COV = 0 then variables are uncorrelated
                If COV > 0 then variables positively correlated
                If COV > < 0 then variables negatively correlated
        
        # np.cov() function: estimate a covariance matrix, given data and weights.
            Each row of the matrix represents a variable, and each column a single observation of 
            all those variables but, X is (n_samples, n_features) matrix that need to be transposed 
            to X.T (n_features, n_samples)"""
        transposed_features = scaled_features.T
        cov_matrix = np.cov(transposed_features)

        # STEP 3 Eigen-decomposition: COMPUTE THE EIGENVECTORS AND EIGENVALUES OF THE COVARIANCE MATRIX TO IDENTIFY THE
        # PRINCIPAL COMPONENTS
        """
            Eigendecomposition: is a process that decomposes a square matrix into eigenvectors and eigenvalues. 
            Eigenvectors are simple unit vectors, and eigenvalues are coefficients which give the magnitude to 
            the eigenvectors.
         
            The eigenvectors points in the same direction of the maximum variation between 2 variables,
            and the associated eigenvalues indicates the importance of it's eigenvectors
            
            The normalized (unit “length”) eigenvectors, such that the column v[:,i] is the eigenvector 
            corresponding to the eigenvalue w[i].
        
        # linalg.eig(A): Compute the eigenvalues and right eigenvectors of A square array."""
        self.eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        """ 
        # Transposed the vector column(eigenvectors) for easier transformation"""
        eigenvectors_transposed = eigenvectors.T
        """  
        # Sort eigenvalues in descendant order:
           Observe that the big elements are coming last in the argsort. 
           So, you can read from the tail of the argsort """
        indices = np.argsort(self.eigenvalues)[::-1]
        """
        # Choose eigenvectors according to the new descendant-order eigenvalues"""
        # new_eigenvalues = self.eigenvalues[indices]
        new_eigenvectors = eigenvectors_transposed[indices]

        # STEP 4: CHOOSE THE FEATURE VECTOR
        """
            Choose whether to keep all these components or discard those of lesser significance
            (of low eigenvalues), and form with the remaining ones a matrix of vectors that
            we call Feature vector.
            This is the first step towards dimensionality reduction, because if we choose to keep
            only p eigenvectors (components) out of n, the final data set will have only p dimensions.

        # Store only p eigenvectors as new components"""
        self.Feature_vector = new_eigenvectors[0:self.n_components]

        # FINALLY: PROJECT THE DATA ALONG THE PRINCIPAL COMPONENTS AXES
        """
            Project the data according to the new dimension - note that we have to transpose the Feature_vector
            as we already transposed the vector column of eigenvectors"""
        projected_vectors = np.dot(scaled_features, self.Feature_vector.T)

        return projected_vectors


def plot_pca(pca_x, y):
    x1 = pca_x[:, 0]
    x2 = pca_x[:, 1]

    fig = plt.figure()
    plt.scatter(x1, x2,
                c=y
                , marker='o'
                , s=50
                , cmap=plt.cm.get_cmap('viridis', 3)
                , label='PC 1 vs 2')

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.show()


def explained_variance(e_values):
    e_variances = []
    for i in range(len(e_values)):
        ev = e_values[i] / np.sum(e_values)
        e_variances.append(ev)
    # The first value is just the sum of explained variances — and must be equal to 1
    print('\n-- The sum of explained variances — and must be equal to 1:\n\t', np.sum(e_variances))
    # The second value is an array, representing the explained variance percentage per principal component.
    ev = [format(x * 100, '.3f') for x in e_variances]
    print('-- Explained variance percentage per principal component:\n\t', ev)


def scaling_data(x):
    """ Scaling or standardizing our training and test data """
    """
        -- Data standardization is the process of rescaling the attributes so that they have 
            mean as 0 and variance as 1.
        -- The ultimate goal to perform standardization is to bring down all the features to 
            a common scale without distorting the differences in the range of the values.
        -- In sklearn.preprocessing.StandardScaler(), centering and scaling happens independently 
            on each feature.
    """
    scaler = StandardScaler()  # Standardize features by removing the mean and scaling to unit variance
    """
        The "fit method" is calculating the mean and variance of each of the features present in our data. 
        The "transform" method is transforming all the features using the respective mean and variance.
    """
    scaled_x = scaler.fit_transform(x)

    return scaled_x


def get_data(data):
    # Create a pandas data frame from iris data
    iris_df = pd.DataFrame(data.data, columns=data.feature_names)
    # Append a target to the data frame
    iris_df['target'] = data.target
    # Create X and y by dropping some columns from iris data frame
    x = iris_df.drop(['target'], axis='columns')
    y = iris_df['target']

    return x, y


if __name__ == '__main__':
    # Load Iris dataset
    iris = datasets.load_iris()
    # Get features and target
    features, target = get_data(iris)

    # Project the data onto the 2 primary principal components
    components = 2
    pca = PCA(n_components=components)
    X_projected = pca.fit_transform(features)

    # print the explained variances
    explained_variance(pca.eigenvalues)

    print('--Dimensionality-reduction: ----------')
    print('\tShape of X:', features.shape)
    print('\tShape of transformed X:', X_projected.shape)

    # Create a Principal Component dataframe
    print('-- Principal Component information:')
    res = pd.DataFrame(X_projected, columns=['PC' + str(i + 1) for i in range(components)])
    res['flower'] = target.apply(lambda x: iris.target_names[x])
    print(res.head())

    import seaborn as sns
    plt.figure(figsize=(8, 6))
    sns.scatterplot(res['PC1'], [0] * len(res), hue=res['flower'], s=200)
    plt.show()

    plot_pca(X_projected, target)
