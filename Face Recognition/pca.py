import numpy as np
#Principal Component Analysis
def PCA(training_DataMatrix, alpha):
    # Step 1: Compute the mean
        #mean will be calculated along the columns (or vertically).
        # 400 row 10304 column
    mean = np.mean(training_DataMatrix, axis=0) 
    # 1 row 10304 column
    
    
    # Step 2: Center the data (m)
    #Aim to making data go toward mean point
    training_data_centralized = training_DataMatrix - mean
    # By centering the data, the first principal component captures 
    #the direction of maximum variance in the data itself, rather than the mean.
    
    # Step 3: Compute the covariance matrix (@ => matrix multiplication)
    cov_matrix = training_data_centralized @ training_data_centralized.T
    #(400, 10304) * (10304,400) = (400, 400)
    print(cov_matrix.shape)
    # X variance of each dimension
    
    # Step 4: Compute the eigenvalues and eigenvectors
        #np.linalg.eig is a function from NumPy that computes the 
        #eigenvalues and eigenvectors of a square matrix.
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    #evalues : array of length 400 => magnitude of variance
    #evectors  2d array (400, 400)=> direction of each pronciple comp.
    #By selecting the top eigenvectors (those with the largest eigenvalues),
    #you can project the data onto a lower-dimensional space.
    #This reduces the dimensionality of the data while retaining most of the variance.
    
    # Step 5: Sort the eigenvectors descendingly by eigenvalues
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx] #highest eigen value
    eigenvectors = eigenvectors[:, idx] #highest eigen vectors (columns rearrange)
    
    # Step 6: Compute the cumulative explained variance ratio
    #This normalization helps interpret how much variance each principal component
    #cumsum => If eigenvalues are [4, 3, 2, 1], then np.cumsum(eigenvalues) would be [4, 7, 9, 10] (make normalized ascendingly)
    #np.sum(eigenvalues) is 10
    normalized_variance_ratio = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    #So, np.cumsum(eigenvalues) / np.sum(eigenvalues) is [0.4, 0.7, 0.9, 1]     X[0.4, 0.3, 0.2, 0.1]
    
    # Step 7: Determine the number of components
    #alpha as a thresholder
    #argmax return index of first occurrence of true
    #+1 adjusts for 0 based indexing to give the actual number of components
    no_components = np.argmax(normalized_variance_ratio >= alpha) + 1
                                #  0    1    2    3     4     5
    #normalized_variance_ratio = [0.4, 0.7, 0.9, 0.95, 0.98, 1.0]
    #alpha = 0.95
    
    # Step 8: Reduce the basis
    #This operation transforms the eigenvectors from the reduced feature space to the original feature space. لحام
    # (400,10304)T * (400* 400) = (10304,400) * (400, 400) = (10304,400)
    eigenvectors_converted = training_data_centralized.T @ eigenvectors
    #This normalization ensures that each eigenvector has unit length in the original feature space.
    eigenfaces = eigenvectors_converted / np.linalg.norm(eigenvectors_converted, axis=0)
    
    # Step 9: Reduce the dimensionality of the data
    projected_data = training_data_centralized @ eigenfaces[:, :no_components]
    
    return mean, eigenfaces[:, :no_components], projected_data