from sklearn.decomposition import PCA

def apply_pca(data, n_components=4):
    """
    Apply PCA to reduce the dimensionality of the data.
    """
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    return reduced_data
