from sklearn.preprocessing import StandardScaler

def standardize_features(X):
    return StandardScaler().fit_transform(X)