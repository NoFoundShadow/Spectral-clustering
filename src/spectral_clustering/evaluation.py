from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

def calculate_ari(y_true, y_pred):
    return adjusted_rand_score(y_true, y_pred)


def calculate_nmi(y_true, y_pred):
    return normalized_mutual_info_score(y_true, y_pred)