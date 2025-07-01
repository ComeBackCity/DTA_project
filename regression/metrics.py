import numpy as np
from sklearn.metrics import roc_auc_score, cohen_kappa_score

# def get_cindex(gt, pred):
#     gt_mask = gt.reshape((1, -1)) > gt.reshape((-1, 1))
#     diff = pred.reshape((1, -1)) - pred.reshape((-1, 1))
#     h_one = (diff > 0)
#     h_half = (diff == 0)
#     CI = np.sum(gt_mask * h_one * 1.0 + gt_mask * h_half * 0.5) / (np.sum(gt_mask) + 1e-9)

#     return CI

def get_cindex(Y, P):
    """
    Computes the Concordance Index (CI) for predictions.

    Args:
        Y (np.ndarray): Ground truth values (1D)
        P (np.ndarray): Predicted values (1D)

    Returns:
        float: Concordance Index
    """
    Y = np.asarray(Y).flatten()
    P = np.asarray(P).flatten()
    
    # Pairwise difference matrices
    diff_Y = Y[:, None] - Y[None, :]
    diff_P = P[:, None] - P[None, :]

    # Valid pairs: Y[i] > Y[j]
    valid = diff_Y > 0

    # Concordant: P[i] > P[j]
    concordant = (diff_P > 0)[valid]

    # Ties: P[i] == P[j]
    ties = (diff_P == 0)[valid]

    concordant_score = np.sum(concordant) + 0.5 * np.sum(ties)
    total_pairs = np.sum(valid)

    return concordant_score / total_pairs if total_pairs > 0 else 0.0


def get_cindex2(Y, P):
    """
    Vectorized concordance index (C-index) computation.

    Args:
        Y (array-like): Ground truth labels. Shape: (N,)
        P (array-like): Predicted values. Shape: (N,)

    Returns:
        float: C-index score.
    """
    Y = np.asarray(Y)
    P = np.asarray(P)

    # Create pairwise differences
    diff_Y = Y[:, None] - Y[None, :]
    diff_P = P[:, None] - P[None, :]

    # Only consider pairs where Y[i] > Y[j] (strictly)
    valid_mask = diff_Y > 0

    # Concordant: P[i] > P[j]
    concordant = (diff_P > 0) & valid_mask
    ties = (diff_P == 0) & valid_mask

    summ = np.sum(concordant) + 0.5 * np.sum(ties)
    pair = np.sum(valid_mask)

    return summ / pair if pair != 0 else 0.0

def ci(y, f):
    # Sort y and f based on sorted order of y
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    
    # Create all pairwise comparisons using numpy broadcasting
    diff_y = y[:, None] - y[None, :]  # y[i] - y[j] for all i, j
    diff_f = f[:, None] - f[None, :]  # f[i] - f[j] for all i, j

    # Only consider pairs where y[i] > y[j]
    valid_pairs = diff_y > 0

    # Calculate the concordant pairs
    S = np.sum((diff_f > 0) * valid_pairs) + 0.5 * np.sum((diff_f == 0) * valid_pairs)
    z = np.sum(valid_pairs)
    
    # Calculate C-index
    ci = S / z if z > 0 else 0.0
    return ci

def ci2(y,f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci

def get_k(y_obs,y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs*y_pred) / float(sum(y_pred*y_pred))

def squared_error_zero(y_obs,y_pred):
    k = get_k(y_obs,y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k*y_pred)) * (y_obs - (k* y_pred)))
    down= sum((y_obs - y_obs_mean)*(y_obs - y_obs_mean))

    return 1 - (upp / float(down))


def get_rm2(ys_orig,ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)

    return r2 * (1 - np.sqrt(np.absolute((r2*r2)-(r02*r02))))

def r_squared_error(y_obs,y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean)*(y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean) )

    return mult / float(y_obs_sq * y_pred_sq)

def positive(y_true):
    return np.sum((y_true == 1))

def negative(y_true):
    return np.sum((y_true == 0))

def true_positive(y_true, y_pred):
    return np.sum(np.bitwise_and(y_true == 1, y_pred == 1))

def false_positive(y_true, y_pred):
    return np.sum(np.bitwise_and(y_true == 0, y_pred == 1))

def true_negative(y_true, y_pred):
    return np.sum(np.bitwise_and(y_true == 0, y_pred == 0))

def false_negative(y_true, y_pred):
    return np.sum(np.bitwise_and(y_true == 1, y_pred == 0))

def accuracy(y_true, y_pred):
    sample_count = 1.
    for s in y_true.shape:
        sample_count *= s

    return np.sum((y_true == y_pred)) / sample_count

def sensitive(y_true, y_pred):
    tp = true_positive(y_true, y_pred)
    p = positive(y_true) + 1e-9
    return tp / p

def specificity(y_true, y_pred):
    tn = true_negative(y_true, y_pred)
    n = negative(y_true) + 1e-9
    return tn / n

def precision(y_true, y_pred):
    tp = true_positive(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    return tp / (tp + fp)

def recall(y_true, y_pred):
    tp = true_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)
    return tp / (tp + fn)

def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    reca = recall(y_true, y_pred)
    fs = (2 * prec * reca) / (prec + reca)
    return fs

auc_score = roc_auc_score
kappa_score = cohen_kappa_score

if __name__ == '__main__':
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 1])
    y_pred = np.array([1, 1, 0, 1, 0, 1, 0, 1])

    sens = sensitive(y_true, y_pred)
    spec = specificity(y_true, y_pred)
    prec = precision(y_true, y_pred)
    reca = recall(y_true, y_pred)
    fs = f1_score(y_true, y_pred)

    print(sens)
    print(spec)
    print(prec)
    print(reca)
    print(fs)

# %%
