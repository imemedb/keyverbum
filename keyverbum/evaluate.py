import numpy as np


def evaluate(true_kws, predicted_kws):
    assert len(true_kws) == len(predicted_kws)

    precisions = []
    recalls = []
    f1s = []
    jaccards = []

    for i in range(len(true_kws)):
        true_kw = set(true_kws[i])
        predicted_kw = set(predicted_kws[i])

        tp = len(true_kw & predicted_kw)
        union = len(true_kw | predicted_kw)
        fp = len(predicted_kw - true_kw)
        fn = len(true_kw - predicted_kw)

        if (tp + fp) == 0:
            prec = 0
        else:
            prec = tp / (tp + fp)

        if (tp + fn) == 0:
            rec = 0
        else:
            rec = tp / (tp + fn)
        if (prec + rec) == 0:
            f1 = 0
        else:
            f1 = (2 * (prec * rec)) / (prec + rec)

        jac = tp / union

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        jaccards.append(jac)
    print("Precision - ", round(np.mean(precisions), 2))
    print("Recall - ", round(np.mean(recalls), 2))
    print("F1 - ", round(np.mean(f1s), 2))
    print("Jaccard - ", round(np.mean(jaccards), 2))
