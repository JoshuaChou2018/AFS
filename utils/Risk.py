import pandas as pd
import numpy as np
import pickle
from math import sqrt

def Performance(predict, y_test):
    tp = 0.0
    tn = 0.0
    fp = 0.0
    fn = 0.0
    tp_idx = []
    tn_idx = []
    fp_idx = []
    fn_idx = []

    for i in range(len(y_test)):
        if (y_test[i] == 1):
            if (predict[i] == 1):
                tp += 1
                tp_idx.append(i)
            else:
                fn += 1
                fn_idx.append(i)
        if (y_test[i] == 0):
            if (predict[i] == 1):
                fp += 1
                fp_idx.append(i)
            else:
                tn += 1
                tn_idx.append(i)

    tpr, tnr, fpr, acc, mcc, fdr, f1 = -1, -1, -1, -1, -1, -1, -1
    try:
        #risk_score = tp/(tp+fn) + tp/(tp+fp) # very similar to f1 score
        #risk_score = 2 * tp / (2 * tp + fp + fn)
        risk_score = tp / (tp + fn)
        #risk_score = tp / (tp + fp)
        #risk_score = tp / (tp + fp + fn)
    except:
        risk_score = -1

    try:
        tpr = tp / (tp + fn)
        tnr = tn / (tn + fp)
        fpr = fp / (fp + tn)
        acc = (tp + tn) / (tp + tn + fp + fn)
        mcc = (tp * tn - fp * fn) / sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        fdr = fp / (fp + tp)
        f1 = 2 * tp / (2 * tp + fp + fn)
    except Exception:
        pass
    return tp, tn, fp, fn, tpr, tnr, fpr, acc, mcc, fdr, f1, tp_idx, tn_idx, fp_idx, fn_idx, risk_score


def riskScore(queryset, member_gt):
    """

    :param queryset: df from MIA_results['query_values_bin']
    :param member_gt: [01] list label the real membership of query set
    :return:
    """
    member_pred = list(
        np.array(list(queryset.loc[:, ['correctness', 'confidence', 'entropy']].sum(axis=1) >= 1), dtype=np.int32))
    tp, tn, fp, fn, tpr, tnr, fpr, acc, mcc, fdr, f1, tp_idx, tn_idx, fp_idx, fn_idx, risk_score = Performance(
        member_pred, member_gt)
    return risk_score

def riskScore2(queryset, member_gt):
    """

    :param queryset: df from MIA_results['query_values_bin']
    :param member_gt: [01] list label the real membership of query set
    :return:
    """
    member_pred = list(
        np.array(list(queryset.loc[:, ['correctness', 'confidence', 'entropy']].sum(axis=1) >= 2), dtype=np.int32))
    tp, tn, fp, fn, tpr, tnr, fpr, acc, mcc, fdr, f1, tp_idx, tn_idx, fp_idx, fn_idx, risk_score = Performance(
        member_pred, member_gt)
    return risk_score