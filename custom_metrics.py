# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 15:04:22 2021


This file of a collection of useful custom made functions to general machine
learning / data science related topics & metrics.

@author: MIKLOS
"""

import math
import warnings
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from minepy import MINE
from scipy.spatial import distance
from scipy import stats
from scipy.special import ndtri

# generate random floating point values
from numpy.random import seed
from numpy.random import rand
import compare_auc_delong_xu as delong

warnings.filterwarnings("ignore", category=DeprecationWarning)


# seed random number generator
seed(1)

# pylint: disable=C0103


def _proportion_confidence_interval(r, n, z):
    """Compute confidence interval for a proportion.

    Follows notation described on pages 46--47 of [1].
    https://gist.github.com/maidens/29939b3383a5e57935491303cf0d8e0b

    References
    ----------
    [1] R. G. Newcombe and D. G. Altman, Proportions and their differences,
    in Statisics with Confidence: Confidence intervals and statisctical
    guidelines, 2nd Ed., D. G. Altman, D. Machin, T. N. Bryant and M. J.
    Gardner (Eds.), pp. 45-57, BMJ Books, 2000.
    """

    A = 2 * r + z**2
    B = z * np.sqrt(z**2 + 4 * r * (1 - r / n))
    C = 2 * (n + z**2)
    return ((A - B) / C, (A + B) / C)


def sensitivity_and_specificity_with_confidence_intervals(TP, FP, FN, TN, alpha=0.95):
    """Compute confidence intervals for sensitivity and specificity
    using Wilson's method.

    This method does not rely on a normal approximation and results in accurate
    confidence intervals even for small sample sizes.
    https://gist.github.com/maidens/29939b3383a5e57935491303cf0d8e0b
    Parameters
    ----------
    TP : int
        Number of true positives
    FP : int
        Number of false positives
    FN : int
        Number of false negatives
    TN : int
        Number of true negatives
    alpha : float, optional
        Desired confidence. Defaults to 0.95, which yields a 95% confidence
        interval.

    Returns
    -------
    sensitivity_point_estimate : float
        Numerical estimate of the test sensitivity
    specificity_point_estimate : float
        Numerical estimate of the test specificity
    sensitivity_confidence_interval : Tuple (float, float)
        Lower and upper bounds on the alpha confidence interval for sensitivity
    specificity_confidence_interval
        Lower and upper bounds on the alpha confidence interval for specificity

    References
    ----------
    [1] R. G. Newcombe and D. G. Altman, Proportions and their differences,
    in Statisics with Confidence: Confidence intervals and statisctical
    guidelines, 2nd Ed., D. G. Altman,
    D. Machin, T. N. Bryant and M. J. Gardner (Eds.), pp. 45-57, BMJ Books,
    2000.

    [2] E. B. Wilson, Probable inference, the law of succession, and
    statistical inference, J Am Stat Assoc 22:209-12, 1927.
    """

    #
    z = -ndtri((1.0 - alpha) / 2)

    # Compute sensitivity using method described in [1]
    sensitivity_point_estimate = TP / (TP + FN)
    sensitivity_confidence_interval = _proportion_confidence_interval(TP, TP + FN, z)

    # Compute specificity using method described in [1]
    specificity_point_estimate = TN / (TN + FP)
    specificity_confidence_interval = _proportion_confidence_interval(TN, TN + FP, z)

    return (
        sensitivity_point_estimate,
        specificity_point_estimate,
        sensitivity_confidence_interval,
        specificity_confidence_interval,
    )


def get_accuracy(mx):
    """Gets accuracy
    (tp + tn) / (tp + fp + tn + fn)
    """
    [tp, fp], [fn, tn] = mx
    return (tp + tn) / (tp + fp + tn + fn)


def get_recall(mx):
    """Gets sensitivity, recall, hit rate, or true positive rate (TPR)
    Same as sensitivity
    """
    [tp, fp], [fn, tn] = mx
    return tp / (tp + fn)


def get_precision(mx):
    """Gets precision or positive predictive value (PPV)"""
    [tp, fp], [fn, tn] = mx
    return tp / (tp + fp)


def get_f1score(mx):
    """Gets F1 score, the harmonic mean of precision and sensitivity
    2*((precision*recall)/(precision+recall))
    """
    return 2 * (
        (get_precision(mx) * get_recall(mx)) / (get_precision(mx) + get_recall(mx))
    )


def get_specificity(mx):
    """Gets specificity, selectivity or true negative rate (TNR)"""
    [tp, fp], [fn, tn] = mx
    return tn / (tn + fp)


def get_sensitivity(mx):
    """Gets sensitivity"""
    [tp, fp], [fn, tn] = mx
    return tp / (tp + fn)


def get_MCC(mx):
    """ Gets the Matthews Correlation Coefficient (MCC)"""
    [tp, fp], [fn, tn] = mx
    return (tp * tn - fp * fn) / math.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    )


def print_confusion_matrix(y_true, y_pred):
    """Print confusion matrix and return confusion matrix as array of ints"""
    print("")

    true = pd.Categorical(
        list(np.where(np.array(y_true) == 1, "Cancer", "Healthy")),
        categories=["Cancer", "Healthy"],
    )

    pred = pd.Categorical(
        list(np.where(np.array(y_pred) == 1, "Cancer", "Healthy")),
        categories=["Cancer", "Healthy"],
    )

    df_finalconf = pd.crosstab(
        pred,
        true,
        rownames=["Predicted"],
        colnames=["Actual"],
        margins=True,
        margins_name="Total",
    )

    print(df_finalconf)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    cm_final = np.array([[tp, fp], [fn, tn]])

    return cm_final


def get_confusion_matrix(y_true, y_pred):
    """ Returns confusion matrix as array of ints"""
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    cm_final = np.array([[tp, fp], [fn, tn]])

    return cm_final


def get_number_of_proteins(feature_list):
    """Input a list of features and outputs the number of proteins present
    in the feature vector
    """
    protein_names = [
        "prot1",
        "prot2",
        "prot3",
        "prot4",
        "prot5",
        "prot6",
        "prot7",
        "prot6",
        "prot7",
        "prot8",
        "prot9",
        "prot10",
    ]

    bool_list = ["False"] * len(protein_names)

    # make features upper in feature list
    feature_list = [w.upper() for w in feature_list]
    protein_names = [w.upper() for w in protein_names]

    for i, protein in enumerate(protein_names):
        for feature in feature_list:
            if protein in feature:
                bool_list[i] = "True"
    num_of_proteins = bool_list.count("True")
    return num_of_proteins


def get_correlation_metrics(x, y):
    """
    Get of Maximal Information Coefficient (MIC), Pearson, Spearman
    and Cosine similarity
    More on MIC: https://bit.ly/3xQfYSI

    Parameters
    ----------
    def get_correlation_metrics : x, y lists of numerical values.

    Returns
    -------
    Series of Pearson coeff, Spearman coeff, MIC & Cosine Similarity.

    """
    # initialise mine alogithm with default parameters
    mine = MINE(alpha=0.6, c=15)

    results = dict()

    # returns both Pearson's coefficient and p-value,
    # keep the first value which is the r coefficient
    results["Pearson coeff"] = stats.pearsonr(x, y)[0]
    results["Spearman coeff"] = stats.spearmanr(x, y)[0]
    mine.compute_score(x, y)
    results["MIC"] = mine.mic()
    results["Cosine Similarity"] = 1 - distance.cosine(x, y)
    return pd.Series(results)


def get_delong_ci(y, pred, alpha=0.95):
    """
    Gets DeLong confidence interval for AUC

    Parameters
    ----------
    y : list
        list of ints, true values.
    pred : list
        list of ints, predictions.
    alpha : float, optional
        . The default is .95.

    Returns
    -------
    auc : float
            AUC
    auc_var : float
        AUC Variance
    auc_ci : tuple
        AUC Confidence Interval given alpha

    """
    y = np.array(y)
    pred = np.array(pred)
    auc, auc_cov = delong.delong_roc_variance(y, pred)

    auc_std = np.sqrt(auc_cov)
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)

    ci = stats.norm.ppf(lower_upper_q, loc=auc, scale=auc_std)

    ci[ci > 1] = 1

    print("AUC:", auc)
    print("AUC COV:", auc_cov)
    print("{0}% AUC CI: {1}".format(str(100 * alpha), ci))
    print("\n")

    return auc, auc_cov, ci


def main():
    """Main function"""
    y = [1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0]
    pred = [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1]

    print_confusion_matrix(y, pred)
    mx_ = get_confusion_matrix(y, pred)

    print("\n")
    print("Mcc: ", get_MCC(mx_))
    print("Acc.: ", get_accuracy(mx_))
    print("F1 score: ", get_f1score(mx_))
    print("Precision: ", get_precision(mx_))
    print("Recall: ", get_recall(mx_))
    print("Sen.: ", get_sensitivity(mx_))
    print("Spec.:", get_specificity(mx_))
    print("\n")

    [tp, fp], [fn, tn] = mx_

    (
        sensitivity_point_estimate,
        specificity_point_estimate,
        sensitivity_confidence_interval,
        specificity_confidence_interval,
    ) = sensitivity_and_specificity_with_confidence_intervals(
        tp, fp, fn, tn, alpha=0.05
    )
    print(
        "Sensitivity: %f, Specificity: %f"
        % (sensitivity_point_estimate, specificity_point_estimate)
    )
    print("alpha = %f CI for sensitivity:" % 0.05, sensitivity_confidence_interval)
    print("alpha = %f CI for specificity:" % 0.05, specificity_confidence_interval)
    print("\n")

    # DeLong confidence interval
    auc, auc_cov, ci = get_delong_ci(y, pred)

    for a in [0.0, 0.5, 0.9, 0.95, 0.99, 0.999999]:
        (
            sensitivity_point_estimate,
            specificity_point_estimate,
            sensitivity_confidence_interval,
            specificity_confidence_interval,
        ) = sensitivity_and_specificity_with_confidence_intervals(
            tp, fp, fn, tn, alpha=a
        )
        print(
            "Sensitivity: %f, Specificity: %f"
            % (sensitivity_point_estimate, specificity_point_estimate)
        )
        print("alpha = %f CI for sensitivity:" % a, sensitivity_confidence_interval)
        print("alpha = %f CI for specificity:" % a, specificity_confidence_interval)
        print("")
    manual_feature_names = (
        "(prot9 - exp(prot10))**3",
        "(-prot10**3 + Abs(prot4))**2",
        "(-prot10 + Abs(prot3))**3",
        "(prot1 + Edad_scaled**3)**3",
        "(prot9 + Abs(prot6))**3",
        "prot9*exp(-prot6)",
        "(prot9 + Abs(prot3))**2",
        "(-prot2**3 + prot10**2)**3",
    )

    print("Number of proteins in the feature vector: ")
    print(get_number_of_proteins(manual_feature_names))

    ## Get of Maximal Information Coefficient (MIC), Pearson, Spearman
    # and Cosine similarity
    # generate random numbers between 0-1 *10
    x = list((rand(10) * 10))
    y = [2.0 + 0.7 * num**2 + 0.5 * num for num in x]

    print("---------------------------")
    out = get_correlation_metrics(x, y)
    print(out)


if __name__ == "__main__":
    main()
