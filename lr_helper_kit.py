# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 09:44:57 2021

This module calculates useful metrics for logistic regression using the
scikit_learn library so that they produce equivalent results to the
statsmodels library. The statsmodels library is mostly compatible with the
functions of the R statistical programming language, sklearn basically does
not provide many functions, such as calculating linear scores,
calculating p-values of coefficients, calculating log-likelihood,
calculating AIC/BIC metrics, etc.

This module is designed to make up for that.

@author: MIKLOS
"""

import sys
import numpy as np
import pandas as pd

from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from scipy.stats import norm
from scipy.stats import chi2
import statsmodels.api as sm

sys.path.append("c:/Users/MIKLOS/Data/_Python_projects/modules/")
import custom_metrics as my_metrics


def get_llr_test(y, y_pred):
    """Likelihood ratio test

    The likelihood-ratio test assesses the goodness of fit of two competing
    statistical models based on the ratio of their likelihoods, specifically
    one found by maximization over the entire parameter space and another found
    after imposing some constraint.

    Parameters
    ----------
    y : array of ints
        true values.
    y_pred : array of floats
        predicted values.

    Returns
    -------
    llr_stat : float
        likelihood ratio test statistic
    pvalue : float
        likelihood ratio test p-value.

    References
    ----------
    https://en.wikipedia.org/wiki/Likelihood-ratio_test
    """

    llr_stat = -2 * (log_likelihood_null(y) - log_likelihood(y, y_pred))

    df = 1  # given the difference in dof
    # compute the p-value
    pvalue = 1 - chi2(df).cdf(llr_stat)

    return llr_stat, pvalue


def get_efrons_pseudo_r_square(y, y_pred):
    """
    This function calculates the Efron’s R2

    Parameters
    ----------
    y : array of ints
        true values.
    y_pred : array of floats
        predicted values.

    Returns
    -------
    pseudo_r_square : float
        pseudo R squared value.

    References
    ----------
    https://datascience.oneoffcoder.com/psuedo-r-squared-logistic-regression.html
    """
    numerator = np.sum(np.power(y - y_pred, 2))
    denominator = np.sum(np.power(y - np.mean(y), 2))

    pseudo_r_square = 1 - numerator / denominator
    return pseudo_r_square


def get_pseudo_r_square(y, y_pred):
    """
    In ordinary least square (OLS) regression, the statistics measures the
    amount of variance explained by the regression model. The value of ranges
    in , with a larger value indicating more variance is explained by the
    model (higher value is better).

    The three main ways to interpret R squared is as follows:

    - explained variable: how much variability is explained by the model
    - goodness-of-fit: how well does the model fit the data
    - correlation: the correlations between the predictions and true values

    This function calculates the McFadden’s R2

    Parameters
    ----------
    y : array of ints
        true values.
    y_pred : array of floats
        predicted values.

    Returns
    -------
    pseudo_r_square : float
        pseudo R squared value.

    References
    ----------
    https://datascience.oneoffcoder.com/psuedo-r-squared-logistic-regression.html
    """

    numerator = log_likelihood(y, y_pred)
    denominator = log_likelihood_null(y)
    pseudo_r_square = 1 - numerator / denominator
    return pseudo_r_square


def log_likelihood_null(y):
    """
    Gets the null model log-likelihood value

    Parameters
    ----------
    y : array of ints
        true values.

    Returns
    -------
    lln : float
        null log-likelihood value.

    """
    log_likelihood_elements = y * np.log(np.mean(y)) + (1 - y) * np.log(1 - np.mean(y))
    lln = np.sum(log_likelihood_elements)

    return lln


def log_likelihood(y, y_pred):
    """
    Gets the log-likelihood value

    Parameters
    ----------
    y : array of ints
        true values.
    y_pred : array of floats
        predicted values.

    Returns
    -------
    ll : float
        log-likelihood value.

    """
    log_likelihood_elements = y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)
    ll = np.sum(log_likelihood_elements)

    # This is equivalent to the following
    # likelihood_elements = np.power(y_pred, y) * np.power(1-y_pred, 1-y)
    # lp = np.prod(likelihood_elements)
    # ll = np.log(lp)

    return ll


def get_aic(model, X, y):
    """
    Gets the AIC statistic for logistic regression.

    The Akaike information criterion is calculated from the maximum
    log-likelihood of the model and the number of parameters (K) used to reach
    that likelihood. The AIC function is 2K – 2(log-likelihood).

    Lower AIC values indicate a better-fit model, and a model with a delta-AIC
    (the difference between the two AIC values being compared) of more than -2
    is considered significantly better than the model it is being compared to.

    Parameters
    ----------
    model : sklearn model
        model.
    X : DataFrame or numpy array
        matrix of the data.
    y : Series or numpy array
        true values.

    Returns
    -------
    aic : float
        information criterion

    References
    ----------
    https://github.com/statsmodels/statsmodels/blob/c30c29a2c2d2ac11c730d21c19a0556839535ff3/statsmodels/tools/eval_measures.py#L309
    """

    nobs = X.shape[0]
    # number of parameters including constant
    df_modelwc = len(model.coef_[0]) + 1
    y_pred = model.predict_proba(X)[:, 1]

    ll = log_likelihood(y, y_pred)

    if nobs / df_modelwc < 40:
        aic = -2.0 * ll + 2.0 * df_modelwc * nobs / (nobs - df_modelwc - 1.0)
    else:
        # statsmodels equation
        aic = -2.0 * ll + 2.0 * df_modelwc
    # aic = -2.0 * ll + 2.0 * df_modelwc
    return aic


def get_bic(model, X, y):
    """
    The Bayesian Information Criterion, or BIC for short, is a method for
    scoring and selecting a model.

    BIC = -2 * LL + log(N) * k

    Where log() has the base-e called the natural logarithm, LL is the
    log-likelihood of the model, N is the number of examples in the training
    dataset, and k is the number of parameters in the model.

    Parameters
    ----------
    model : sklearn model
        model.
    X : DataFrame or numpy array
        matrix of the data.
    y : Series or numpy array
        true values.

    Returns
    -------
    bic : float
        information criterion

    References
    ----------
    https://github.com/statsmodels/statsmodels/blob/c30c29a2c2d2ac11c730d21c19a0556839535ff3/statsmodels/tools/eval_measures.py#L309

    """
    nobs = X.shape[0]
    # number of parameters including constant
    df_modelwc = len(model.coef_[0]) + 1
    y_pred = model.predict_proba(X)[:, 1]

    ll = log_likelihood(y, y_pred)

    # statsmodels equation
    bic = -2.0 * ll + np.log(nobs) * df_modelwc

    return bic


def hosmer_lemeshow(pihat, Y):
    """Hosmer-Lemeshow goodness of Fit test
    https://stackoverflow.com/questions/40327399/hosmer-lemeshow-goodness-of-fit-test-in-python
    https://colors-newyork.com/how-do-you-interpret-the-hosmer-and-lemeshow-goodness-of-fit-test/#How_do_you_interpret_the_Hosmer_and_Lemeshow_goodness_of_fit_test

    Parameters
    ----------
    pihat : Numpy array of floats
        Array of model outputs.
    Y : Numpy array of floats
        True values.

    Returns
    -------
    DataFrame
        DataFrame containing chi squared value and p_value.
    """
    # pihat=model.predict()
    pihatcat = pd.cut(
        pihat,
        np.percentile(pihat, [0, 25, 50, 75, 100]),
        labels=False,
        include_lowest=True,
    )  # here we've chosen only 4 groups

    meanprobs = [0] * 4
    expevents = [0] * 4
    obsevents = [0] * 4
    meanprobs2 = [0] * 4
    expevents2 = [0] * 4
    obsevents2 = [0] * 4

    for i in range(4):
        meanprobs[i] = np.mean(pihat[pihatcat == i])
        expevents[i] = np.sum(pihatcat == i) * np.array(meanprobs[i])
        obsevents[i] = np.sum(Y[pihatcat == i])
        meanprobs2[i] = np.mean(1 - pihat[pihatcat == i])
        expevents2[i] = np.sum(pihatcat == i) * np.array(meanprobs2[i])
        obsevents2[i] = np.sum(1 - Y[pihatcat == i])
    data1 = {"meanprobs": meanprobs, "meanprobs2": meanprobs2}
    data2 = {"expevents": expevents, "expevents2": expevents2}
    data3 = {"obsevents": obsevents, "obsevents2": obsevents2}

    m = pd.DataFrame(data1)
    e = pd.DataFrame(data2)
    o = pd.DataFrame(data3)

    # The statistic for the test, which follows, under the null hypothesis,
    # The chi-squared distribution with degrees of freedom equal to amount of
    # groups - 2. Thus 4 - 2 = 2
    tt = sum(sum((np.array(o) - np.array(e)) ** 2 / np.array(e)))
    pvalue = 1 - chi2.cdf(tt, 2)

    return pd.DataFrame(
        [[chi2.cdf(tt, 2).round(2), pvalue.round(2)]], columns=["Chi2", "p - value"]
    )


def probabilistic_to_linear(pred_values_prob):
    """
    Convert probabilistic score to linear

    Parameters
    ----------
    pred_values_prob : Numpy array of floats or float number
        Array of probabilistic model outputs or float number.

    Returns
    -------
    Numpy array of floats or float number
        Array of linear model outputs or float number.

    """
    return np.log(pred_values_prob / (1 - pred_values_prob))


def linear_to_probabilistic(score):
    """
    Convert linear score to probabilistic

    Parameters
    ----------
    score : Numpy array of floats or float number
        Array of linear model outputs or float number.

    Returns
    -------
    Numpy array of floats or float number
        Array of probabilistic model outputs or float number.

    """

    return np.exp(score) / (np.exp(score) + 1)


def logit_pvalue(model, x):
    """
    Calculates Logistic Regression coefficient p-values

    Parameters
    ----------
    model : sklearn model
        LR model.
    x : array
        matrix you want to calculate the coefficients and p values on.

    Returns
    -------
    p1 : array
        p values.

    """

    p1 = model.predict_proba(x)
    n1 = len(p1)
    m1 = len(model.coef_[0]) + 1
    coefs = np.concatenate([model.intercept_, model.coef_[0]])
    x_full = np.matrix(np.insert(np.array(x), 0, 1, axis=1))
    answ = np.zeros((m1, m1))
    for i in range(n1):
        answ = (
            answ
            + np.dot(np.transpose(x_full[i, :]), x_full[i, :]) * p1[i, 1] * p1[i, 0]
        )
    vcov = np.linalg.inv(np.matrix(answ))
    se = np.sqrt(np.diag(vcov))
    t1 = coefs / se
    p1 = (1 - norm.cdf(abs(t1))) * 2

    return p1


def lr_predic_linear(model, X_test):
    """
    Get Logistic regression linear predictions

    Parameters
    ----------
    model : sklearn model
        model to get coefficients.
    X_test : DataFrama or numpy Array
        Test data to predict on.

    Returns
    -------
    pred_lin in array type.

    """

    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.values
    coefs = [model.intercept_[0]]
    coefs.extend(model.coef_[0])

    pred_lin = []

    for row_d in X_test:
        row_d = np.array(row_d)
        score_lin = np.array(coefs[0]) + np.sum(np.array(coefs[1:]) * row_d)
        score_lin = float(score_lin)
        pred_lin.append(score_lin)
    pred_lin = np.array(pred_lin)
    return pred_lin


def find_younden_threshold(model, X, y, lr_linear_prediction=False):
    """Finds the optimal probability threshold point for a classification
    model based on the Younden index in the range 0.1 to 1, with 0.01
    increments

    The Younden index is defined as follows:
    J = Max(Sensitivity + Specificity – 1)

    Parameters
    ----------
    model : sklearn model
        The classification model in question.
    X_test : Array of floats
        Matrix with test data, where rows are observations.
    y_test : Array of ints
        Vector with target test data, where rows are observations.
    lr_linear_prediction: bool
        If false prediction is probabilistic, if True prediction is linear.
        Applies only to Logistic Regresion!

    Returns
    -------
    optimal_threshold : float
        optimal threshold based on Younden index.

    """

    if lr_linear_prediction is False:
        thresholds = np.arange(0.1, 1, 0.01)
    else:
        thresholds = np.arange(-15, 15, 0.05)
    sensitivity = []
    specificity = []

    for threshold in thresholds:
        if lr_linear_prediction is False:
            pred_values_prob = model.predict_proba(X)[:, 1]
            y_pred = np.array([1 if x >= threshold else 0 for x in pred_values_prob])
        else:
            pred_values_lin = lr_predic_linear(model, X)
            y_pred = np.array([1 if x >= threshold else 0 for x in pred_values_lin])
        mx = my_metrics.get_confusion_matrix(y, y_pred)

        sensitivity.append(my_metrics.get_sensitivity(mx))
        specificity.append(my_metrics.get_specificity(mx))
    d = {
        "Threshold": thresholds,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }
    df = pd.DataFrame(d)
    df["Younden"] = df["sensitivity"] + df["specificity"] - 1
    # df.to_excel("Younden_thr.xlsx")
    optimal_threshold = thresholds[np.argmax(df["Younden"])]
    return optimal_threshold


def find_threshold_specificity_based(model, X, y, grater_then=0.8):
    """Finds the optimal probability threshold point for a classification
    model based on the specificity value, default is specificity >= 0.8
    """

    thresholds = np.arange(0.1, 1, 0.01)

    sensitivity = []
    specificity = []

    for threshold in thresholds:
        pred_values_prob = model.predict_proba(X)[:, 1]
        y_pred = np.array([1 if x >= threshold else 0 for x in pred_values_prob])

        mx = my_metrics.get_confusion_matrix(y, y_pred)

        sensitivity.append(my_metrics.get_sensitivity(mx))
        specificity.append(my_metrics.get_specificity(mx))
    d = {
        "Threshold": thresholds,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }
    df = pd.DataFrame(d)
    idx = df.loc[df["specificity"] > grater_then].first_valid_index()
    optimal_threshold = float(df.iloc[[idx]]["Threshold"])

    return optimal_threshold


def main():
    """Main function"""
    # test()

    df = pd.read_csv("C:/Users/MIKLOS/Data/_Python_projects/_github/titanic-data.csv")
    df["Sex"] = np.where(df["Sex"] == "male", 0, 1)
    df = df.dropna()
    model = LogisticRegression(solver="newton-cg", C=1, penalty="none", max_iter=100000)

    # columns = ["feat" + str(i) for i in range(X.shape[1])]
    # X = pd.DataFrame(X, columns=columns)
    X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    print(Counter(y_train))
    print(Counter(y_test))
    model.fit(X_train, y_train)

    # pred_lin = lr_predic_linear(model, X_test)

    # pred_values_prob = model.predict_proba(X_test)[:, 1]
    # # convert probabilistic outout to linear output
    # score = np.log(pred_values_prob / (1 - pred_values_prob))
    # score = probabilistic_to_linear(pred_values_prob)
    # print(score)
    # # -> pred_lin and score are the same (except for the n-th digits)!

    opt_thr_prob = find_younden_threshold(model, X_train, y_train)
    opt_thr_lin = find_younden_threshold(
        model, X_train, y_train, lr_linear_prediction=True
    )

    print("Younden Prob threshold: ", opt_thr_prob)
    print("Younden Lin threshold calculated: ", opt_thr_lin)
    print(
        "Younden Prob threshold converted to linear: ",
        probabilistic_to_linear(opt_thr_prob),
    )
    print(
        "Younden Lin threshold converted to Prob: ",
        linear_to_probabilistic(opt_thr_lin),
    )
    print("---------------------")
    # coefs = [model.intercept_[0]]
    # coefs.extend(model.coef_[0])

    log_reg = sm.Logit(y_train, sm.add_constant(X_train)).fit()
    # log_reg = sm.Logit(y_train, X_train).fit()
    print(log_reg.summary())
    print("\nStatsmodels coefs:")
    print(log_reg.params)
    print("\nSklearn coefs:")
    cols = list(X_train.columns)
    cols.insert(0, "Constant")
    model_coefs = list(model.coef_[0])
    model_coefs.insert(0, model.intercept_[0])
    coef_df = pd.DataFrame(zip(cols, model_coefs))
    coef_df.columns = ["Feature", "Coefficients"]
    print(coef_df.round(3))
    # feature_sel
    # print(model.coef_[0])

    print("\nSklearn p-values:")
    pvalues = list(logit_pvalue(model, X_train))
    p_value_df = pd.DataFrame(zip(cols, pvalues))
    p_value_df.columns = ["Feature", "p-value"]
    print(p_value_df.round(3))

    y_pred = model.predict_proba(X_train)[:, 1]

    print("---------------------")
    print("\nStatsmodels log-likelihood and null log-likelihood:")
    print(log_reg.llf)
    print(log_reg.llnull)

    print("\nHand calculated log-likelihood and null log-likelihood:")
    print(log_likelihood(y_train, y_pred))
    print(log_likelihood_null(y_train))
    print("---------------------")

    print("\nStatsmodels AIC / BIC:")
    print(log_reg.aic)
    print(log_reg.bic)

    print("\nCustom AIC/BIC")
    print(get_aic(model, X_train, y_train))
    print(get_bic(model, X_train, y_train))
    print("---------------------")

    print("\nHosmer Lemeshow test")
    hosmer_lemeshow_test = hosmer_lemeshow(y_pred, y_train)
    print(hosmer_lemeshow_test)

    print("---------------------")
    print("\nStatsmodels pseudo R-squared:")
    print(log_reg.prsquared)
    print("\nCustom pseudo R-squared:")
    print(get_pseudo_r_square(y_train, y_pred))

    print("---------------------")
    llr_stat, pvalue = get_llr_test(y_train, y_pred)
    print("\nStatsmodels LLR value and p-value:")
    print(log_reg.llr)
    print(log_reg.llr_pvalue)
    print("\nCustom pseudo LLR value and p-value:")
    print(llr_stat)
    print(pvalue)


if __name__ == "__main__":
    main()
