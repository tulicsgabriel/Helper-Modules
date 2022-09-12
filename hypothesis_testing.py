# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 11:46:12 2021

This script provides a function that automatizes hypothesis testing of a
dataframe colums selecting a grouping variable. Does normality testing, then
two-sample t-test or Mann-Whitney U-test.

Change log: 
2022-05-15: Added  Levene’s test to check if the groups have equal variances.
this is important for the assumptions of t-test.

Added a two sample contingency test function to perform Chi-Square Test for
contingency tables.

2022-09-12: Added frequency columns for the contingency test table.

@author: MIKLOS
"""

import datetime
import os
import pandas as pd
import numpy as np
from collections import Counter

from scipy.stats import ttest_ind
from scipy.stats import levene
from scipy.stats import mannwhitneyu
from scipy.stats import shapiro
from scipy.stats import chi2_contingency


def get_date_tag():
    """Get date tag to track outputed files"""

    date_tag = datetime.datetime.now()
    return (
        date_tag.strftime("%Y")
        + "_"
        + date_tag.strftime("%m")
        + "_"
        + date_tag.strftime("%d")
    )


def color_boolean(val):
    """Condition of True values"""
    color = ""
    if val is True:
        color = "green"
    elif val is False:
        color = "red"
    else:
        color = None
    return "background-color: %s" % color


def two_sample_t_test(group1, group2, d, alpha):
    """Gets two-sample t-test"""
    t_value, p_value = ttest_ind(group1, group2)
    d["t-test p value"].append(p_value)

    if p_value > alpha:
        d["Sample means are the same"].append(True)
    else:
        d["Sample means are the same"].append(False)
    d["Mann-Whitney U test p value"].append(None)
    d["Sample distributions are the same"].append(None)
    return d


def two_sample_wilcoxon_test(group1, group2, d, alpha):
    """Gets two sample Mann-Whitney U test"""
    statistic, p_value = mannwhitneyu(group1, group2, alternative="two-sided")
    d["Mann-Whitney U test p value"].append(p_value)

    if p_value > alpha:
        d["Sample distributions are the same"].append(True)
    else:
        d["Sample distributions are the same"].append(False)
    d["t-test p value"].append(None)
    d["Sample means are the same"].append(None)
    return d


def two_sample_hypothesis_testing(df, features, group, alpha=0.05):
    """Helper function to run the two-sample hypothesis tests feature by
    feature. It also checks the assumptions of the two sample t-test and runs
    the mann-whitney u test if the assumptions are violated.
    """

    d = {
        "Feature": [],
        "Group 1": [],
        "Group 2": [],
        "Group 1 mean": [],
        "Group 2 mean": [],
        "Group 1 std": [],
        "Group 2 std": [],
        "Shapiro test p value group 1": [],
        "Shapiro test p value group 2": [],
        "Group 1 is_normal": [],
        "Group 2 is_normal": [],
        "t-test p value": [],
        "Sample means are the same": [],
        "Mann-Whitney U test p value": [],
        "Sample distributions are the same": [],
    }

    for feature in features:

        d["Feature"].append(feature)

        group1_name = list(set(list(df[group])))[0]
        group2_name = list(set(list(df[group])))[1]
        d["Group 1"].append(group1_name)
        d["Group 2"].append(group2_name)

        group1 = df[df[group] == list(set(list(df[group])))[0]][feature].dropna()
        group2 = df[df[group] == list(set(list(df[group])))[1]][feature].dropna()

        levene_stat, levene_p = levene(group1, group2)

        d["Group 1 mean"].append(group1.mean())
        d["Group 2 mean"].append(group2.mean())
        d["Group 1 std"].append(group1.std())
        d["Group 2 std"].append(group2.std())

        # normality test
        p_group1 = shapiro(group1)[1]
        p_group2 = shapiro(group2)[1]
        d["Shapiro test p value group 1"].append(p_group1)
        d["Shapiro test p value group 2"].append(p_group2)

        if p_group1 > alpha:
            d["Group 1 is_normal"].append(True)
        else:
            d["Group 1 is_normal"].append(False)
        if p_group2 > alpha:
            d["Group 2 is_normal"].append(True)
        else:
            d["Group 2 is_normal"].append(False)
        if (
            p_group1 > alpha
            and p_group2 > alpha
            and len(group1) > 30
            and len(group2) > 30
            and levene_p > alpha
        ):
            d = two_sample_t_test(group1, group2, d, alpha)
        else:
            d = two_sample_wilcoxon_test(group1, group2, d, alpha)
    out_df = pd.DataFrame(d)
    out_df.style.apply(color_boolean)

    hypot_folder = "./hypothesis_testing/"
    if not os.path.exists(hypot_folder):
        os.makedirs(hypot_folder)
    date_tag = get_date_tag()
    filename = f"hypothesis_testing_by_{group}_{date_tag}.xlsx"
    out_df.style.applymap(
        color_boolean,
        subset=["Sample means are the same", "Sample distributions are the same"],
    ).to_excel(hypot_folder + filename, index=False, freeze_panes=(1, 0))

    print("\n------------")
    print("Hypothesis testing done.\n")
    return out_df


def two_sample_contingency_test(df, features, group, alpha=0.05):
    """Chi-Square Test for contingency table"""
    d = {
        "Feature": [],
        "Group 1": [],
        "Group 2": [],
        "Group 1 freq.": [],
        "Group 2 freq.": [],
        "chi value": [],
        "p-value": [],
        "H0: no relation between the variables": [],
        "H1: significant relationship between the variables": [],
    }

    for feature in features:

        d["Feature"].append(feature)

        group1_name = list(set(list(df[group])))[0]
        group2_name = list(set(list(df[group])))[1]
        d["Group 1"].append(group1_name)
        d["Group 2"].append(group2_name)

        d["Group 1 freq."].append(Counter(df[df[group] == group1_name][feature])[1])
        d["Group 2 freq."].append(Counter(df[df[group] == group2_name][feature])[1])

        contigency = pd.crosstab(df[group], df[feature])
        chi, p_value, dof, expected = chi2_contingency(contigency)
        d["chi value"].append(chi)
        d["p-value"].append(p_value)

        if p_value > alpha:
            d["H0: no relation between the variables"].append(True)
            d["H1: significant relationship between the variables"].append(False)
        else:
            d["H0: no relation between the variables"].append(False)
            d["H1: significant relationship between the variables"].append(True)
    out_df = pd.DataFrame(d)
    out_df.style.apply(color_boolean)
    hypot_folder = "./hypothesis_testing/"
    if not os.path.exists(hypot_folder):
        os.makedirs(hypot_folder)
    date_tag = get_date_tag()
    filename = f"contingency_chisquare_by_{group}_{date_tag}.xlsx"
    out_df.style.applymap(
        color_boolean,
        subset=[
            "H0: no relation between the variables",
            "H1: significant relationship between the variables",
        ],
    ).to_excel(hypot_folder + filename, index=False, freeze_panes=(1, 0))

    print("\n------------")
    print("Contingency chisquare testing done.\n")
    return out_df


def main():
    """Main function"""
    print("It's okay.")

    data = {
        "age": [35, 43, 32, 19, 67, 89, 45, 65, 54, 65],
        "sex": [
            "male",
            "female",
            "female",
            "male",
            "female",
            "female",
            "male",
            "female",
            "female",
            "male",
        ],
        "height": [180, 170, 170, 195, 166, 167, 168, 167, 170, 190],
        "weight": [80, 61, 59, 85, 55, 55, 81, 65, 60, 88],
        "ac": [94, 80, 82, 84, 88, 77, 62, 64, 87, 100],
        "brown_hair": [1, 0, 0, 1, 0, 1, 1, 1, 0, 1],
    }
    df = pd.DataFrame(data)

    out_df = two_sample_hypothesis_testing(df, ["height", "weight", "age"], "sex")
    print(out_df)
    
    out_df_cont = two_sample_contingency_test(df, ["brown_hair"], "sex")
    print(out_df_cont)


if __name__ == "__main__":
    main()
