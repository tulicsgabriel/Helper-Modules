# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 11:46:12 2021

This script provides a function that automatizes hypothesis testing of a
dataframe colums selecting a grouping variable. Does normality testing, then
two-sample t-test or Mann-Whitney U-test.

@author: MIKLOS
"""

import datetime
import os
import pandas as pd
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
from scipy.stats import shapiro


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

        group1 = df[df[group] == list(set(list(df[group])))[0]][feature]
        group2 = df[df[group] == list(set(list(df[group])))[1]][feature]

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
    filename = "hypothesis_testing_by_" + group + "_" + date_tag + ".xlsx"
    out_df.style.applymap(
        color_boolean,
        subset=["Sample means are the same", "Sample distributions are the same"],
    ).to_excel(hypot_folder + filename, index=False, freeze_panes=(1, 0))

    print("\n------------")
    print("Hypothesis testing done.\n")
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
    }
    df = pd.DataFrame(data)

    out_df = two_sample_hypothesis_testing(df, ["height", "weight", "age"], "sex")
    print(out_df)


if __name__ == "__main__":
    main()
