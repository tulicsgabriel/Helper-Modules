# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 18:50:56 2021

This script provides a Variance Inflation Factor analysis based feature
selection

@author: MIKLOS
"""
import os
import pandas as pd
from sklearn.linear_model import LinearRegression

# pylint: disable=C0103


def calc_reg_return_vif(X, y):
    """
    Utility function to calculate the VIF. This section calculates the linear
    regression inverse R squared.

    Parameters
    ----------
    X : DataFrame
        Input data.
    y : Series
        Target.
    Returns
    -------
    vif : float
        Calculated VIF value.
    """
    X = X.values
    y = y.values

    if X.shape[1] == 1:
        print("Note, there is only one predictor here")
        X = X.reshape(-1, 1)
    reg = LinearRegression().fit(X, y)
    vif = 1 / (1 - reg.score(X, y))

    return vif


def get_vif(df):
    """
    Calculating VIF using function from scratch

    Parameters
    ----------
    df : DataFrame
        without target variable.
    Returns
    -------
    vif : DataFrame
        giving the feature - VIF value pair.
    """

    vif = pd.DataFrame()

    vif_list = []
    for feature in list(df.columns):
        y = df[feature]
        X = df.drop(feature, axis="columns")
        vif_list.append(calc_reg_return_vif(X, y))
    vif["feature"] = df.columns
    vif["VIF"] = vif_list
    return vif


def vif_feature_selection(df, save_steps=False, drop_excluded=True):
    """ Variance Inflation Factor analysis based feature selection

    Parameters
    ----------
    df : DataFrame
        Input DataFrame not containing target variable.
    save_steps : bool, optional
        If True, it creates a folder and saves the steps of the feature
        analysis in Excel format in the folder. The default is False.
    drop_excluded : bool, optional
        If True, drops the excluded values from the dataframe.
        The default is True.

    Returns
    -------
    df : DataFrame
        Output dataFrame with excluded features (or not).

    """

    vif = get_vif(df)

    if save_steps:
        vif_folder = "./vif_feature_selection_steps/"
        if not os.path.exists(vif_folder):
            os.makedirs(vif_folder)
        vif.style.background_gradient(cmap="Blues").to_excel(
            vif_folder + "0_vif_all.xlsx", index=False
        )
    max_value_id = int(vif[["VIF"]].idxmax())
    max_value = float(vif[["VIF"]].max())
    max_value_feature = vif.iloc[max_value_id, 0]

    i = 1
    while max_value > 10:
        df = df.drop([max_value_feature], axis="columns")
        vif = get_vif(df)

        if save_steps:
            vif.style.background_gradient(cmap="Blues").to_excel(
                vif_folder + str(i) + "_vif_" + max_value_feature + ".xlsx", index=False
            )
        max_value_id = int(vif[["VIF"]].idxmax())
        max_value = float(vif[["VIF"]].max())
        max_value_feature = vif.iloc[max_value_id, 0]
        i = i + 1
    print("\n------------")
    print("Variance Inflation Factor analysis results:\n")
    print(vif)

    if drop_excluded:
        df = df[list(vif.feature)]
    return df


def main():
    """ Main function
    """
    print("It's okay.")


if __name__ == "__main__":
    main()
