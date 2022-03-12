# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 11:48:09 2021

@author: MIKLOS


This script calculates Anthropometric Indices (Body related measures) from
clinical features such as height, weight, age, sex and (waist /) abdominal
circumference (ac).

Features:
    - weight in kg, ex. 83 kg
    - height in cm, ex. 173 cm
    - ac in cm, ex 89 cm

The input & output are pandas dataframes.
"""

import pandas as pd
import numpy as np

# pylint: disable=C0103


def get_bmi(df):
    """Calculates the Body Mass Index
    weight in kg and height in cm
    """

    df["BMI"] = np.round(df["weight"] / (np.square(df["height"])) * 10000, 1)
    df["is_Obese"] = np.where(df["BMI"] >= 30, 1, 0)
    return df


def get_corpulence_index(df):
    """The Corpulence Index (CI) or Ponderal Index (PI) is a measure of
    leanness (corpulence) of a person calculated as a relationship between
    mass and height.

    wieght / (height^3)

    Ref.: https://en.wikipedia.org/wiki/Corpulence_index
    """

    df["CI"] = np.round(df["weight"] / (np.power(df["height"] / 100, 3)), 2)
    df["is_CI_optimal"] = np.where(df["CI"] >= 12, 1, 0)
    df["is_Obese_CI"] = np.where(df["CI"] >= 17, 1, 0)
    return df


def get_broca_index_diff(df):
    """
    The Broca Index is an estimation of ideal body weight using a heigh
    measurement only.

    Here I return the person's actual weight - the broca index weight.

    weight in kg and height in cm
    Ref.: https://www.topendsports.com/testing/tests/broca-index.htm
    """

    df["BI_diff"] = df["weight"] - (df["height"] - 100)
    return df


def convert_sex(df):
    """Male -> 0, Female -> 1"""

    df["sex"] = np.where(df["sex"] == "Male", 0, 1)
    return df


def get_relative_fat_mass(df):
    """Estimation of overweight or obesity in humans.

    RFM for adult males: 64 – 20 × (height / waist circumference)
    RFM for adult females: 76 – 20 × (height / waist circumference)

    Ref.: https://en.wikipedia.org/wiki/Relative_Fat_Mass
    """

    df["RMF"] = np.where(
        df["sex"] == 0,
        (64 - 20 * (df["height"] / df["ac"])),
        (76 - 20 * (df["height"] / df["ac"])),
    )
    return df


def get_total_body_water(df):
    """Total Body Water (TBW) Explained

    Body water is defined as the water content in the tissues, blood, bones
    and elsewhere in the body. All the percentages of body water sum up to
    total body water (TBW). Ensuring this value remains constant and within
    healthy limits is part of homeostasis.

    Watson Formulas

    Male TBW = 2.447 - (0.09156 * age) + (0.1074 * height) + (0.3362 * weight)
    Female TBW = -2.097 + (0.1069 * height) * (0.2466 * weight)

    Ref.: https://www.mdapp.co/total-body-water-tbw-calculator-448/
    """

    df["TBW"] = np.where(
        df["sex"] == 0,
        2.447
        - (0.09156 * df["age"])
        + (0.1074 * df["height"])
        + (0.3362 * df["weight"]),
        -2.097 + (0.1069 * df["height"]) + (0.2466 * df["weight"]),
    )
    return df


def get_body_adiposity_index(df):
    """Body adiposity index (BAI)

    The body adiposity index (BAI) is a method of estimating the amount of
    body fat in humans.

    Ref.: https://en.wikipedia.org/wiki/Body_adiposity_index

    """
    df["BAI"] = df["ac"] / ((df["height"] / 100) * np.sqrt((df["height"] / 100))) - 18
    return df


def get_body_shape_index(df):
    """Body Shape Index (Absi index)

    A metric for assessing the health implications of a given human body
    height, mass and waist circumference (WC).

    ABSI = WC/ (BMI^2/3 * height^1/2)

    Ref.: https://en.wikipedia.org/wiki/Body_shape_index
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4544789/
    """

    df["ABSI"] = df["ac"] / (np.power(df["BMI"], 2 / 3) * np.power(df["height"], 1 / 2))
    return df


def get_waist_to_height_ratio(df):
    """Waist-to-height ratio
    A measure of the distribution of body fat.

    WHR= waist/height

    Ref.: https://en.wikipedia.org/wiki/Waist-to-height_ratio
    """

    df["WHR"] = df["ac"] / df["height"]
    return df


def get_body_roundness_index(df):
    """Body Roundness Index (BRI)

    To predict body fat, the percentage of visceral adipose tissue,
    and establish an initial impression of an individual’s physical health.

    Ref.: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4544789/

    """

    df["BRI"] = 364.2 - 365.5 * np.sqrt(
        1
        - (
            (np.power(df["ac"] / (np.power(2 * np.pi, 2)), 2))
            / (np.power(0.5 * df["height"], 2))
        )
    )
    return df


def fill_missing_values_with_median(df):
    """Fill missing values with median considering sex differences."""

    df["height"] = df["height"].fillna(df.groupby("sex")["height"].transform("median"))
    df["weight"] = df["weight"].fillna(df.groupby("sex")["weight"].transform("median"))
    df["ac"] = df["ac"].fillna(df.groupby("sex")["ac"].transform("median"))

    return df


def get_all_metrics(df):
    """Function to get all the metrics"""
    df = get_bmi(df)
    df = convert_sex(df)
    df = get_corpulence_index(df)
    df = get_broca_index_diff(df)
    df = get_relative_fat_mass(df)
    df = get_total_body_water(df)
    df = get_body_adiposity_index(df)
    df = get_body_shape_index(df)
    df = get_waist_to_height_ratio(df)
    df = get_body_roundness_index(df)

    return df


def main():
    """Main function"""

    data = {
        "age": [35, 43, 32, 19, 67, 89, 45, 65, 54, np.nan],
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
        "height": [180, np.nan, 170, 195, 166, 167, np.nan, 167, 170, 190],
        "weight": [80, 61, 59, 85, 55, 55, 81, 65, 60, 88],
        "ac": [94, np.nan, 82, 84, np.nan, 77, 62, 64, 87, 100],
    }
    df = pd.DataFrame(data)

    df = fill_missing_values_with_median(df)
    df = get_all_metrics(df)
    print(df)


if __name__ == "__main__":
    main()
