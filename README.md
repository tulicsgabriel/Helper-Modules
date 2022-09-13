
# Data Science / Machine Learning helper modules

In the Python modules below, I have collected some useful functions and bundled them together. These reusable modules can save you a lot of time in data analysis/machine learning.

Modules: lr_helper_kit, vif_analysis, hypothesis_testing, custom_metrics, anthropometric_indices

### lr_helper_kit

This module calculates useful metrics for Logistic Regression using the scikit_learn library so that they produce equivalent results to the statsmodels library. The statsmodels library is mostly compatible with the functions of the R statistical programming language, sklearn basically does not provide many functions, such as calculating linear scores, calculating p-values of coefficients, calculating log-likelihood, calculating AIC/BIC metrics, etc.

This module is designed to make up for that.

Functions:
- get_llr_test: Likelihood ratio test
- get_efrons_pseudo_r_square: Calculates the Efron’s R2
- get_pseudo_r_square: Calculates the McFadden’s R2
- log_likelihood_null: Gets the null model log-likelihood value
- log_likelihood: Gets the log-likelihood value
- get_aic: Gets the Akaike information Criterion statistic for logistic regression
- get_bic: Gets the Bayesian Information Criterion statistic for logistic regression
- hosmer_lemeshow: Hosmer-Lemeshow goodness of Fit test
- probabilistic_to_linear: Converts probabilistic score to linear
- linear_to_probabilistic: Converts linear score to probabilistic
- logit_pvalue: Calculates Logistic Regression coefficient p-values
- lr_predic_linear: Get Logistic regression linear predictions
- find_younden_threshold: Finds the optimal probability threshold point for a classification model based on the Younden index in the range 0.1 to 1, with 0.01 increments
- find_threshold_specificity_based: Finds the optimal probability threshold point for a classification model based on the specificity value, default is specificity >= 0.8

### vif_analysis

This script provides a Variance Inflation Factor analysis based feature
selection. The code produces the same results as is you'd work in R language.

Functions:
- get_vif: Calculating VIF using function from scratch
- calc_reg_return_vif: Utility function to calculate the VIF. This section calculates the linear regression inverse R squared.
- vif_feature_selection: Variance Inflation Factor analysis based feature selection

### hypothesis_testing

This script provides a function that automatizes hypothesis testing of a dataframe colums selecting a grouping variable. Does normality testing, then two-sample t-test or Mann-Whitney U-test. Saves results in a formatted Excel file.

Functions:
- get_date_tag: Get date tag to track outputed files
- color_boolean: Background color of True values
- two_sample_t_test: Gets two-sample t-test
- two_sample_wilcoxon_test: Gets two sample Mann-Whitney U test
- two_sample_hypothesis_testing: Helper function to run the two-sample hypothesis tests feature by feature. It also checks the assumptions of the two sample t-test and runs the mann-whitney u test if the assumptions are violated.
- two_sample_contingency_test: Function to run chi-Square Test for contingency table


### custom_metrics

This file of a collection of useful custom made functions to general machine
learning / data science related topics & metrics.

Functions:
- sensitivity_and_specificity_with_confidence_intervals: Compute confidence intervals for sensitivity and specificity using Wilson's method.
- _proportion_confidence_interval: Compute confidence interval for a proportion
- get_accuracy: Gets accuracy
- get_recall: Gets sensitivity, recall, hit rate, or true positive rate (TPR). Same as sensitivity
- get_precision: Gets precision or positive predictive value (PPV)
- get_specificity: Gets specificity, selectivity or true negative rate (TNR)
- get_sensitivity: Gets sensitivity
- get_MCC: Gets the Matthews Correlation Coefficient (MCC)
- get_f1score: Gets F1 score, the harmonic mean of precision and sensitivity
- print_confusion_matrix: Print confusion matrix and return confusion matrix as array of ints
- get_confusion_matrix: returns confusion matrix as array of ints
- get_correlation_metrics: Get of Maximal Information Coefficient (MIC), Pearson, Spearman and Cosine similarity
- get_delong_ci: Gets DeLong confidence interval for AUC

### anthropometric_indices

This script calculates Anthropometric Indices (Body related measures) from
clinical features such as height, weight, age, sex and (waist /) abdominal
circumference (ac).

Features:
    - weight in kg, ex. 83 kg
    - height in cm, ex. 173 cm
    - ac in cm, ex 89 cm

The input & output are pandas dataframes.

Functions:
- get_bmi: Calculates the Body Mass Index
- get_corpulence_index: Gets the Corpulence Index (CI) or Ponderal Index (PI)
- get_broca_index_diff: The Broca Index is an estimation of ideal body weight using a heigh measurement only. Here I return the person's actual weight - the broca index weight.
- get_relative_fat_mass: Calculates the Relative Fat Mass, an estimation of overweight or obesity in humans
- get_total_body_water: Calculates the Total Body Water (TBW) index
- get_body_adiposity_index: Calculates the Body adiposity index (BAI)
- get_body_shape_index: Calculates the Body Shape Index (Absi index)
- get_waist_to_height_ratio: Calculates the Waist-to-height ratio
- get_body_roundness_index: Body Roundness Index (BRI)
- fill_missing_values_with_median: Fill missing values with median considering sex differences
- get_all_metrics: Function to get all the metrics
- convert_sex: Male -> 0, Female -> 1

## Usage/Examples

You can use the modules by simply creating a modules folder and copying the modules there. Then import them like this:

```python
import sys

sys.path.append("your-path-to-the-moduls-folder/modules/")

import custom_metrics as my_metrics
import anthropometric_indices as ai
import lr_helper_kit as lr_kit
import vif_analysis as vif
import hypothesis_testing as ht
```


## Requirements

I used the following versions of packages while writing the functions:

```
minepy==1.2.5
numpy==1.21.5
pandas==1.4.1
scikit_learn==1.0.2
scipy==1.8.0
statsmodels==0.13.2
```

Statsmodels is only used for comparison reasons.

## Related

Here are some related projects

If you want to learn more about Linear regression & Logistic regression models, I've implemented the algorithms from scratch and compared to the sklearn library implementation. In the header docstirng I've also listed some important resources. Check it out!

[Machine-Learning-models-from-scratch](https://github.com/tulicsgabriel/Machine-Learning-models-from-scratch)

## Authors

- [@tulicsgabriel](https://www.github.com/tulicsgabriel)
