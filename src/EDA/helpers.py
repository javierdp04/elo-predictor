from scipy.stats import chi2_contingency
import pandas as pd

def are_there_outliers(data, variable):
    q1, q3 = data[variable].quantile([0.25, 0.75])
    iqr = q3 - q1
    return ((data[variable] < q1 - 1.5 * iqr) | (data[variable] > q3 + 1.5 * iqr)).any()
