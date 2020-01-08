import numpy as np
import pandas as pd

from sklearn.preprocessing import Imputer

def load_dataset(target):
    """
    function to load the dataset
    out:
    x_df: Pandas DataFrame of the feature, (n_samples, n_variables)
    y_df: Pandas DataFrame of the label, (n_samples, )
    date: Numpy array of the admission date, (n_samples, )
    """

    return x_df, y_df, date

def preprocess_dataset(x_df, y_df, th):
    # imputation
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    x_data = imp.fit_transform(x_df)

    # binarize y
    y_df[y_df < th] = 1
    y_df[y_df >= th] = 0
    y_data = y_df.values
    weight = float(np.sum(y_data==0)) / np.sum(y_data==1)

    print('data shape {}'.format(x_data.shape))
    print('data ratio {}:{}'.format(np.sum(y_data==1), np.sum(y_data==0)))

    return x_data, y_data, weight
