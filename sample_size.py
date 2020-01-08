import random
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

import data_utils

def subsampling_analysis():
    data_ratios = np.arange(0.3, 1.05, 0.05)
    n_rep = 100
    targets = ['vitB1', 'vitB12', 'folate']

    # threshold
    th_dict = dict()
    th_dict['vitB1'] = 30
    th_dict['vitB12'] = 180
    th_dict['folate'] = 4

    aucs_dict = dict()
    for target in targets:
        # load the dataset
        x_df, y_df, date = data_utils.load_dataset(target=target)

        # preprocess the dataset
        x_data, y_data, weight = data_utils.preprocess_dataset(x_df, y_df, th=th_dict[target])

        # split into train and test
        n_train = np.sum(date < 20170000)
        x_data, x_test, y_data, y_test = train_test_split(x_data, y_data,
                                                          train_size=n_train,
                                                          shuffle=False)

        aucs = []
        for data_ratio in data_ratios:
            _aucs = [0] * n_rep
            for i in range(n_rep):
                idx_train = random.sample(list(range(n_train)), int(data_ratio * n_train))

                # fit and predict
                model = RandomForestClassifier(n_estimators=50,
                                               min_samples_split=2,
                                               max_depth=10,
                                               class_weight={1: weight},
                                               random_state=42)
                model.fit(x_data[idx_train], y_data[idx_train])
                prob_test = model.predict_proba(x_test)[:, 1]
                auc_value = roc_auc_score(y_test, prob_test)
                _aucs[i] = auc_value

            aucs.append(_aucs)

        aucs_dict[target] = aucs

    # plot and fit
    curve_params = []
    fig = plt.figure()
    for target, c in zip(targets, ['c', 'y', 'm']):
        aucs = aucs_dict[target]
        aucs_mean = []
        aucs_se = []
        data_sizes = []
        for i in range(len(data_ratios)):
            aucs_mean.append(np.mean(aucs[i]))
            aucs_se.append(np.std(aucs[i]) / np.sqrt(n_rep))
            data_sizes.append(int(data_ratios[i] * n_train) / float(n_train))
        data_sizes = np.array(data_sizes)
        plt.errorbar(data_sizes, aucs_mean, yerr=aucs_se, color=c, fmt='o', label=target)

        # curve fit
        def f(x, a, b):
            return 0.5 + a * x / (x + b)

        popt, pcov = curve_fit(f, data_sizes, aucs_mean, p0=[1, 0])
        plt.plot(data_sizes, f(data_sizes, *popt), c + '-')
        curve_params.append(popt)

    plt.xlabel('Proportion of the training dataset')
    plt.ylabel('AUC')
    plt.legend(loc='upper left', borderaxespad=0, fontsize=18)
    plt.savefig('sample_size.pdf')

    for idx in range(3):
        a = curve_params[idx][0]
        b = curve_params[idx][1]
        c = (a + 0.5) * 0.99 - 0.5
        print(b * c / (a - c))

if __name__ == "__main__":
    subsampling_analysis()
