import argparse
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc

import data_utils

def draw_roc(y_test, prob_test, title):
    plt.figure(figsize=(8, 6))
    fpr, tpr, th = roc_curve(y_test, prob_test)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, 'r')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.savefig('roc_' + title + '.pdf')

def test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', choices=['vitB1', 'vitB12', 'folate'])
    parser.add_argument('--modelType', choices=['lr', 'svc', 'rf', 'knn'])
    parser.add_argument('--reverse', action='store_true')
    opt = parser.parse_args()

    # threshold
    th_dict = dict()
    th_dict['vitB1'] = 30
    th_dict['vitB12'] = 180
    th_dict['folate'] = 4

    # load the dataset
    x_df, y_df, date = data_utils.load_dataset(target=opt.target)

    # preprocess the dataset
    x_data, y_data, weight = data_utils.preprocess_dataset(x_df, y_df, th=th_dict[opt.target])

    # split into train and test
    n_train = np.sum(date < 20170000)
    if opt.reverse:
        x_data, y_data = x_data[::-1], y_data[::-1]
    x_data, x_test, y_data, y_test = train_test_split(x_data, y_data,
                                                      train_size=n_train,
                                                      shuffle=False)

    # model
    if opt.modelType == 'lr':
        model = LogisticRegression(C=1e1, random_state=42, class_weight={1: weight})
    elif opt.modelType == 'svc':
        model = SVC(kernel='rbf', C=1e6, gamma=1e-9, class_weight={1: weight},
                    probability=True, random_state=42)
    elif opt.modelType == 'rf':
        model = RandomForestClassifier(n_estimators=50,
                                       min_samples_split=2,
                                       max_depth=10,
                                       class_weight={1: weight},
                                       random_state=42)
    elif opt.modelType == 'knn':
        model = KNeighborsClassifier(algorithm='auto',
                                     leaf_size=1,
                                     metric='minkowski',
                                     metric_params=None,
                                     n_jobs=1,
                                     n_neighbors=37,
                                     p=1,
                                     weights='uniform')

    # fit and predict
    model.fit(x_data, y_data)
    prob_test = model.predict_proba(x_test)[:, 1]

    # evaluation
    auc_value = roc_auc_score(y_test, prob_test)
    print('AUC: {:.4f}'.format(auc_value))
    draw_roc(y_test, prob_test, opt.modelType)

if __name__ == "__main__":
    test()
