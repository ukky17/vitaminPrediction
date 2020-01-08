import argparse
import numpy as np

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics.scorer import make_scorer

import data_utils

def grid_search():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', choices=['vitB1', 'vitB12', 'folate'])
    parser.add_argument('--modelType', choices=['svc', 'rf', 'knn'])
    parser.add_argument('--reverse', action='store_true')
    opt = parser.parse_args()

    # threshold values
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

    # model and hyperparameter set
    if opt.modelType == 'svc':
        model = SVC(class_weight={1: weight})
        param_dict = {'gamma': [1e-10, 1e-9, 1e-8, 1e-7, 1e-6],
                      'C': [1e3, 1e4, 1e5, 1e6, 1e7],
                      'kernel': ['rbf']}
    elif opt.modelType == 'rf':
        model = RandomForestClassifier(class_weight={1: weight})
        param_dict = {'n_estimators': [10, 30, 50, 70, 90],
                      'min_samples_split': [2, 3, 5, 7],
                      'max_depth': [2, 4, 6, 8, 10, 12]}
    elif opt.modelType == 'knn':
        model = KNeighborsClassifier()
        param_dict = {'n_neighbors': [21, 25, 29, 33, 37, 41],
                      'weights': ['uniform','distance'],
                      'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                      'leaf_size': [1,2,3,4,5,7,9,11],
                      'p': [1, 2]}

    # grid search
    my_scorer = make_scorer(roc_auc_score, greater_is_better=True)
    gs = GridSearchCV(model, param_dict, cv=5, scoring='roc_auc')
    gs.fit(x_data, y_data)

    # evaluate
    print('best hyperparameter set')
    print(gs.best_estimator_)
    print()
    means = gs.cv_results_['mean_test_score']
    stds = gs.cv_results_['std_test_score']
    params = gs.cv_results_['params']
    for m, s, p in zip(means, stds, params):
        print('{:.5f} (+/- {:.5f}) for {}'.format(m, s, p))
    print()

if __name__ == "__main__":
    grid_search()
