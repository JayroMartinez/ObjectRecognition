import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

def emg_classification(data):

    # families = np.unique(data['Family'])
    # bins = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    # l1VSl2 = [0, 0.25, 0.5, 0.75, 1]
    # c_param = [0.01, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5]
    cv = 5

    # for test and develop
    family = 'Mugs'
    bin = 10
    l1_param = 0.75
    c_par = 0.25

    selected_df = data.loc[data['Family'] == family]

    # num_trials = np.unique(data['Trial num'])
    # number_of_trials = len(num_trials)

    skf = StratifiedKFold(n_splits=cv)

    for train, test in skf.split(selected_df['Trial num'].astype(int), selected_df['Given Object'].astype(str)):

        train_trials = selected_df.iloc[train]['Trial num']
        test_trials = selected_df.iloc[test]['Trial num']

        # take each trial, take each ep, create bins & compute mean



    a=1