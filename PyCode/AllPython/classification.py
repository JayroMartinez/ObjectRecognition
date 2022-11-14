import numpy as np
import itertools
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from collections import Counter
from sklearn.utils.class_weight import compute_sample_weight
import csv
from scipy.stats import zscore
import pandas as pd

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def emg_classification(data):

    families = np.unique(data['Family'])
    bins = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    l1VSl2 = [0, 0.25, 0.5, 0.75, 1]
    c_param = [0.01, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5]
    cv = 3

    # for test and develop
    # family = 'Plates'
    # num_bin = 10
    # l1_param = 0.5
    # c_par = 2

    result_file = open('./results/results_file.csv', 'a')  # Open file in append mode
    # when reading the file, remember to read it completely, overwrite the redundant entries and save it again after removing the duplicates

    for family in families:
        for num_bin in bins:
            for l1_param in l1VSl2:
                for c_par in c_param:

                    total_score = []

                    selected_df = data.loc[data['Family'] == family]  # select particular family
                    emg_cols = [col for col in selected_df.columns if ('flexion' in col) or ('extension' in col)]
                    # orig_size = selected_df.shape[0]
                    selected_df.dropna(subset=emg_cols, axis=0, inplace=True)  # drop rows containing NaN values
                    # empty_rows = selected_df[selected_df[emg_cols].isnull().any(axis=1)]
                    # selected_df.drop(index=empty_rows.index, axis=0, inplace=True)  # drop rows containing empty values
                    # after_drop_size = selected_df.shape[0]
                    # print("FAMILY:", family)
                    # print("EMG datapoints dropped because of NaNs:", orig_size - after_drop_size)
                    # print("% of EMG datapoints dropped because of NaNs:", round(((orig_size - after_drop_size) / orig_size) * 100, 2),"%")
                    # print("-------------------------------------------------------------\n")
                    # print("\nNumber of EPs: ", len(np.unique(selected_df['EP total'])))

                    to_kfold = selected_df.drop_duplicates(subset=['EP total', 'Given Object'])  # only way I found to avoid overlapping
                    # print("Complete Dataset\n", to_kfold['Given Object'].value_counts())

                    skf = StratifiedKFold(n_splits=cv)
                    # WARNING: the skf.split returns the indexes
                    for train, test in skf.split(to_kfold['EP total'].astype(int), to_kfold['Given Object'].astype(str)):

                        # coin = [x for x in train if x in test]
                        # print("Coincidences: ", coin)
                        #
                        # print("Number of Train EPs: ", len(train))
                        # print("Number of Test EPs: ", len(test))
                        # print("Number of Train + Test EPs: ", len(train) + len(test))

                        train_eps = to_kfold.iloc[train]['EP total']  # because skf.split returns the indexes
                        test_eps = to_kfold.iloc[test]['EP total']  # because skf.split returns the indexes

                        # train_eps_data = selected_df.loc[selected_df['EP total'].isin(train_eps)]
                        # test_eps_data = selected_df.loc[selected_df['EP total'].isin(test_eps)]

                        train_data = []
                        train_labels = []

                        # dropped = 0  # Number of dropped EPs

                        # take each ep, create bins & compute mean
                        for trn_iter in train_eps:

                            train_ep = selected_df.loc[selected_df['EP total'] == trn_iter]
                            ep_emg_data = train_ep[emg_cols]
                            ep_in_bins = np.array_split(ep_emg_data, num_bin)

                            with warnings.catch_warnings():
                                warnings.filterwarnings('error')
                                try:
                                    ep_bin_mean = [np.nanmean(x, axis=0) for x in ep_in_bins]  # size = [num_bins] X [64]
                                    flat_ep_mean = list(itertools.chain.from_iterable(ep_bin_mean))  # size = [num_bins X 64] (unidimensional)
                                    train_data.append(flat_ep_mean)
                                    train_labels.append(np.unique(train_ep['Given Object'])[0])
                                except RuntimeWarning:
                                    print("Dropped EP", trn_iter, "from family ", family)
                                    # dropped += 1

                        test_data = []
                        test_labels = []

                        for tst_iter in test_eps:

                            test_ep = selected_df.loc[selected_df['EP total'] == tst_iter]
                            ep_emg_data = test_ep[emg_cols]
                            ep_in_bins = np.array_split(ep_emg_data, num_bin)

                            with warnings.catch_warnings():
                                warnings.filterwarnings('error')
                                try:
                                    ep_bin_mean = [np.nanmean(x, axis=0) for x in ep_in_bins]  # size = [num_bins] X [64]
                                    flat_ep_mean = list(itertools.chain.from_iterable(ep_bin_mean))  # size = [num_bins X 64] (unidimensional)
                                    test_data.append(flat_ep_mean)
                                    test_labels.append(np.unique(test_ep['Given Object'])[0])
                                except RuntimeWarning:
                                    print("Dropped EP", tst_iter, "from family ", family)
                                    # dropped += 1

                        train_counter = Counter(train_labels)
                        # test_counter = Counter(test_labels)
                        # print("Train Labels count: ", train_counter)
                        # print("Test Labels count: ", test_counter)
                        # print("Dropped ", dropped, " EPs out of ", len(train_eps) + len(test_eps), ", ", round((dropped / (len(train_eps) + len(test_eps))) * 100, 2), "%")
                        # print("------------------------------------------------\n")

                        # build model
                        # log_model = LogisticRegression(penalty='elasticnet', C=c_par, class_weight='balanced', random_state=42, solver='saga', max_iter=25000, multi_class='multinomial', n_jobs=-1, l1_ratio=l1_param)
                        log_model = LogisticRegression(penalty='elasticnet', C=c_par, class_weight='balanced', random_state=42, solver='saga', max_iter=5000, multi_class='ovr', n_jobs=-1, l1_ratio=l1_param)
                        # compute weights (because unbalanced dataset)
                        weights = compute_sample_weight(class_weight='balanced', y=train_labels)
                        # lab_w = [[train_labels[it], weights[it]] for it in range(len(weights))]
                        # train model
                        log_model.fit(X=train_data, y=train_labels, sample_weight=weights)
                        # log_model.fit(X=train_data, y=train_labels)
                        # get predictions
                        # pred = log_model.predict_proba(test_data)
                        pred = log_model.predict(test_data)
                        test_counter = Counter(test_labels)
                        pred_counter = Counter(pred)
                        # print("Test Labels count: ", test_counter)
                        # print("Prediction Labels count: ", pred_counter)
                        # cl = log_model.classes_
                        # print("Predictions:", pred)
                        sc = round(log_model.score(X=test_data, y=test_labels)*100, 2)
                        # print("Score: ", sc)
                        total_score.append(sc)
                        # print("Num", cl[0], train_labels.count(cl[0]))
                        # print("Num", cl[1], train_labels.count(cl[1]))
                        # print("Num", cl[2], train_labels.count(cl[2]))
                        # print("========================================\n")
                        # a=1

                    # print("EMG Mean score after", cv, 'folds with C =', c_par, ', L1Ratio =', l1_param, 'and', num_bin, 'bins for family', family, ':', round(np.mean(total_score), 2), "%\n")

                    wr = csv.writer(result_file)
                    results_to_write = ['EMG', family, num_bin, l1_param, c_par, total_score, round(np.mean(total_score), 2)]
                    wr.writerow(results_to_write)

    result_file.close()


def kinematic_classification(data):

    families = np.unique(data['Family'])
    bins = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    l1VSl2 = [0, 0.25, 0.5, 0.75, 1]
    c_param = [0.01, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5]
    cv = 3

    # for test and develop
    # family = 'Plates'
    # num_bin = 10
    # l1_param = 0.5
    # c_par = 2

    result_file = open('./results/results_file.csv', 'a')  # Open file in append mode
    # when reading the file, remember to read it completely, overwrite the redundant entries and save it again after removing the duplicates

    for family in families:
        for num_bin in bins:
            for l1_param in l1VSl2:
                for c_par in c_param:

                    total_score = []

                    selected_df = data.loc[data['Family'] == family]  # select particular family
                    kin_cols = ['ThumbRotate', 'ThumbMPJ', 'ThumbIj', 'IndexMPJ', 'IndexPIJ',
                           'MiddleMPJ', 'MiddlePIJ', 'RingMIJ', 'RingPIJ', 'PinkieMPJ',
                           'PinkiePIJ', 'PalmArch', 'WristPitch', 'WristYaw', 'Index_Proj_J1_Z',
                           'Pinkie_Proj_J1_Z', 'Ring_Proj_J1_Z', 'Middle_Proj_J1_Z',
                           'Thumb_Proj_J1_Z']
                    orig_size = selected_df.shape[0]
                    selected_df.dropna(subset=kin_cols, axis=0, inplace=True)  # drop rows containing NaN values
                    # empty_rows = selected_df[selected_df[emg_cols].isnull().any(axis=1)]
                    # selected_df.drop(index=empty_rows.index, axis=0, inplace=True)  # drop rows containing empty values
                    after_drop_size = selected_df.shape[0]
                    # print("FAMILY:", family)
                    # print("Kin datapoints dropped because of NaNs:", orig_size - after_drop_size)
                    # print("% of Kin datapoints dropped because of NaNs:", round(((orig_size - after_drop_size) / orig_size) * 100, 2), "%")
                    # print("-------------------------------------------------------------\n")
                    # print("\nNumber of EPs: ", len(np.unique(selected_df['EP total'])))

                    to_kfold = selected_df.drop_duplicates(
                        subset=['EP total', 'Given Object'])  # only way I found to avoid overlapping
                    # print("Complete Dataset\n", to_kfold['Given Object'].value_counts())

                    skf = StratifiedKFold(n_splits=cv)
                    # WARNING: the skf.split returns the indexes
                    for train, test in skf.split(to_kfold['EP total'].astype(int),
                                                 to_kfold['Given Object'].astype(str)):

                        # coin = [x for x in train if x in test]
                        # print("Coincidences: ", coin)
                        #
                        # print("Number of Train EPs: ", len(train))
                        # print("Number of Test EPs: ", len(test))
                        # print("Number of Train + Test EPs: ", len(train) + len(test))

                        train_eps = to_kfold.iloc[train]['EP total']  # because skf.split returns the indexes
                        test_eps = to_kfold.iloc[test]['EP total']  # because skf.split returns the indexes

                        # train_eps_data = selected_df.loc[selected_df['EP total'].isin(train_eps)]
                        # test_eps_data = selected_df.loc[selected_df['EP total'].isin(test_eps)]

                        train_data = []
                        train_labels = []

                        # dropped = 0  # Number of dropped EPs

                        # take each ep, create bins & compute mean
                        for trn_iter in train_eps:

                            train_ep = selected_df.loc[selected_df['EP total'] == trn_iter]
                            ep_kin_data = train_ep[kin_cols]
                            ep_in_bins = np.array_split(ep_kin_data, num_bin)

                            with warnings.catch_warnings():
                                warnings.filterwarnings('error')
                                try:
                                    ep_bin_mean = [np.nanmean(x, axis=0) for x in
                                                   ep_in_bins]  # size = [num_bins] X [64]
                                    flat_ep_mean = list(itertools.chain.from_iterable(
                                        ep_bin_mean))  # size = [num_bins X 64] (unidimensional)
                                    train_data.append(flat_ep_mean)
                                    train_labels.append(np.unique(train_ep['Given Object'])[0])
                                except RuntimeWarning:
                                    print("Dropped EP", trn_iter, "from family ", family)
                                    # dropped += 1

                        test_data = []
                        test_labels = []

                        for tst_iter in test_eps:

                            test_ep = selected_df.loc[selected_df['EP total'] == tst_iter]
                            ep_kin_data = test_ep[kin_cols]
                            ep_in_bins = np.array_split(ep_kin_data, num_bin)

                            with warnings.catch_warnings():
                                warnings.filterwarnings('error')
                                try:
                                    ep_bin_mean = [np.nanmean(x, axis=0) for x in
                                                   ep_in_bins]  # size = [num_bins] X [64]
                                    flat_ep_mean = list(itertools.chain.from_iterable(
                                        ep_bin_mean))  # size = [num_bins X 64] (unidimensional)
                                    test_data.append(flat_ep_mean)
                                    test_labels.append(np.unique(test_ep['Given Object'])[0])
                                except RuntimeWarning:
                                    print("Dropped EP", tst_iter, "from family ", family)
                                    # dropped += 1

                        train_counter = Counter(train_labels)
                        # test_counter = Counter(test_labels)
                        # print("Train Labels count: ", train_counter)
                        # print("Test Labels count: ", test_counter)
                        # print("Dropped ", dropped, " EPs out of ", len(train_eps) + len(test_eps), ", ", round((dropped / (len(train_eps) + len(test_eps))) * 100, 2), "%")
                        # print("------------------------------------------------\n")

                        # build model
                        # log_model = LogisticRegression(penalty='elasticnet', C=c_par, class_weight='balanced', random_state=42, solver='saga', max_iter=25000, multi_class='multinomial', n_jobs=-1, l1_ratio=l1_param)
                        log_model = LogisticRegression(penalty='elasticnet', C=c_par, class_weight='balanced',
                                                       random_state=42, solver='saga', max_iter=5000, multi_class='ovr',
                                                       n_jobs=-1, l1_ratio=l1_param)
                        # compute weights (because unbalanced dataset)
                        weights = compute_sample_weight(class_weight='balanced', y=train_labels)
                        # lab_w = [[train_labels[it], weights[it]] for it in range(len(weights))]
                        # train model
                        log_model.fit(X=train_data, y=train_labels, sample_weight=weights)
                        # log_model.fit(X=train_data, y=train_labels)
                        # get predictions
                        # pred = log_model.predict_proba(test_data)
                        pred = log_model.predict(test_data)
                        test_counter = Counter(test_labels)
                        pred_counter = Counter(pred)
                        # print("Test Labels count: ", test_counter)
                        # print("Prediction Labels count: ", pred_counter)
                        # cl = log_model.classes_
                        # print("Predictions:", pred)
                        sc = round(log_model.score(X=test_data, y=test_labels) * 100, 2)
                        # print("Score: ", sc)
                        total_score.append(sc)
                        # print("Num", cl[0], train_labels.count(cl[0]))
                        # print("Num", cl[1], train_labels.count(cl[1]))
                        # print("Num", cl[2], train_labels.count(cl[2]))
                        # print("========================================\n")
                        # a = 1

                    # print("Kin Mean score after", cv, 'folds with C =', c_par, ', L1Ratio =', l1_param, 'and', num_bin,
                    #       'bins for family', family, ':', round(np.mean(total_score), 2), "%\n")

                    wr = csv.writer(result_file)
                    results_to_write = ['Kin', family, num_bin, l1_param, c_par, total_score,
                                        round(np.mean(total_score), 2)]
                    wr.writerow(results_to_write)

    result_file.close()


def multiple_source_classification(data):

    print("Multimodal")
    # families = np.unique(data['Family'])
    families = ['Geometric', 'Mugs', 'Plates']
    bins = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    # bins = [45, 50]
    l1VSl2 = [0, 0.25, 0.5, 0.75, 1]
    c_param = [0.01, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5]
    cv = 3

    # for test and develop
    # family = 'Plates'
    # num_bin = 10
    # l1_param = 0.5
    # c_par = 2

    result_file = open('./results/results_file.csv', 'a')  # Open file in append mode
    # when reading the file, remember to read it completely, overwrite the redundant entries and save it again after removing the duplicates

    for family in families:
        for num_bin in bins:
            for l1_param in l1VSl2:
                for c_par in c_param:

                    total_score = []

                    selected_df = data.loc[data['Family'] == family]  # select particular family
                    orig_size = selected_df.shape[0]
                    selected_df.dropna(axis=0, inplace=True)  # drop rows containing NaN values
                    # empty_rows = selected_df[selected_df[emg_cols].isnull().any(axis=1)]
                    # selected_df.drop(index=empty_rows.index, axis=0, inplace=True)  # drop rows containing empty values
                    after_drop_size = selected_df.shape[0]
                    # print("FAMILY:", family)
                    # print("Multimodal datapoints dropped because of NaNs:", orig_size - after_drop_size)
                    # print("% of Multimodal datapoints dropped because of NaNs:",
                    #       round(((orig_size - after_drop_size) / orig_size) * 100, 2), "%")
                    # print("-------------------------------------------------------------\n")
                    # print("\nNumber of EPs: ", len(np.unique(selected_df['EP total'])))

                    to_kfold = selected_df.drop_duplicates(subset=['EP total', 'Given Object'])  # only way I found to avoid overlapping
                    # print("Complete Dataset\n", to_kfold['Given Object'].value_counts())

                    skf = StratifiedKFold(n_splits=cv)
                    # WARNING: the skf.split returns the indexes
                    for train, test in skf.split(to_kfold['EP total'].astype(int),
                                                 to_kfold['Given Object'].astype(str)):

                        # coin = [x for x in train if x in test]
                        # print("Coincidences: ", coin)
                        #
                        # print("Number of Train EPs: ", len(train))
                        # print("Number of Test EPs: ", len(test))
                        # print("Number of Train + Test EPs: ", len(train) + len(test))

                        train_eps = to_kfold.iloc[train]['EP total']  # because skf.split returns the indexes
                        test_eps = to_kfold.iloc[test]['EP total']  # because skf.split returns the indexes

                        # train_eps_data = selected_df.loc[selected_df['EP total'].isin(train_eps)]
                        # test_eps_data = selected_df.loc[selected_df['EP total'].isin(test_eps)]

                        train_data = []
                        train_labels = []

                        dropped = 0  # Number of dropped EPs

                        # take each ep, create bins & compute mean
                        for trn_iter in train_eps:

                            train_ep = selected_df.loc[selected_df['EP total'] == trn_iter]
                            ep_numeric_data = train_ep.select_dtypes(include='float64')
                            ep_in_bins = np.array_split(ep_numeric_data, num_bin)

                            with warnings.catch_warnings():
                                warnings.filterwarnings('error')
                                try:
                                    ep_bin_mean = [np.nanmean(x, axis=0) for x in ep_in_bins]  # size = [num_bins] X [64]
                                    flat_ep_mean = list(itertools.chain.from_iterable(ep_bin_mean))  # size = [num_bins X 64] (unidimensional)
                                    train_data.append(flat_ep_mean)
                                    train_labels.append(np.unique(train_ep['Given Object'])[0])
                                except RuntimeWarning:
                                    # print("Dropped EP", trn_iter, "from family ", family)
                                    dropped += 1

                        test_data = []
                        test_labels = []

                        for tst_iter in test_eps:

                            test_ep = selected_df.loc[selected_df['EP total'] == tst_iter]
                            ep_in_bins = np.array_split(ep_numeric_data, num_bin)

                            with warnings.catch_warnings():
                                warnings.filterwarnings('error')
                                try:
                                    ep_bin_mean = [np.nanmean(x, axis=0) for x in ep_in_bins]  # size = [num_bins] X [64]
                                    flat_ep_mean = list(itertools.chain.from_iterable(ep_bin_mean))  # size = [num_bins X 64] (unidimensional)
                                    test_data.append(flat_ep_mean)
                                    test_labels.append(np.unique(test_ep['Given Object'])[0])
                                except RuntimeWarning:
                                    # print("Dropped EP", tst_iter, "from family ", family)
                                    dropped += 1

                        # Z-Score normalization for Train and Test data
                        train_df = pd.DataFrame(train_data)
                        train_df.apply(zscore)
                        test_df = pd.DataFrame(test_data)
                        test_df.apply(zscore)

                        # train_counter = Counter(train_labels)
                        # test_counter = Counter(test_labels)
                        # print("Train Labels count: ", train_counter)
                        # print("Test Labels count: ", test_counter)
                        # print("Dropped ", dropped, " EPs out of ", len(train_eps) + len(test_eps), ", ", round((dropped / (len(train_eps) + len(test_eps))) * 100, 2), "%")
                        # print("------------------------------------------------\n")

                        if test_df.shape[0] > 0:
                            # build model
                            # log_model = LogisticRegression(penalty='elasticnet', C=c_par, class_weight='balanced', random_state=42, solver='saga', max_iter=25000, multi_class='multinomial', n_jobs=-1, l1_ratio=l1_param)
                            log_model = LogisticRegression(penalty='elasticnet', C=c_par, class_weight='balanced',
                                                           random_state=42, solver='saga', max_iter=5000, multi_class='ovr',
                                                           n_jobs=-1, l1_ratio=l1_param)
                            # compute weights (because unbalanced dataset)
                            weights = compute_sample_weight(class_weight='balanced', y=train_labels)
                            # lab_w = [[train_labels[it], weights[it]] for it in range(len(weights))]
                            # train model
                            log_model.fit(X=train_df, y=train_labels, sample_weight=weights)
                            # log_model.fit(X=train_data, y=train_labels)
                            # get predictions
                            # pred = log_model.predict_proba(test_data)
                            pred = log_model.predict(test_df)
                            # test_counter = Counter(test_labels)
                            # pred_counter = Counter(pred)
                            # print("Test Labels count: ", test_counter)
                            # print("Prediction Labels count: ", pred_counter)
                            # cl = log_model.classes_
                            # print("Predictions:", pred)
                            sc = round(log_model.score(X=test_df, y=test_labels) * 100, 2)
                            # print("Score: ", sc)
                            total_score.append(sc)
                            # print("Num", cl[0], train_labels.count(cl[0]))
                            # print("Num", cl[1], train_labels.count(cl[1]))
                            # print("Num", cl[2], train_labels.count(cl[2]))
                            # print("========================================\n")
                        else:
                            sc = -1
                            total_score.append(sc)

                    # print("Multimodal Mean score after", cv, 'folds with C =', c_par, ', L1Ratio =', l1_param, 'and',
                    #       num_bin, 'bins for family', family, ':', round(np.mean(total_score), 2), "%\n")

                    wr = csv.writer(result_file)
                    results_to_write = ['Multimodal', family, num_bin, l1_param, c_par, total_score,
                                        round(np.mean(total_score), 2)]
                    wr.writerow(results_to_write)

    result_file.close()
