import numpy as np
import itertools
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from collections import Counter
from sklearn.utils.class_weight import compute_sample_weight
import csv
from scipy.stats import zscore
import pandas as pd
from multiprocessing.pool import Pool

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def emg_aux_classif(input_data):


    data = input_data[0]
    params = input_data[1]
    cv = input_data[2]

    family = params[0]
    num_bin = params[1]
    l1_param = params[2]
    c_par = params[3]

    total_score = []

    selected_df = data.loc[data['Family'] == family]  # select particular family
    emg_cols = [col for col in selected_df.columns if ('flexion' in col) or ('extension' in col)]
    selected_df.dropna(subset=emg_cols, axis=0, inplace=True)  # drop rows containing NaN values

    to_kfold = selected_df.drop_duplicates(subset=['EP total', 'Given Object'])  # only way I found to avoid overlapping

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    # WARNING: the skf.split returns the indexes
    for train, test in skf.split(to_kfold['EP total'].astype(int), to_kfold['Given Object'].astype(str)):

        train_eps = to_kfold.iloc[train]['EP total']  # because skf.split returns the indexes
        test_eps = to_kfold.iloc[test]['EP total']  # because skf.split returns the indexes

        train_data = []
        train_labels = []

        dropped = 0  # Number of dropped EPs

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
                    # print("Dropped EP", trn_iter, "from family ", family)
                    dropped += 1

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
                    # print("Dropped EP", tst_iter, "from family ", family)
                    dropped += 1

        # build model
        log_model = LogisticRegression(penalty='elasticnet', C=c_par, class_weight='balanced', random_state=42, solver='saga', max_iter=25000, multi_class='ovr', n_jobs=-1, l1_ratio=l1_param)
        # compute weights (because unbalanced dataset)
        weights = compute_sample_weight(class_weight='balanced', y=train_labels)
        # train model
        log_model.fit(X=train_data, y=train_labels, sample_weight=weights)
        sc = round(log_model.score(X=test_data, y=test_labels)*100, 2)
        total_score.append(sc)

    result = ['EMG']
    result.extend(params)
    result.append(total_score)
    result.append(round(np.mean(total_score), 2))

    return result


def kin_aux_classif(input_data):


    data = input_data[0]
    params = input_data[1]
    cv = input_data[2]

    family = params[0]
    num_bin = params[1]
    l1_param = params[2]
    c_par = params[3]

    total_score = []

    selected_df = data.loc[data['Family'] == family]  # select particular family
    selected_df = data.loc[data['Family'] == family]  # select particular family
    kin_cols = ['ThumbRotate', 'ThumbMPJ', 'ThumbIj', 'IndexMPJ', 'IndexPIJ',
                'MiddleMPJ', 'MiddlePIJ', 'RingMIJ', 'RingPIJ', 'PinkieMPJ',
                'PinkiePIJ', 'PalmArch', 'WristPitch', 'WristYaw', 'Index_Proj_J1_Z',
                'Pinkie_Proj_J1_Z', 'Ring_Proj_J1_Z', 'Middle_Proj_J1_Z',
                'Thumb_Proj_J1_Z']
    selected_df.dropna(subset=kin_cols, axis=0, inplace=True)  # drop rows containing NaN values

    to_kfold = selected_df.drop_duplicates(subset=['EP total', 'Given Object'])  # only way I found to avoid overlapping

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    # WARNING: the skf.split returns the indexes
    for train, test in skf.split(to_kfold['EP total'].astype(int), to_kfold['Given Object'].astype(str)):

        train_eps = to_kfold.iloc[train]['EP total']  # because skf.split returns the indexes
        test_eps = to_kfold.iloc[test]['EP total']  # because skf.split returns the indexes

        train_data = []
        train_labels = []

        dropped = 0  # Number of dropped EPs

        # take each ep, create bins & compute mean
        for trn_iter in train_eps:

            train_ep = selected_df.loc[selected_df['EP total'] == trn_iter]
            ep_kin_data = train_ep[kin_cols]
            ep_in_bins = np.array_split(ep_kin_data, num_bin)

            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    ep_bin_mean = [np.nanmean(x, axis=0) for x in ep_in_bins]  # size = [num_bins] X [64]
                    flat_ep_mean = list(
                        itertools.chain.from_iterable(ep_bin_mean))  # size = [num_bins X 64] (unidimensional)
                    train_data.append(flat_ep_mean)
                    train_labels.append(np.unique(train_ep['Given Object'])[0])
                except RuntimeWarning:
                    # print("Dropped EP", trn_iter, "from family ", family)
                    dropped += 1

        test_data = []
        test_labels = []

        for tst_iter in test_eps:

            test_ep = selected_df.loc[selected_df['EP total'] == tst_iter]
            ep_kin_data = test_ep[kin_cols]
            ep_in_bins = np.array_split(ep_kin_data, num_bin)

            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    ep_bin_mean = [np.nanmean(x, axis=0) for x in ep_in_bins]  # size = [num_bins] X [64]
                    flat_ep_mean = list(
                        itertools.chain.from_iterable(ep_bin_mean))  # size = [num_bins X 64] (unidimensional)
                    test_data.append(flat_ep_mean)
                    test_labels.append(np.unique(test_ep['Given Object'])[0])
                except RuntimeWarning:
                    # print("Dropped EP", tst_iter, "from family ", family)
                    dropped += 1

        # build model
        log_model = LogisticRegression(penalty='elasticnet', C=c_par, class_weight='balanced', random_state=42,
                                       solver='saga', max_iter=25000, multi_class='ovr', n_jobs=-1, l1_ratio=l1_param)
        # compute weights (because unbalanced dataset)
        weights = compute_sample_weight(class_weight='balanced', y=train_labels)
        # train model
        log_model.fit(X=train_data, y=train_labels, sample_weight=weights)
        sc = round(log_model.score(X=test_data, y=test_labels) * 100, 2)
        total_score.append(sc)

    result = ['Kin']
    result.extend(params)
    result.append(total_score)
    result.append(round(np.mean(total_score), 2))

    return result


def multiple_source_aux_classif(input_data):

    data = input_data[0]
    params = input_data[1]
    cv = input_data[2]

    family = params[0]
    num_bin = params[1]
    l1_param = params[2]
    c_par = params[3]

    total_score = []

    selected_df = data.loc[data['Family'] == family]  # select particular family
    selected_df.dropna(axis=0, inplace=True)  # drop rows containing NaN values

    to_kfold = selected_df.drop_duplicates(subset=['EP total', 'Given Object'])  # only way I found to avoid overlapping

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    # WARNING: the skf.split returns the indexes
    for train, test in skf.split(to_kfold['EP total'].astype(int), to_kfold['Given Object'].astype(str)):

        train_eps = to_kfold.iloc[train]['EP total']  # because skf.split returns the indexes
        test_eps = to_kfold.iloc[test]['EP total']  # because skf.split returns the indexes

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
                    flat_ep_mean = list(
                        itertools.chain.from_iterable(ep_bin_mean))  # size = [num_bins X 64] (unidimensional)
                    train_data.append(flat_ep_mean)
                    train_labels.append(np.unique(train_ep['Given Object'])[0])
                except RuntimeWarning:
                    # print("Dropped EP", trn_iter, "from family ", family)
                    dropped += 1

        test_data = []
        test_labels = []

        for tst_iter in test_eps:

            test_ep = selected_df.loc[selected_df['EP total'] == tst_iter]
            ep_numeric_data = test_ep.select_dtypes(include='float64')
            ep_in_bins = np.array_split(ep_numeric_data, num_bin)

            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    ep_bin_mean = [np.nanmean(x, axis=0) for x in ep_in_bins]
                    flat_ep_mean = list(
                        itertools.chain.from_iterable(ep_bin_mean))
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

        if test_df.shape[0] > 0:
            # build model
            log_model = LogisticRegression(penalty='elasticnet', C=c_par, class_weight='balanced',
                                           random_state=42, solver='saga', max_iter=25000,
                                           multi_class='ovr',
                                           n_jobs=-1, l1_ratio=l1_param)
            # compute weights (because unbalanced dataset)
            weights = compute_sample_weight(class_weight='balanced', y=train_labels)
            # train model
            log_model.fit(X=train_df, y=train_labels, sample_weight=weights)
            sc = round(log_model.score(X=test_df, y=test_labels) * 100, 2)
            total_score.append(sc)

        else:
            sc = 0
            total_score.append(sc)

    result = ['Multimodal']
    result.extend(params)
    result.append(total_score)
    result.append(round(np.mean(total_score), 2))

    return result


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

    # we need to build the object to be iterated in the multiprocessing pool
    all_param = list(itertools.product(families, bins, l1VSl2, c_param))
    data_and_iter = [[data, x, cv] for x in all_param]

    result_file = open('./results/results_file.csv', 'a')  # Open file in append mode
    wr = csv.writer(result_file)

    # multiprocessing
    with Pool() as pool:

        result = pool.map_async(emg_aux_classif, data_and_iter)

        for res in result.get():
            wr.writerow(res)

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

    # we need to build the object to be iterated in the multiprocessing pool
    all_param = list(itertools.product(families, bins, l1VSl2, c_param))
    data_and_iter = [[data, x, cv] for x in all_param]

    result_file = open('./results/results_file.csv', 'a')  # Open file in append mode
    wr = csv.writer(result_file)

    # multiprocessing
    with Pool() as pool:

        result = pool.map_async(kin_aux_classif, data_and_iter)

        for res in result.get():
            wr.writerow(res)

    result_file.close()


def multiple_source_classification(data):

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

    # we need to build the object to be iterated in the multiprocessing pool
    all_param = list(itertools.product(families, bins, l1VSl2, c_param))
    data_and_iter = [[data, x, cv] for x in all_param]

    result_file = open('./results/results_file.csv', 'a')  # Open file in append mode
    wr = csv.writer(result_file)

    # multiprocessing
    with Pool() as pool:
        result = pool.map_async(multiple_source_aux_classif, data_and_iter)

        for res in result.get():
            wr.writerow(res)

    result_file.close()
