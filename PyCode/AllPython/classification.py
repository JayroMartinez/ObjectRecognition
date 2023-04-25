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

    # model weights
    weight_filename = './results/weights_EMG_' + family + '.csv'
    weight_file = open(weight_filename, 'a')  # Open file in append mode
    weight_wr = csv.writer(weight_file)

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

        # model weight extraction and saving
        [weight_wr.writerow(x) for x in log_model.coef_]
    # model weight file close
    weight_file.close()
    # print(log_model.classes_)


    result = ['EMG']
    result.extend(params)
    result.append(total_score)
    result.append(round(np.mean(total_score), 2))
    print("RESULT:", result)

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

    # model weights
    weight_filename = './results/weights_Kin_' + family + '.csv'
    weight_file = open(weight_filename, 'a')  # Open file in append mode
    weight_wr = csv.writer(weight_file)

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

        # model weight extraction and saving
        [weight_wr.writerow(x) for x in log_model.coef_]
    # model weight file close
    weight_file.close()

    result = ['Kin']
    result.extend(params)
    result.append(total_score)
    result.append(round(np.mean(total_score), 2))
    print("RESULT:", result)

    return result


def tact_aux_classif(input_data):


    data = input_data[0]
    params = input_data[1]
    cv = input_data[2]

    family = params[0]
    num_bin = params[1]
    l1_param = params[2]
    c_par = params[3]

    total_score = []

    # model weights
    weight_filename = './results/weights_Tact_' + family + '.csv'
    weight_file = open(weight_filename, 'a')  # Open file in append mode
    weight_wr = csv.writer(weight_file)

    selected_df = data.loc[data['Family'] == family]  # select particular family
    tact_cols = ['rmo', 'mdo', 'rmi', 'mmo', 'pcim', 'ldd', 'rmm', 'rp', 'rdd', 'lmi', 'rdo', 'lmm', 'lp', 'rdm', 'ldm', 'ptip', 'idi', 'mdi', 'ido', 'mmm', 'ipi', 'mdm', 'idd', 'idm', 'imo', 'pdi', 'mmi', 'pdm', 'imm', 'mdd', 'pdii', 'mp', 'ptod', 'ptmd', 'tdo', 'pcid', 'imi', 'tmm', 'tdi', 'tmi', 'ptop', 'ptid', 'ptmp', 'tdm', 'tdd', 'tmo', 'pcip', 'ip', 'pcmp', 'rdi', 'ldi', 'lmo', 'pcmd', 'ldo', 'pdl', 'pdr', 'pdlo', 'lpo']
    selected_df.dropna(subset=tact_cols, axis=0, inplace=True)  # drop rows containing NaN values

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
            ep_tact_data = train_ep[tact_cols]
            ep_in_bins = np.array_split(ep_tact_data, num_bin)

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
            ep_tact_data = test_ep[tact_cols]
            ep_in_bins = np.array_split(ep_tact_data, num_bin)

            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    ep_bin_mean = [np.nanmean(x, axis=0) for x in ep_in_bins]  # size = [num_bins] X [num_sensors]
                    flat_ep_mean = list(
                        itertools.chain.from_iterable(ep_bin_mean))  # size = [num_bins X num_sensors] (unidimensional)
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

        # model weight extraction and saving
        [weight_wr.writerow(x) for x in log_model.coef_]
    # model weight file close
    weight_file.close()

    result = ['Tactile']
    result.extend(params)
    result.append(total_score)
    result.append(round(np.mean(total_score), 2))
    print("RESULT:", result)

    return result


def multiple_source_aux_classif(input_data):

    data = input_data[0]
    params = input_data[1]
    cv = input_data[2]

    family = params[0]
    num_bin = params[1]
    l1_param = params[2]
    c_par = params[3]

    # # for test and develop
    # family = 'Balls'
    # num_bin = 20
    # l1_param = 1
    # c_par = 0.1

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


def hierarchical_aux_classif(input_data):

    data = input_data[0]
    params = input_data[1]
    cv = input_data[2]

    family = params[0]
    top_C = params[1]

    kin_bins = 20
    kin_l1 = 1
    kin_c = 0.1

    emg_bins = 40
    emg_l1 = 0
    emg_c = 1.25

    tact_bins = 20
    tact_l1 = 0
    tact_c = 0.01

    # for develop and test
    kir = []
    kir_2 = []

    # kin_total_score = []
    # emg_total_score = []
    total_score = []

    selected_df = data.loc[data['Family'] == family]  # select particular family

    kin_cols = ['ThumbRotate', 'ThumbMPJ', 'ThumbIj', 'IndexMPJ', 'IndexPIJ',
                'MiddleMPJ', 'MiddlePIJ', 'RingMIJ', 'RingPIJ', 'PinkieMPJ',
                'PinkiePIJ', 'PalmArch', 'WristPitch', 'WristYaw', 'Index_Proj_J1_Z',
                'Pinkie_Proj_J1_Z', 'Ring_Proj_J1_Z', 'Middle_Proj_J1_Z',
                'Thumb_Proj_J1_Z']
    emg_cols = [col for col in selected_df.columns if ('flexion' in col) or ('extension' in col)]
    tact_cols = ['rmo', 'mdo', 'rmi', 'mmo', 'pcim', 'ldd', 'rmm', 'rp', 'rdd', 'lmi', 'rdo', 'lmm', 'lp', 'rdm',
                 'ldm', 'ptip', 'idi', 'mdi', 'ido', 'mmm', 'ipi', 'mdm', 'idd', 'idm', 'imo', 'pdi', 'mmi', 'pdm',
                 'imm', 'mdd', 'pdii', 'mp', 'ptod', 'ptmd', 'tdo', 'pcid', 'imi', 'tmm', 'tdi', 'tmi', 'ptop',
                 'ptid', 'ptmp', 'tdm', 'tdd', 'tmo', 'pcip', 'ip', 'pcmp', 'rdi', 'ldi', 'lmo', 'pcmd', 'ldo',
                 'pdl', 'pdr', 'pdlo', 'lpo']

    selected_df.dropna(axis=0, inplace=True)  # drop rows containing NaN values

    to_kfold = selected_df.drop_duplicates(
        subset=['EP total', 'Given Object'])  # only way I found to avoid overlapping

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    # WARNING: the skf.split returns the indexes
    for train, test in skf.split(to_kfold['EP total'].astype(int), to_kfold['Given Object'].astype(str)):

        train_eps = to_kfold.iloc[train]['EP total']  # because skf.split returns the indexes
        test_eps = to_kfold.iloc[test]['EP total']  # because skf.split returns the indexes

        kin_train_data = []
        emg_train_data = []
        tact_train_data = []
        train_labels = []

        trn_dropped = 0  # Number of dropped EPs in training dataset
        tst_dropped = 0  # Number of dropped EPs in test dataset

        for trn_iter in train_eps:

            train_ep = selected_df.loc[selected_df['EP total'] == trn_iter]

            ep_kin_data = train_ep[kin_cols]
            kin_in_bins = np.array_split(ep_kin_data, kin_bins)

            ep_emg_data = train_ep[emg_cols]
            emg_in_bins = np.array_split(ep_emg_data, emg_bins)

            ep_tact_data = train_ep[tact_cols]
            tact_in_bins = np.array_split(ep_tact_data, tact_bins)

            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:

                    kin_bin_mean = [np.nanmean(x, axis=0) for x in kin_in_bins]  # size = [num_bins] X [64]
                    flat_kin_mean = list(
                        itertools.chain.from_iterable(kin_bin_mean))  # size = [num_bins X 64] (unidimensional)

                    emg_bin_mean = [np.nanmean(x, axis=0) for x in emg_in_bins]  # size = [num_bins] X [64]
                    flat_emg_mean = list(
                        itertools.chain.from_iterable(emg_bin_mean))  # size = [num_bins X 64] (unidimensional)

                    tact_bin_mean = [np.nanmean(x, axis=0) for x in tact_in_bins]  # size = [num_bins] X [64]
                    flat_tact_mean = list(
                        itertools.chain.from_iterable(tact_bin_mean))  # size = [num_bins X 64] (unidimensional)

                    kin_train_data.append(flat_kin_mean)
                    emg_train_data.append(flat_emg_mean)
                    tact_train_data.append(flat_tact_mean)
                    train_labels.append(np.unique(train_ep['Given Object'])[0])

                except RuntimeWarning:
                    # print("Dropped EP", trn_iter, "from family ", family)
                    trn_dropped += 1

        kin_test_data = []
        emg_test_data = []
        tact_test_data = []

        test_labels = []

        for tst_iter in test_eps:

            test_ep = selected_df.loc[selected_df['EP total'] == tst_iter]

            ep_kin_data = test_ep[kin_cols]
            kin_in_bins = np.array_split(ep_kin_data, kin_bins)

            ep_emg_data = test_ep[emg_cols]
            emg_in_bins = np.array_split(ep_emg_data, emg_bins)

            ep_tact_data = test_ep[tact_cols]
            tact_in_bins = np.array_split(ep_tact_data, tact_bins)

            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:

                    kin_bin_mean = [np.nanmean(x, axis=0) for x in kin_in_bins]  # size = [num_bins] X [64]
                    flat_kin_mean = list(
                        itertools.chain.from_iterable(kin_bin_mean))  # size = [num_bins X 64] (unidimensional)

                    emg_bin_mean = [np.nanmean(x, axis=0) for x in emg_in_bins]  # size = [num_bins] X [64]
                    flat_emg_mean = list(
                        itertools.chain.from_iterable(emg_bin_mean))  # size = [num_bins X 64] (unidimensional)

                    tact_bin_mean = [np.nanmean(x, axis=0) for x in tact_in_bins]  # size = [num_bins] X [64]
                    flat_tact_mean = list(
                        itertools.chain.from_iterable(tact_bin_mean))  # size = [num_bins X 64] (unidimensional)

                    kin_test_data.append(flat_kin_mean)
                    emg_test_data.append(flat_emg_mean)
                    tact_test_data.append(flat_tact_mean)
                    test_labels.append(np.unique(test_ep['Given Object'])[0])

                except RuntimeWarning:
                    # print("Dropped EP", tst_iter, "from family ", family)
                    tst_dropped += 1

        # compute weights (because unbalanced dataset)
        weights = compute_sample_weight(class_weight='balanced', y=train_labels)

        # build kinematic model
        kin_log_model = LogisticRegression(penalty='elasticnet', C=kin_c, class_weight='balanced',
                                           random_state=42,
                                           solver='saga', max_iter=25000, multi_class='ovr', n_jobs=-1,
                                           l1_ratio=kin_l1)

        # train kinematic model
        kin_log_model.fit(X=kin_train_data, y=train_labels, sample_weight=weights)
        # sc = round(kin_log_model.score(X=kin_test_data, y=test_labels) * 100, 2)
        # kin_total_score.append(sc)

        # build EMG model
        emg_log_model = LogisticRegression(penalty='elasticnet', C=emg_c, class_weight='balanced',
                                           random_state=42,
                                           solver='saga', max_iter=25000, multi_class='ovr', n_jobs=-1,
                                           l1_ratio=emg_l1)

        # train EMG model
        emg_log_model.fit(X=emg_train_data, y=train_labels, sample_weight=weights)
        # sc = round(emg_log_model.score(X=emg_test_data, y=test_labels) * 100, 2)
        # emg_total_score.append(sc)

        # build Tactile model
        tact_log_model = LogisticRegression(penalty='elasticnet', C=tact_c, class_weight='balanced',
                                            random_state=42,
                                            solver='saga', max_iter=25000, multi_class='ovr', n_jobs=-1,
                                            l1_ratio=tact_l1)

        # train EMG model
        tact_log_model.fit(X=tact_train_data, y=train_labels, sample_weight=weights)
        # sc = round(emg_log_model.score(X=emg_test_data, y=test_labels) * 100, 2)
        # emg_total_score.append(sc)

        # get prediction probabilities from first layer to train second layer
        kin_model_pred_proba = kin_log_model.predict_proba(X=kin_train_data)
        emg_model_pred_proba = emg_log_model.predict_proba(X=emg_train_data)
        tact_model_pred_proba = tact_log_model.predict_proba(X=tact_train_data)

        pred_proba = np.concatenate([kin_model_pred_proba, emg_model_pred_proba, tact_model_pred_proba], axis=1)

        # build & train top layer classifier
        top_log_model = LogisticRegression(C=top_C, class_weight='balanced', random_state=42, solver='saga',
                                           max_iter=25000,
                                           multi_class='ovr', n_jobs=-1)
        top_log_model.fit(X=pred_proba, y=train_labels, sample_weight=weights)

        # get probabilities from first layer on test set to feed the second layer
        kin_test_pred = kin_log_model.predict_proba(X=kin_test_data)
        emg_test_pred = emg_log_model.predict_proba(X=emg_test_data)
        tact_test_pred = tact_log_model.predict_proba(X=tact_test_data)
        test_proba = np.concatenate([kin_test_pred, emg_test_pred, tact_test_pred], axis=1)

        # get prediction accuracy from second layer
        sc = round(top_log_model.score(X=test_proba, y=test_labels) * 100, 2)
        total_score.append(sc)

    result = ['Hierarchical']
    result.extend([family, '0', '0', top_C])
    result.append(total_score)
    result.append(round(np.mean(total_score), 2))

    return result


def emg_classification(data):

    families = np.unique(data['Family'])
    bins = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    l1VSl2 = [0, 0.25, 0.5, 0.75, 1]
    c_param = [0.01, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5]
    cv = 3

    # for testing
    # bins = 40
    # l1VSl2 = 0
    # c_param = 1.25
    # data_and_iter = [[data, [x, bins, l1VSl2, c_param], cv] for x in families]

    # we need to build the object to be iterated in the multiprocessing pool
    all_param = list(itertools.product(families, bins, l1VSl2, c_param))
    data_and_iter = [[data, x, cv] for x in all_param]

    result_file = open('./results/results_file.csv', 'a')  # Open file in append mode
    wr = csv.writer(result_file)

    # multiprocessing
    with Pool() as pool:

        result = pool.map_async(emg_aux_classif, data_and_iter)

        for res in result.get():
            # wr.writerow(res)
            a=1

    result_file.close()


def kinematic_classification(data):

    families = np.unique(data['Family'])
    bins = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    l1VSl2 = [0, 0.25, 0.5, 0.75, 1]
    c_param = [0.01, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5]
    cv = 3

    # for testing
    # bins = 20
    # l1VSl2 = 1
    # c_param = 0.1
    # data_and_iter = [[data, [x, bins, l1VSl2, c_param], cv] for x in families]

    # we need to build the object to be iterated in the multiprocessing pool
    all_param = list(itertools.product(families, bins, l1VSl2, c_param))
    data_and_iter = [[data, x, cv] for x in all_param]

    result_file = open('./results/results_file.csv', 'a')  # Open file in append mode
    wr = csv.writer(result_file)

    # multiprocessing
    with Pool() as pool:

        result = pool.map_async(kin_aux_classif, data_and_iter)

        for res in result.get():
            # wr.writerow(res)
            a=1

    result_file.close()


def tactile_classification(data):

    families = np.unique(data['Family'])
    # bins = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    # l1VSl2 = [0, 0.25, 0.5, 0.75, 1]
    # c_param = [0.01, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5]
    cv = 3

    # for testing
    bins = 20
    l1VSl2 = 0
    c_param = 0.01
    data_and_iter = [[data, [x, bins, l1VSl2, c_param], cv] for x in families]

    # we need to build the object to be iterated in the multiprocessing pool
    # all_param = list(itertools.product(families, bins, l1VSl2, c_param))
    # data_and_iter = [[data, x, cv] for x in all_param]

    result_file = open('./results/results_file.csv', 'a')  # Open file in append mode
    wr = csv.writer(result_file)

    # multiprocessing
    with Pool() as pool:

        result = pool.map_async(tact_aux_classif, data_and_iter)

        for res in result.get():
            wr.writerow(res)
            a=1

    result_file.close()


def multiple_source_classification(data):

    families = np.unique(data['Family'])
    bins = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    l1VSl2 = [0, 0.25, 0.5, 0.75, 1]
    c_param = [0.01, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5]
    cv = 3

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


def hierarchical_classification(data):

    families = np.unique(data['Family'])
    c_param = [0.01, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5]
    cv = 3

    # we need to build the object to be iterated in the multiprocessing pool
    all_param = list(itertools.product(families, c_param))
    data_and_iter = [[data, x, cv] for x in all_param]

    result_file = open('./results/results_file.csv', 'a')  # Open file in append mode
    wr = csv.writer(result_file)

    # multiprocessing
    with Pool() as pool:
        result = pool.map_async(hierarchical_aux_classif, data_and_iter)

        for res in result.get():
            wr.writerow(res)

    result_file.close()


def ep_seq_classification(data):

    all_eps = np.unique(data['EP'])

    families = np.unique(data['Family'])

    coincidences = []
    object = []

    for family in families:
        selected_df = data.loc[data['Family'] == family]  # select particular family
        eps_in_fam = np.unique(selected_df['EP'])

        weight_filename = './results/weights_EP_Labs_' + family + '.csv'
        weight_file = open(weight_filename, 'a')  # Open file in append mode
        weight_wr = csv.writer(weight_file)

        trials_to_iter = np.unique(selected_df['Trial num'].values)

        ep_seq = []
        objects = []
        for tr in trials_to_iter:
            trial_df = selected_df.loc[selected_df['Trial num'] == str(tr)]
            eps_in_trial = np.unique(trial_df['EP'])

            coincidences = []
            for x in all_eps:
                check = x in eps_in_trial
                coincidences.append(int(check))

            ep_seq.append(coincidences)
            objects.append(np.unique(selected_df.loc[selected_df['Trial num'] == str(tr)]['Given Object'])[0])

        ep_df = pd.DataFrame(data=ep_seq, columns=all_eps)
        ep_df['Given Object'] = objects

        acc = []

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        # WARNING: the skf.split returns the indexes
        for train, test in skf.split(ep_df.drop('Given Object', axis=1), ep_df['Given Object'].astype(str)):

            model = LogisticRegression(random_state=42).fit(ep_df.iloc[train].drop('Given Object', axis=1), ep_df.iloc[train]['Given Object'])

            predicted = model.predict(ep_df.iloc[test].drop('Given Object', axis=1))
            labels = ep_df.iloc[test]['Given Object']

            hits = [int(list(labels.values)[x] == list(predicted)[x]) for x in range(0, len(predicted))]
            acc.append(round(np.sum(hits)*100/len(predicted), 2))

            # model weight extraction and saving
            [weight_wr.writerow(x) for x in model.coef_]

        print("Mean accuracy for family", family, "is", round(np.mean(acc), 2), "%")

        # model weight file close
        weight_file.close()


def ep_dur_classification(data):

    families = np.unique(data['Family'])

    for family in families:

        selected_df = data.loc[data['Family'] == family]  # select particular family

        weight_filename = './results/weights_EP_Dur_' + family + '.csv'
        weight_file = open(weight_filename, 'a')  # Open file in append mode
        weight_wr = csv.writer(weight_file)

        acc = []

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        # WARNING: the skf.split returns the indexes
        for train, test in skf.split(selected_df.drop(['Object', 'Family'], axis=1), selected_df['Object'].astype(str)):

            model = LogisticRegression(random_state=42).fit(selected_df.iloc[train].drop(['Object', 'Family'], axis=1), selected_df.iloc[train]['Object'])

            predicted = model.predict(selected_df.iloc[test].drop(['Object', 'Family'], axis=1))
            labels = selected_df.iloc[test]['Object']

            hits = [int(list(labels.values)[x] == list(predicted)[x]) for x in range(0, len(predicted))]
            acc.append(round(np.sum(hits)*100/len(predicted), 2))

            # model weight extraction and saving
            [weight_wr.writerow(x) for x in model.coef_]

        print("Mean accuracy for family", family, "is", round(np.mean(acc), 2), "%")

        # model weight file close
        weight_file.close()
