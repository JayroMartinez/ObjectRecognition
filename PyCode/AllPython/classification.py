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
import random

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
    random_score = []

    # model weights
    weight_filename = './results/Raw/weights/w_EMG_' + family + '.csv'
    weight_file = open(weight_filename, 'a')  # Open file in append mode
    weight_wr = csv.writer(weight_file)

    selected_df = data.loc[data['Family'] == family]  # select particular family
    emg_cols = [col for col in selected_df.columns if ('flexion' in col) or ('extension' in col)]
    selected_df.dropna(subset=emg_cols, axis=0, inplace=True)  # drop rows containing NaN values

    to_kfold = selected_df.drop_duplicates(subset=['Trial num', 'Given Object'])  # only way I found to avoid overlapping

    random_states = [42, 43, 44]

    for rnd_st in random_states:

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=rnd_st)
        # WARNING: the skf.split returns the indexes
        for train, test in skf.split(to_kfold['Trial num'].astype(int), to_kfold['Given Object'].astype(str)):

            train_trials = to_kfold.iloc[train]['Trial num']  # because skf.split returns the indexes
            test_trials = to_kfold.iloc[test]['Trial num']  # because skf.split returns the indexes

            train_data = []
            train_labels = []

            dropped = 0  # Number of dropped EPs

            # take each ep, create bins & compute mean
            for trn_iter in train_trials:

                train_tri = selected_df.loc[selected_df['Trial num'] == trn_iter]
                tr_emg_data = train_tri[emg_cols]
                tr_in_bins = np.array_split(tr_emg_data, num_bin)

                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    try:
                        tr_bin_mean = [np.nanmean(x, axis=0) for x in tr_in_bins]  # size = [num_bins] X [64]
                        flat_tr_mean = list(itertools.chain.from_iterable(tr_bin_mean))  # size = [num_bins X 64] (unidimensional)
                        train_data.append(flat_tr_mean)
                        train_labels.append(np.unique(train_tri['Given Object'])[0])
                    except RuntimeWarning:
                        # print("Dropped EP", trn_iter, "from family ", family)
                        dropped += 1

            test_data = []
            test_labels = []

            for tst_iter in test_trials:

                test_tri = selected_df.loc[selected_df['Trial num'] == tst_iter]
                tst_emg_data = test_tri[emg_cols]
                tst_in_bins = np.array_split(tst_emg_data, num_bin)

                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    try:
                        tst_bin_mean = [np.nanmean(x, axis=0) for x in tst_in_bins]  # size = [num_bins] X [64]
                        flat_tst_mean = list(itertools.chain.from_iterable(tst_bin_mean))  # size = [num_bins X 64] (unidimensional)
                        test_data.append(flat_tst_mean)
                        test_labels.append(np.unique(test_tri['Given Object'])[0])
                    except RuntimeWarning:
                        # print("Dropped EP", tst_iter, "from family ", family)
                        dropped += 1

            # build model
            log_model = LogisticRegression(penalty='elasticnet', C=c_par, class_weight='balanced', random_state=rnd_st, solver='saga', max_iter=25000, multi_class='ovr', n_jobs=-1, l1_ratio=l1_param)
            # compute weights (because unbalanced dataset)
            weights = compute_sample_weight(class_weight='balanced', y=train_labels)
            # train model
            log_model.fit(X=train_data, y=train_labels, sample_weight=weights)
            sc = round(log_model.score(X=test_data, y=test_labels)*100, 2)
            total_score.append(sc)
            # model weight extraction and saving
            [weight_wr.writerow(x) for x in log_model.coef_]

            # random shuffle model
            random.shuffle(train_labels)
            random.shuffle(test_labels)
            weights = compute_sample_weight(class_weight='balanced', y=train_labels)
            log_model.fit(X=train_data, y=train_labels, sample_weight=weights)
            rnd_sc = round(log_model.score(X=test_data, y=test_labels) * 100, 2)
            random_score.append(rnd_sc)

    # model weight file close
    weight_file.close()
    # print(log_model.classes_)


    result = ['EMG']
    result.extend(params)
    result.append(total_score)
    result.append(round(np.mean(total_score), 2))
    # print("RESULT:", result)
    random_result = ['EMG_Rand']
    random_result.extend(params)
    random_result.append(random_score)
    random_result.append(round(np.mean(random_score), 2))

    return [result, random_result]


def kin_aux_classif(input_data):


    data = input_data[0]
    params = input_data[1]
    cv = input_data[2]

    family = params[0]
    num_bin = params[1]
    l1_param = params[2]
    c_par = params[3]

    total_score = []
    random_score = []

    # model weights
    weight_filename = './results/Raw/weights/w_Kin_' + family + '.csv'
    weight_file = open(weight_filename, 'a')  # Open file in append mode
    weight_wr = csv.writer(weight_file)

    selected_df = data.loc[data['Family'] == family]  # select particular family
    kin_cols = ['ThumbRotate', 'ThumbMPJ', 'ThumbIj', 'IndexMPJ', 'IndexPIJ',
                'MiddleMPJ', 'MiddlePIJ', 'RingMIJ', 'RingPIJ', 'PinkieMPJ',
                'PinkiePIJ', 'PalmArch', 'WristPitch', 'WristYaw', 'Index_Proj_J1_Z',
                'Pinkie_Proj_J1_Z', 'Ring_Proj_J1_Z', 'Middle_Proj_J1_Z',
                'Thumb_Proj_J1_Z']
    selected_df.dropna(subset=kin_cols, axis=0, inplace=True)  # drop rows containing NaN values

    to_kfold = selected_df.drop_duplicates(subset=['Trial num', 'Given Object'])  # only way I found to avoid overlapping

    random_states = [42, 43, 44]
    for rnd_st in random_states:

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=rnd_st)
        # WARNING: the skf.split returns the indexes
        for train, test in skf.split(to_kfold['Trial num'].astype(int), to_kfold['Given Object'].astype(str)):

            train_trials = to_kfold.iloc[train]['Trial num']  # because skf.split returns the indexes
            test_trials = to_kfold.iloc[test]['Trial num']  # because skf.split returns the indexes

            train_data = []
            train_labels = []

            dropped = 0  # Number of dropped EPs

            # take each ep, create bins & compute mean
            for trn_iter in train_trials:

                train_tri = selected_df.loc[selected_df['Trial num'] == trn_iter]
                tr_kin_data = train_tri[kin_cols]
                tr_in_bins = np.array_split(tr_kin_data, num_bin)

                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    try:
                        tr_bin_mean = [np.nanmean(x, axis=0) for x in tr_in_bins]  # size = [num_bins] X [64]
                        flat_tr_mean = list(
                            itertools.chain.from_iterable(tr_bin_mean))  # size = [num_bins X 64] (unidimensional)
                        train_data.append(flat_tr_mean)
                        train_labels.append(np.unique(train_tri['Given Object'])[0])
                    except RuntimeWarning:
                        # print("Dropped EP", trn_iter, "from family ", family)
                        dropped += 1

            test_data = []
            test_labels = []

            for tst_iter in test_trials:

                test_tri = selected_df.loc[selected_df['Trial num'] == tst_iter]
                tst_kin_data = test_tri[kin_cols]
                tst_in_bins = np.array_split(tst_kin_data, num_bin)

                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    try:
                        tst_bin_mean = [np.nanmean(x, axis=0) for x in tst_in_bins]  # size = [num_bins] X [64]
                        flat_tst_mean = list(
                            itertools.chain.from_iterable(tst_bin_mean))  # size = [num_bins X 64] (unidimensional)
                        test_data.append(flat_tst_mean)
                        test_labels.append(np.unique(test_tri['Given Object'])[0])
                    except RuntimeWarning:
                        # print("Dropped EP", tst_iter, "from family ", family)
                        dropped += 1

            # build model
            log_model = LogisticRegression(penalty='elasticnet', C=c_par, class_weight='balanced', random_state=rnd_st,
                                           solver='saga', max_iter=25000, multi_class='ovr', n_jobs=-1, l1_ratio=l1_param)
            # compute weights (because unbalanced dataset)
            weights = compute_sample_weight(class_weight='balanced', y=train_labels)
            # train model
            log_model.fit(X=train_data, y=train_labels, sample_weight=weights)
            sc = round(log_model.score(X=test_data, y=test_labels) * 100, 2)
            total_score.append(sc)
            # model weight extraction and saving
            [weight_wr.writerow(x) for x in log_model.coef_]

            # random shuffle model
            random.shuffle(train_labels)
            random.shuffle(test_labels)
            weights = compute_sample_weight(class_weight='balanced', y=train_labels)
            log_model.fit(X=train_data, y=train_labels, sample_weight=weights)
            rnd_sc = round(log_model.score(X=test_data, y=test_labels) * 100, 2)
            random_score.append(rnd_sc)

    # model weight file close
    weight_file.close()

    result = ['Kin']
    result.extend(params)
    result.append(total_score)
    result.append(round(np.mean(total_score), 2))
    # print("RESULT:", result)
    random_result = ['Kin_Rand']
    random_result.extend(params)
    random_result.append(random_score)
    random_result.append(round(np.mean(random_score), 2))

    return [result, random_result]


def tact_aux_classif(input_data):


    data = input_data[0]
    params = input_data[1]
    cv = input_data[2]

    family = params[0]
    num_bin = params[1]
    l1_param = params[2]
    c_par = params[3]

    total_score = []
    random_score = []

    # model weights
    weight_filename = './results/Raw/weights/w_Tact_' + family + '.csv'
    weight_file = open(weight_filename, 'a')  # Open file in append mode
    weight_wr = csv.writer(weight_file)

    selected_df = data.loc[data['Family'] == family]  # select particular family
    tact_cols = ['rmo', 'mdo', 'rmi', 'mmo', 'pcim', 'ldd', 'rmm', 'rp', 'rdd', 'lmi', 'rdo', 'lmm', 'lp', 'rdm', 'ldm', 'ptip', 'idi', 'mdi', 'ido', 'mmm', 'ipi', 'mdm', 'idd', 'idm', 'imo', 'pdi', 'mmi', 'pdm', 'imm', 'mdd', 'pdii', 'mp', 'ptod', 'ptmd', 'tdo', 'pcid', 'imi', 'tmm', 'tdi', 'tmi', 'ptop', 'ptid', 'ptmp', 'tdm', 'tdd', 'tmo', 'pcip', 'ip', 'pcmp', 'rdi', 'ldi', 'lmo', 'pcmd', 'ldo', 'pdl', 'pdr', 'pdlo', 'lpo']
    selected_df.dropna(subset=tact_cols, axis=0, inplace=True)  # drop rows containing NaN values

    to_kfold = selected_df.drop_duplicates(subset=['Trial num', 'Given Object'])  # only way I found to avoid overlapping

    random_states = [42, 43, 44]
    for rnd_st in random_states:

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=rnd_st)
        # WARNING: the skf.split returns the indexes
        for train, test in skf.split(to_kfold['Trial num'].astype(int), to_kfold['Given Object'].astype(str)):

            train_trials = to_kfold.iloc[train]['Trial num']  # because skf.split returns the indexes
            test_trials = to_kfold.iloc[test]['Trial num']  # because skf.split returns the indexes

            train_data = []
            train_labels = []

            dropped = 0  # Number of dropped EPs

            # take each ep, create bins & compute mean
            for trn_iter in train_trials:

                train_tri = selected_df.loc[selected_df['Trial num'] == trn_iter]
                tr_tact_data = train_tri[tact_cols]
                tr_in_bins = np.array_split(tr_tact_data, num_bin)

                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    try:
                        tr_bin_mean = [np.nanmean(x, axis=0) for x in tr_in_bins]  # size = [num_bins] X [64]
                        flat_tr_mean = list(
                            itertools.chain.from_iterable(tr_bin_mean))  # size = [num_bins X 64] (unidimensional)
                        train_data.append(flat_tr_mean)
                        train_labels.append(np.unique(train_tri['Given Object'])[0])
                    except RuntimeWarning:
                        # print("Dropped EP", trn_iter, "from family ", family)
                        dropped += 1

            test_data = []
            test_labels = []

            for tst_iter in test_trials:

                test_tri = selected_df.loc[selected_df['Trial num'] == tst_iter]
                tst_tact_data = test_tri[tact_cols]
                tst_in_bins = np.array_split(tst_tact_data, num_bin)

                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    try:
                        tst_bin_mean = [np.nanmean(x, axis=0) for x in tst_in_bins]  # size = [num_bins] X [num_sensors]
                        flat_tst_mean = list(
                            itertools.chain.from_iterable(tst_bin_mean))  # size = [num_bins X num_sensors] (unidimensional)
                        test_data.append(flat_tst_mean)
                        test_labels.append(np.unique(test_tri['Given Object'])[0])
                    except RuntimeWarning:
                        # print("Dropped EP", tst_iter, "from family ", family)
                        dropped += 1

            # build model
            log_model = LogisticRegression(penalty='elasticnet', C=c_par, class_weight='balanced', random_state=rnd_st,
                                           solver='saga', max_iter=25000, multi_class='ovr', n_jobs=-1, l1_ratio=l1_param)
            # compute weights (because unbalanced dataset)
            weights = compute_sample_weight(class_weight='balanced', y=train_labels)
            # train model
            log_model.fit(X=train_data, y=train_labels, sample_weight=weights)
            sc = round(log_model.score(X=test_data, y=test_labels) * 100, 2)
            total_score.append(sc)
            # model weight extraction and saving
            [weight_wr.writerow(x) for x in log_model.coef_]

            # random shuffle model
            random.shuffle(train_labels)
            random.shuffle(test_labels)
            weights = compute_sample_weight(class_weight='balanced', y=train_labels)
            log_model.fit(X=train_data, y=train_labels, sample_weight=weights)
            rnd_sc = round(log_model.score(X=test_data, y=test_labels) * 100, 2)
            random_score.append(rnd_sc)

    # model weight file close
    weight_file.close()

    result = ['Tactile']
    result.extend(params)
    result.append(total_score)
    result.append(round(np.mean(total_score), 2))
    # print("RESULT:", result)
    random_result = ['Tact_Rand']
    random_result.extend(params)
    random_result.append(random_score)
    random_result.append(round(np.mean(random_score), 2))

    return [result, random_result]


def multiple_source_aux_classif(input_data):

    data = input_data[0]
    params = input_data[1]
    cv = input_data[2]

    family = params[0]
    num_bin = params[1]
    l1_param = params[2]
    c_par = params[3]

    # # for test and develop
    # family = 'Ball'
    # num_bin = 20
    # l1_param = 1
    # c_par = 0.1

    kin_cols = ['ThumbRotate', 'ThumbMPJ', 'ThumbIj', 'IndexMPJ', 'IndexPIJ',
                'MiddleMPJ', 'MiddlePIJ', 'RingMIJ', 'RingPIJ', 'PinkieMPJ',
                'PinkiePIJ', 'PalmArch', 'WristPitch', 'WristYaw', 'Index_Proj_J1_Z',
                'Pinkie_Proj_J1_Z', 'Ring_Proj_J1_Z', 'Middle_Proj_J1_Z',
                'Thumb_Proj_J1_Z']
    emg_cols = [col for col in data.columns if ('flexion' in col) or ('extension' in col)]
    tact_cols = ['rmo', 'mdo', 'rmi', 'mmo', 'pcim', 'ldd', 'rmm', 'rp', 'rdd', 'lmi', 'rdo', 'lmm', 'lp', 'rdm',
                 'ldm', 'ptip', 'idi', 'mdi', 'ido', 'mmm', 'ipi', 'mdm', 'idd', 'idm', 'imo', 'pdi', 'mmi', 'pdm',
                 'imm', 'mdd', 'pdii', 'mp', 'ptod', 'ptmd', 'tdo', 'pcid', 'imi', 'tmm', 'tdi', 'tmi', 'ptop',
                 'ptid', 'ptmp', 'tdm', 'tdd', 'tmo', 'pcip', 'ip', 'pcmp', 'rdi', 'ldi', 'lmo', 'pcmd', 'ldo',
                 'pdl', 'pdr', 'pdlo', 'lpo']

    #########
    ## TEST NOT ALL SOURCES
    #########
    data.drop(columns=tact_cols, inplace=True)

    total_score = []
    random_score = []

    selected_df = data.loc[data['Family'] == family]  # select particular family
    selected_df.dropna(axis=0, inplace=True)  # drop rows containing NaN values

    to_kfold = selected_df.drop_duplicates(subset=['Trial num', 'Given Object'])  # only way I found to avoid overlapping

    random_states = [42, 43, 44]
    for rnd_st in random_states:

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=rnd_st)
        # WARNING: the skf.split returns the indexes
        for train, test in skf.split(to_kfold['Trial num'].astype(int), to_kfold['Given Object'].astype(str)):

            train_trials = to_kfold.iloc[train]['Trial num']  # because skf.split returns the indexes
            test_trials = to_kfold.iloc[test]['Trial num']  # because skf.split returns the indexes

            train_data = []
            train_labels = []

            dropped = 0  # Number of dropped EPs

            # take each ep, create bins & compute mean
            for trn_iter in train_trials:

                train_tri = selected_df.loc[selected_df['Trial num'] == trn_iter]
                tr_numeric_data = train_tri.select_dtypes(include='float64')
                tr_in_bins = np.array_split(tr_numeric_data, num_bin)

                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    try:
                        tr_bin_mean = [np.nanmean(x, axis=0) for x in tr_in_bins]  # size = [num_bins] X [64]
                        flat_tr_mean = list(
                            itertools.chain.from_iterable(tr_bin_mean))  # size = [num_bins X 64] (unidimensional)
                        train_data.append(flat_tr_mean)
                        train_labels.append(np.unique(train_tri['Given Object'])[0])
                    except RuntimeWarning:
                        # print("Dropped EP", trn_iter, "from family ", family)
                        dropped += 1

            test_data = []
            test_labels = []

            for tst_iter in test_trials:

                test_tri = selected_df.loc[selected_df['Trial num'] == tst_iter]
                tst_numeric_data = test_tri.select_dtypes(include='float64')
                tst_in_bins = np.array_split(tst_numeric_data, num_bin)

                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    try:
                        tst_bin_mean = [np.nanmean(x, axis=0) for x in tst_in_bins]
                        flat_tst_mean = list(
                            itertools.chain.from_iterable(tst_bin_mean))
                        test_data.append(flat_tst_mean)
                        test_labels.append(np.unique(test_tri['Given Object'])[0])
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
                                               random_state=rnd_st, solver='saga', max_iter=25000,
                                               multi_class='ovr',
                                               n_jobs=-1, l1_ratio=l1_param)
                # compute weights (because unbalanced dataset)
                weights = compute_sample_weight(class_weight='balanced', y=train_labels)
                # train model
                log_model.fit(X=train_df, y=train_labels, sample_weight=weights)
                sc = round(log_model.score(X=test_df, y=test_labels) * 100, 2)
                total_score.append(sc)

                # random shuffle model
                # random.shuffle(train_labels)
                # random.shuffle(test_labels)
                # weights = compute_sample_weight(class_weight='balanced', y=train_labels)
                # log_model.fit(X=train_data, y=train_labels, sample_weight=weights)
                # rnd_sc = round(log_model.score(X=test_data, y=test_labels) * 100, 2)
                # random_score.append(rnd_sc)

            else:
                sc = 0
                total_score.append(sc)
                # random_score.append(sc)

    result = ['Multimodal_KE']
    result.extend(params)
    result.append(total_score)
    result.append(round(np.mean(total_score), 2))
    # random_result = ['Multimodal_Rand']
    # random_result.extend(params)
    # random_result.append(random_score)
    # random_result.append(round(np.mean(random_score), 2))
    #
    # return [result, random_result]
    return result


def hierarchical_aux_classif(input_data):

    data = input_data[0]
    params = input_data[1]
    cv = input_data[2]

    family = params[0]
    top_C = params[1]

    kin_bins = 40
    kin_l1 = 0.25
    kin_c = 0.1

    emg_bins = 10
    emg_l1 = 0
    emg_c = 1.5

    # tact_bins = 5
    # tact_l1 = 0.5
    # tact_c = 0.25

    # kin_total_score = []
    # emg_total_score = []
    total_score = []
    random_score = []

    selected_df = data.loc[data['Family'] == family]  # select particular family

    kin_cols = ['ThumbRotate', 'ThumbMPJ', 'ThumbIj', 'IndexMPJ', 'IndexPIJ',
                'MiddleMPJ', 'MiddlePIJ', 'RingMIJ', 'RingPIJ', 'PinkieMPJ',
                'PinkiePIJ', 'PalmArch', 'WristPitch', 'WristYaw', 'Index_Proj_J1_Z',
                'Pinkie_Proj_J1_Z', 'Ring_Proj_J1_Z', 'Middle_Proj_J1_Z',
                'Thumb_Proj_J1_Z']
    emg_cols = [col for col in selected_df.columns if ('flexion' in col) or ('extension' in col)]
    # tact_cols = ['rmo', 'mdo', 'rmi', 'mmo', 'pcim', 'ldd', 'rmm', 'rp', 'rdd', 'lmi', 'rdo', 'lmm', 'lp', 'rdm',
    #              'ldm', 'ptip', 'idi', 'mdi', 'ido', 'mmm', 'ipi', 'mdm', 'idd', 'idm', 'imo', 'pdi', 'mmi', 'pdm',
    #              'imm', 'mdd', 'pdii', 'mp', 'ptod', 'ptmd', 'tdo', 'pcid', 'imi', 'tmm', 'tdi', 'tmi', 'ptop',
    #              'ptid', 'ptmp', 'tdm', 'tdd', 'tmo', 'pcip', 'ip', 'pcmp', 'rdi', 'ldi', 'lmo', 'pcmd', 'ldo',
    #              'pdl', 'pdr', 'pdlo', 'lpo']

    selected_df.dropna(axis=0, inplace=True)  # drop rows containing NaN values

    to_kfold = selected_df.drop_duplicates(
        subset=['Trial num', 'Given Object'])  # only way I found to avoid overlapping

    random_states = [42, 43, 44]

    for rnd_st in random_states:

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=rnd_st)
        # WARNING: the skf.split returns the indexes
        for train, test in skf.split(to_kfold['Trial num'].astype(int), to_kfold['Given Object'].astype(str)):

            train_eps = to_kfold.iloc[train]['Trial num']  # because skf.split returns the indexes
            test_eps = to_kfold.iloc[test]['Trial num']  # because skf.split returns the indexes

            kin_train_data = []
            emg_train_data = []
            # tact_train_data = []
            train_labels = []

            trn_dropped = 0  # Number of dropped EPs in training dataset
            tst_dropped = 0  # Number of dropped EPs in test dataset

            for trn_iter in train_eps:

                train_ep = selected_df.loc[selected_df['Trial num'] == trn_iter]

                ep_kin_data = train_ep[kin_cols]
                kin_in_bins = np.array_split(ep_kin_data, kin_bins)

                ep_emg_data = train_ep[emg_cols]
                emg_in_bins = np.array_split(ep_emg_data, emg_bins)

                # ep_tact_data = train_ep[tact_cols]
                # tact_in_bins = np.array_split(ep_tact_data, tact_bins)

                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    try:

                        kin_bin_mean = [np.nanmean(x, axis=0) for x in kin_in_bins]  # size = [num_bins] X [64]
                        flat_kin_mean = list(
                            itertools.chain.from_iterable(kin_bin_mean))  # size = [num_bins X 64] (unidimensional)

                        emg_bin_mean = [np.nanmean(x, axis=0) for x in emg_in_bins]  # size = [num_bins] X [64]
                        flat_emg_mean = list(
                            itertools.chain.from_iterable(emg_bin_mean))  # size = [num_bins X 64] (unidimensional)

                        # tact_bin_mean = [np.nanmean(x, axis=0) for x in tact_in_bins]  # size = [num_bins] X [64]
                        # flat_tact_mean = list(
                        #     itertools.chain.from_iterable(tact_bin_mean))  # size = [num_bins X 64] (unidimensional)

                        kin_train_data.append(flat_kin_mean)
                        emg_train_data.append(flat_emg_mean)
                        # tact_train_data.append(flat_tact_mean)
                        train_labels.append(np.unique(train_ep['Given Object'])[0])

                    except RuntimeWarning:
                        # print("Dropped EP", trn_iter, "from family ", family)
                        trn_dropped += 1

            kin_test_data = []
            emg_test_data = []
            tact_test_data = []

            test_labels = []

            for tst_iter in test_eps:

                test_ep = selected_df.loc[selected_df['Trial num'] == tst_iter]

                ep_kin_data = test_ep[kin_cols]
                kin_in_bins = np.array_split(ep_kin_data, kin_bins)

                ep_emg_data = test_ep[emg_cols]
                emg_in_bins = np.array_split(ep_emg_data, emg_bins)

                # ep_tact_data = test_ep[tact_cols]
                # tact_in_bins = np.array_split(ep_tact_data, tact_bins)

                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    try:

                        kin_bin_mean = [np.nanmean(x, axis=0) for x in kin_in_bins]  # size = [num_bins] X [64]
                        flat_kin_mean = list(
                            itertools.chain.from_iterable(kin_bin_mean))  # size = [num_bins X 64] (unidimensional)

                        emg_bin_mean = [np.nanmean(x, axis=0) for x in emg_in_bins]  # size = [num_bins] X [64]
                        flat_emg_mean = list(
                            itertools.chain.from_iterable(emg_bin_mean))  # size = [num_bins X 64] (unidimensional)

                        # tact_bin_mean = [np.nanmean(x, axis=0) for x in tact_in_bins]  # size = [num_bins] X [64]
                        # flat_tact_mean = list(
                        #     itertools.chain.from_iterable(tact_bin_mean))  # size = [num_bins X 64] (unidimensional)

                        kin_test_data.append(flat_kin_mean)
                        emg_test_data.append(flat_emg_mean)
                        # tact_test_data.append(flat_tact_mean)
                        test_labels.append(np.unique(test_ep['Given Object'])[0])

                    except RuntimeWarning:
                        # print("Dropped EP", tst_iter, "from family ", family)
                        tst_dropped += 1

            # compute weights (because unbalanced dataset)
            weights = compute_sample_weight(class_weight='balanced', y=train_labels)

            # build kinematic model
            kin_log_model = LogisticRegression(penalty='elasticnet', C=kin_c, class_weight='balanced',
                                               random_state=rnd_st,
                                               solver='saga', max_iter=25000, multi_class='ovr', n_jobs=-1,
                                               l1_ratio=kin_l1)

            # train kinematic model
            kin_log_model.fit(X=kin_train_data, y=train_labels, sample_weight=weights)
            # sc = round(kin_log_model.score(X=kin_test_data, y=test_labels) * 100, 2)
            # kin_total_score.append(sc)

            # build EMG model
            emg_log_model = LogisticRegression(penalty='elasticnet', C=emg_c, class_weight='balanced',
                                               random_state=rnd_st,
                                               solver='saga', max_iter=25000, multi_class='ovr', n_jobs=-1,
                                               l1_ratio=emg_l1)

            # train EMG model
            emg_log_model.fit(X=emg_train_data, y=train_labels, sample_weight=weights)
            # sc = round(emg_log_model.score(X=emg_test_data, y=test_labels) * 100, 2)
            # emg_total_score.append(sc)

            # # build Tactile model
            # tact_log_model = LogisticRegression(penalty='elasticnet', C=tact_c, class_weight='balanced',
            #                                     random_state=rnd_st,
            #                                     solver='saga', max_iter=25000, multi_class='ovr', n_jobs=-1,
            #                                     l1_ratio=tact_l1)
            #
            # # train EMG model
            # tact_log_model.fit(X=tact_train_data, y=train_labels, sample_weight=weights)

            # get prediction probabilities from first layer to train second layer
            kin_model_pred_proba = kin_log_model.predict_proba(X=kin_train_data)
            emg_model_pred_proba = emg_log_model.predict_proba(X=emg_train_data)
            # tact_model_pred_proba = tact_log_model.predict_proba(X=tact_train_data)

            # pred_proba = np.concatenate([kin_model_pred_proba, emg_model_pred_proba, tact_model_pred_proba], axis=1)
            pred_proba = np.concatenate([kin_model_pred_proba, emg_model_pred_proba], axis=1)

            # build & train top layer classifier
            top_log_model = LogisticRegression(C=top_C, class_weight='balanced', random_state=rnd_st, solver='saga',
                                               max_iter=25000,
                                               multi_class='ovr', n_jobs=-1)
            top_log_model.fit(X=pred_proba, y=train_labels, sample_weight=weights)

            # get probabilities from first layer on test set to feed the second layer
            kin_test_pred = kin_log_model.predict_proba(X=kin_test_data)
            emg_test_pred = emg_log_model.predict_proba(X=emg_test_data)
            # tact_test_pred = tact_log_model.predict_proba(X=tact_test_data)
            # test_proba = np.concatenate([kin_test_pred, emg_test_pred, tact_test_pred], axis=1)
            test_proba = np.concatenate([kin_test_pred, emg_test_pred], axis=1)

            # get prediction accuracy from second layer
            sc = round(top_log_model.score(X=test_proba, y=test_labels) * 100, 2)
            total_score.append(sc)

            # RANDOM CLASSIFIER
            # random.shuffle(train_labels)
            # kin_log_model.fit(X=kin_train_data, y=train_labels, sample_weight=weights)
            # emg_log_model.fit(X=emg_train_data, y=train_labels, sample_weight=weights)
            # # tact_log_model.fit(X=tact_train_data, y=train_labels, sample_weight=weights)
            # kin_model_pred_rand_proba = kin_log_model.predict_proba(X=kin_train_data)
            # emg_model_pred_rand_proba = emg_log_model.predict_proba(X=emg_train_data)
            # # tact_model_pred_rand_proba = tact_log_model.predict_proba(X=tact_train_data)
            # # pred_rand_proba = np.concatenate([kin_model_pred_rand_proba, emg_model_pred_rand_proba, tact_model_pred_rand_proba], axis=1)
            # pred_rand_proba = np.concatenate([kin_model_pred_rand_proba, emg_model_pred_rand_proba], axis=1)
            # top_log_model.fit(X=pred_rand_proba, y=train_labels, sample_weight=weights)
            # kin_test_pred = kin_log_model.predict_proba(X=kin_test_data)
            # emg_test_pred = emg_log_model.predict_proba(X=emg_test_data)
            # # tact_test_pred = tact_log_model.predict_proba(X=tact_test_data)
            # # test_proba = np.concatenate([kin_test_pred, emg_test_pred, tact_test_pred], axis=1)
            # test_proba = np.concatenate([kin_test_pred, emg_test_pred], axis=1)
            # rand_sc = round(top_log_model.score(X=test_proba, y=test_labels) * 100, 2)
            # random_score.append(rand_sc)


    result = ['Hierarchical_KE']
    result.extend([family, '0', '0', top_C])
    result.append(total_score)
    result.append(round(np.mean(total_score), 2))
    # random_result = ['Hierarchical_Rand']
    # random_result.extend([family, '0', '0', '0'])
    # random_result.append(random_score)
    # random_result.append(round(np.mean(random_score), 2))
    #
    # return [result, random_result]
    return result


def emg_classification(data):

    families = np.unique(data['Family'])
    cv = 3

    bins = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    l1VSl2 = [0, 0.25, 0.5, 0.75, 1]
    c_param = [0.01, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5]
    # we need to build the object to be iterated in the multiprocessing pool
    all_param = list(itertools.product(families, bins, l1VSl2, c_param))
    data_and_iter = [[data, x, cv] for x in all_param]

    # for testing
    # bins = 40
    # l1VSl2 = 0
    # c_param = 1.25
    # data_and_iter = [[data, [x, bins, l1VSl2, c_param], cv] for x in families]

    result_file = open('./results/Raw/accuracy/raw_results.csv', 'a')  # Open file in append mode
    wr = csv.writer(result_file)

    # multiprocessing
    with Pool() as pool:

        result = pool.map_async(emg_aux_classif, data_and_iter)

        for res in result.get():
            wr.writerow(res[0])
            wr.writerow(res[1])
            # a=1

    result_file.close()


def kinematic_classification(data):

    families = np.unique(data['Family'])
    cv = 3

    bins = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    l1VSl2 = [0, 0.25, 0.5, 0.75, 1]
    c_param = [0.01, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5]
    # we need to build the object to be iterated in the multiprocessing pool
    all_param = list(itertools.product(families, bins, l1VSl2, c_param))
    data_and_iter = [[data, x, cv] for x in all_param]

    # for testing
    # bins = 20
    # l1VSl2 = 1
    # c_param = 0.1
    # data_and_iter = [[data, [x, bins, l1VSl2, c_param], cv] for x in families]

    result_file = open('./results/Raw/accuracy/raw_results.csv', 'a')  # Open file in append mode
    wr = csv.writer(result_file)

    # multiprocessing
    with Pool() as pool:

        result = pool.map_async(kin_aux_classif, data_and_iter)

        for res in result.get():
            wr.writerow(res[0])
            wr.writerow(res[1])
            # a=1

    result_file.close()


def tactile_classification(data):

    families = np.unique(data['Family'])
    cv = 3

    bins = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    l1VSl2 = [0, 0.25, 0.5, 0.75, 1]
    c_param = [0.01, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5]
    # we need to build the object to be iterated in the multiprocessing pool
    all_param = list(itertools.product(families, bins, l1VSl2, c_param))
    data_and_iter = [[data, x, cv] for x in all_param]

    # for testing
    # bins = 20
    # l1VSl2 = 0
    # c_param = 0.01
    # data_and_iter = [[data, [x, bins, l1VSl2, c_param], cv] for x in families]

    result_file = open('./results/Raw/accuracy/raw_results.csv', 'a')  # Open file in append mode
    wr = csv.writer(result_file)

    # multiprocessing
    with Pool() as pool:

        result = pool.map_async(tact_aux_classif, data_and_iter)

        for res in result.get():
            wr.writerow(res[0])
            wr.writerow(res[1])
            # a=1

    result_file.close()


def multiple_source_classification(data):

    families = np.unique(data['Family'])
    cv = 3

    bins = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    l1VSl2 = [0, 0.25, 0.5, 0.75, 1]
    c_param = [0.01, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5]
    # we need to build the object to be iterated in the multiprocessing pool
    all_param = list(itertools.product(families, bins, l1VSl2, c_param))
    data_and_iter = [[data, x, cv] for x in all_param]

    result_file = open('./results/Raw/accuracy/raw_results.csv', 'a')  # Open file in append mode
    wr = csv.writer(result_file)

    # multiprocessing
    with Pool() as pool:

        result = pool.map_async(multiple_source_aux_classif, data_and_iter)

        for res in result.get():
            # wr.writerow(res[0])
            # wr.writerow(res[1])
            wr.writerow(res)

    result_file.close()


def hierarchical_classification(data):

    families = np.unique(data['Family'])
    c_param = [0.01, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5]
    cv = 3

    # we need to build the object to be iterated in the multiprocessing pool
    all_param = list(itertools.product(families, c_param))
    data_and_iter = [[data, x, cv] for x in all_param]

    result_file = open('./results/Raw/accuracy/raw_results.csv', 'a')  # Open file in append mode
    wr = csv.writer(result_file)

    # multiprocessing
    with Pool() as pool:
        result = pool.map_async(hierarchical_aux_classif, data_and_iter)

        for res in result.get():
            # wr.writerow(res[0])
            # wr.writerow(res[1])
            wr.writerow(res)

    result_file.close()


def ask_ep_presabs_classification(data):
    ## THIS FUNCTION IS TO CLASSIFY OBJECTS BASED ON EPs

    result_file = open('./results/EP/accuracy/ep_presabs_ask_results_file.csv', 'a')  # Open file in append mode

    families = np.unique(data['Family'])

    random_states = [42, 43, 44]

    for family in families:

        selected_df = data.loc[data['Family'] == family]  # select particular family

        weight_filename = './results/EP/weights/w_ask_EP_PresAbs_' + family + '.csv'
        weight_file = open(weight_filename, 'a')  # Open file in append mode
        weight_wr = csv.writer(weight_file)

        acc = []
        rand_acc = []

        for rnd_st in random_states:

            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=rnd_st)
            # WARNING: the skf.split returns the indexes
            for train, test in skf.split(selected_df.drop(['Given', 'Asked', 'Family'], axis=1),
                                         selected_df['Asked'].astype(str)):
                model = LogisticRegression(random_state=rnd_st).fit(
                    selected_df.iloc[train].drop(['Given', 'Asked', 'Family'], axis=1), selected_df.iloc[train]['Asked'])

                predicted = model.predict(selected_df.iloc[test].drop(['Given', 'Asked', 'Family'], axis=1))
                labels = selected_df.iloc[test]['Asked']

                hits = [int(list(labels.values)[x] == list(predicted)[x]) for x in range(0, len(predicted))]
                acc.append(round(np.sum(hits) * 100 / len(predicted), 2))
                # model weight extraction and saving
                [weight_wr.writerow(x) for x in model.coef_]

                # random shuffle model selected_df.iloc[train]['Asked'].sample(frac = 1)
                model.fit(
                    selected_df.iloc[train].drop(['Given', 'Asked', 'Family'], axis=1),
                    selected_df.iloc[train]['Asked'].sample(frac = 1))
                rnd_sc = round(model.score(selected_df.iloc[test].drop(['Given', 'Asked', 'Family'], axis=1),
                                           selected_df.iloc[test]['Asked'].sample(frac = 1)) * 100, 2)
                rand_acc.append(rnd_sc)

        print("(ASKED) Pre/Abs Mean accuracy for family", family, "is", round(np.mean(acc), 2), "%. All acc: ",
              [round(x, 2) for x in acc])
        wr = csv.writer(result_file)
        wr.writerow([acc, family])
        wr.writerow([rand_acc, 'Random'])

        # model weight file close
        weight_file.close()
    result_file.close()


def ask_ep_dur_classification(data):

    ## THIS FUNCTION IS TO CLASSIFY OBJECTS BASED ON EPs
    result_file = open('./results/EP/accuracy/ep_dur_ask_results_file.csv', 'a')  # Open file in append mode

    families = np.unique(data['Family'])

    random_states = [42, 43, 44]

    for family in families:

        selected_df = data.loc[data['Family'] == family]  # select particular family

        weight_filename = './results/EP/weights/w_ask_EP_Dur_' + family + '.csv'
        weight_file = open(weight_filename, 'a')  # Open file in append mode
        weight_wr = csv.writer(weight_file)

        acc = []
        rand_acc = []

        for rnd_st in random_states:

            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=rnd_st)
            # WARNING: the skf.split returns the indexes
            for train, test in skf.split(selected_df.drop(['Given', 'Asked', 'Family'], axis=1), selected_df['Asked'].astype(str)):

                model = LogisticRegression(random_state=rnd_st).fit(selected_df.iloc[train].drop(['Given', 'Asked', 'Family'], axis=1), selected_df.iloc[train]['Asked'])

                predicted = model.predict(selected_df.iloc[test].drop(['Given', 'Asked', 'Family'], axis=1))
                labels = selected_df.iloc[test]['Asked']

                hits = [int(list(labels.values)[x] == list(predicted)[x]) for x in range(0, len(predicted))]
                acc.append(round(np.sum(hits)*100/len(predicted), 2))
                # model weight extraction and saving
                [weight_wr.writerow(x) for x in model.coef_]

                # random shuffle model selected_df.iloc[train]['Asked'].sample(frac = 1)
                model.fit(
                    selected_df.iloc[train].drop(['Given', 'Asked', 'Family'], axis=1),
                    selected_df.iloc[train]['Asked'].sample(frac=1))
                rnd_sc = round(model.score(selected_df.iloc[test].drop(['Given', 'Asked', 'Family'], axis=1),
                                           selected_df.iloc[test]['Asked'].sample(frac=1)) * 100, 2)
                rand_acc.append(rnd_sc)

        print("(ASKED) Dur Mean accuracy for family", family, "is", round(np.mean(acc), 2), "%. All acc: ", [round(x, 2) for x in acc])
        wr = csv.writer(result_file)
        wr.writerow([acc, family])
        wr.writerow([rand_acc, 'Random'])

        # model weight file close
        weight_file.close()
    result_file.close()


def ask_ep_count_classification(data):
    ## THIS FUNCTION IS TO CLASSIFY OBJECTS BASED ON EPs

    result_file = open('./results/EP/accuracy/ep_count_ask_results_file.csv', 'a')  # Open file in append mode

    families = np.unique(data['Family'])

    random_states = [42, 43, 44]

    for family in families:

        selected_df = data.loc[data['Family'] == family]  # select particular family

        weight_filename = './results/EP/weights/w_ask_EP_Count_' + family + '.csv'
        weight_file = open(weight_filename, 'a')  # Open file in append mode
        weight_wr = csv.writer(weight_file)

        acc = []
        rand_acc = []

        for rnd_st in random_states:

            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=rnd_st)
            # WARNING: the skf.split returns the indexes
            for train, test in skf.split(selected_df.drop(['Given', 'Asked', 'Family'], axis=1),
                                         selected_df['Asked'].astype(str)):
                model = LogisticRegression(random_state=rnd_st).fit(
                    selected_df.iloc[train].drop(['Given', 'Asked', 'Family'], axis=1), selected_df.iloc[train]['Asked'])

                predicted = model.predict(selected_df.iloc[test].drop(['Given', 'Asked', 'Family'], axis=1))
                labels = selected_df.iloc[test]['Asked']

                hits = [int(list(labels.values)[x] == list(predicted)[x]) for x in range(0, len(predicted))]
                acc.append(round(np.sum(hits) * 100 / len(predicted), 2))
                # model weight extraction and saving
                [weight_wr.writerow(x) for x in model.coef_]

                # random shuffle model selected_df.iloc[train]['Asked'].sample(frac = 1)
                model.fit(
                    selected_df.iloc[train].drop(['Given', 'Asked', 'Family'], axis=1),
                    selected_df.iloc[train]['Asked'].sample(frac=1))
                rnd_sc = round(model.score(selected_df.iloc[test].drop(['Given', 'Asked', 'Family'], axis=1),
                                           selected_df.iloc[test]['Asked'].sample(frac=1)) * 100, 2)
                rand_acc.append(rnd_sc)

        print("(ASKED) Count Mean accuracy for family", family, "is", round(np.mean(acc), 2), "%. All acc: ",
              [round(x, 2) for x in acc])
        wr = csv.writer(result_file)
        wr.writerow([acc, family])
        wr.writerow([rand_acc, 'Random'])

        # model weight file close
        weight_file.close()
    result_file.close()


def giv_ep_presabs_classification(data):
    ## THIS FUNCTION IS TO CLASSIFY OBJECTS BASED ON EPs

    result_file = open('./results/EP/accuracy/ep_presabs_giv_results_file.csv', 'a')  # Open file in append mode

    families = np.unique(data['Family'])

    random_states = [42, 43, 44]

    for family in families:

        selected_df = data.loc[data['Family'] == family]  # select particular family

        weight_filename = './results/EP/weights/w_giv_EP_PresAbs_' + family + '.csv'
        weight_file = open(weight_filename, 'a')  # Open file in append mode
        weight_wr = csv.writer(weight_file)

        acc = []
        rand_acc = []

        for rnd_st in random_states:

            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=rnd_st)
            # WARNING: the skf.split returns the indexes
            for train, test in skf.split(selected_df.drop(['Given', 'Asked', 'Family'], axis=1),
                                         selected_df['Given'].astype(str)):
                model = LogisticRegression(random_state=rnd_st).fit(
                    selected_df.iloc[train].drop(['Given', 'Asked', 'Family'], axis=1), selected_df.iloc[train]['Given'])

                predicted = model.predict(selected_df.iloc[test].drop(['Given', 'Asked', 'Family'], axis=1))
                labels = selected_df.iloc[test]['Given']

                hits = [int(list(labels.values)[x] == list(predicted)[x]) for x in range(0, len(predicted))]
                acc.append(round(np.sum(hits) * 100 / len(predicted), 2))
                # model weight extraction and saving
                [weight_wr.writerow(x) for x in model.coef_]

                # random shuffle model selected_df.iloc[train]['Asked'].sample(frac = 1)
                model.fit(
                    selected_df.iloc[train].drop(['Given', 'Asked', 'Family'], axis=1),
                    selected_df.iloc[train]['Given'].sample(frac=1))
                rnd_sc = round(model.score(selected_df.iloc[test].drop(['Given', 'Asked', 'Family'], axis=1),
                                           selected_df.iloc[test]['Given'].sample(frac=1)) * 100, 2)
                rand_acc.append(rnd_sc)

        print("(GIVEN) Pre/Abs Mean accuracy for family", family, "is", round(np.mean(acc), 2), "%. All acc: ",
              [round(x, 2) for x in acc])
        wr = csv.writer(result_file)
        wr.writerow([acc, family])
        wr.writerow([rand_acc, 'Random'])

        # model weight file close
        weight_file.close()
    result_file.close()


def giv_ep_dur_classification(data):
    ## THIS FUNCTION IS TO CLASSIFY OBJECTS BASED ON EPs
    result_file = open('./results/EP/accuracy/ep_dur_giv_results_file.csv', 'a')  # Open file in append mode

    families = np.unique(data['Family'])

    random_states = [42, 43, 44]

    for family in families:

        selected_df = data.loc[data['Family'] == family]  # select particular family

        weight_filename = './results/EP/weights/w_giv_EP_Dur_' + family + '.csv'
        weight_file = open(weight_filename, 'a')  # Open file in append mode
        weight_wr = csv.writer(weight_file)

        acc = []
        rand_acc = []

        for rnd_st in random_states:

            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=rnd_st)
            # WARNING: the skf.split returns the indexes
            for train, test in skf.split(selected_df.drop(['Given', 'Asked', 'Family'], axis=1), selected_df['Given'].astype(str)):

                model = LogisticRegression(random_state=rnd_st).fit(selected_df.iloc[train].drop(['Given', 'Asked', 'Family'], axis=1), selected_df.iloc[train]['Given'])

                predicted = model.predict(selected_df.iloc[test].drop(['Given', 'Asked', 'Family'], axis=1))
                labels = selected_df.iloc[test]['Given']

                hits = [int(list(labels.values)[x] == list(predicted)[x]) for x in range(0, len(predicted))]
                acc.append(round(np.sum(hits)*100/len(predicted), 2))
                # model weight extraction and saving
                [weight_wr.writerow(x) for x in model.coef_]

                # random shuffle model selected_df.iloc[train]['Asked'].sample(frac = 1)
                model.fit(
                    selected_df.iloc[train].drop(['Given', 'Asked', 'Family'], axis=1),
                    selected_df.iloc[train]['Given'].sample(frac=1))
                rnd_sc = round(model.score(selected_df.iloc[test].drop(['Given', 'Asked', 'Family'], axis=1),
                                           selected_df.iloc[test]['Given'].sample(frac=1)) * 100, 2)
                rand_acc.append(rnd_sc)

        print("(GIVEN) Dur Mean accuracy for family", family, "is", round(np.mean(acc), 2), "%. All acc: ", [round(x, 2) for x in acc])
        wr = csv.writer(result_file)
        wr.writerow([acc, family])
        wr.writerow([rand_acc, 'Random'])

        # model weight file close
        weight_file.close()
    result_file.close()


def giv_ep_count_classification(data):
    ## THIS FUNCTION IS TO CLASSIFY OBJECTS BASED ON EPs

    result_file = open('./results/EP/accuracy/ep_count_giv_results_file.csv', 'a')  # Open file in append mode

    families = np.unique(data['Family'])

    random_states = [42, 43, 44]

    for family in families:

        selected_df = data.loc[data['Family'] == family]  # select particular family

        weight_filename = './results/EP/weights/w_giv_EP_Count_' + family + '.csv'
        weight_file = open(weight_filename, 'a')  # Open file in append mode
        weight_wr = csv.writer(weight_file)

        acc = []
        rand_acc = []

        for rnd_st in random_states:

            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=rnd_st)
            # WARNING: the skf.split returns the indexes
            for train, test in skf.split(selected_df.drop(['Given', 'Asked', 'Family'], axis=1),
                                         selected_df['Given'].astype(str)):
                model = LogisticRegression(random_state=rnd_st).fit(
                    selected_df.iloc[train].drop(['Given', 'Asked', 'Family'], axis=1), selected_df.iloc[train]['Given'])

                predicted = model.predict(selected_df.iloc[test].drop(['Given', 'Asked', 'Family'], axis=1))
                labels = selected_df.iloc[test]['Given']

                hits = [int(list(labels.values)[x] == list(predicted)[x]) for x in range(0, len(predicted))]
                acc.append(round(np.sum(hits) * 100 / len(predicted), 2))
                # model weight extraction and saving
                [weight_wr.writerow(x) for x in model.coef_]

                # random shuffle model selected_df.iloc[train]['Asked'].sample(frac = 1)
                model.fit(
                    selected_df.iloc[train].drop(['Given', 'Asked', 'Family'], axis=1),
                    selected_df.iloc[train]['Given'].sample(frac=1))
                rnd_sc = round(model.score(selected_df.iloc[test].drop(['Given', 'Asked', 'Family'], axis=1),
                                           selected_df.iloc[test]['Given'].sample(frac=1)) * 100, 2)
                rand_acc.append(rnd_sc)

        print("(GIVEN) Count Mean accuracy for family", family, "is", round(np.mean(acc), 2), "%. All acc: ",
              [round(x, 2) for x in acc])
        wr = csv.writer(result_file)
        wr.writerow([acc, family])
        wr.writerow([rand_acc, 'Random'])

        # model weight file close
        weight_file.close()
    result_file.close()


def fam_ep_presabs_classification(data):
    ## THIS FUNCTION IS TO CLASSIFY OBJECT FAMILY BASED ON EPs

    result_file = open('./results/EP/accuracy/ep_presabs_fam_results_file.csv', 'a')  # Open file in append mode

    random_states = [42, 43, 44]

    weight_filename = './results/EP/weights/w_fam_EP_PresAbs.csv'
    weight_file = open(weight_filename, 'a')  # Open file in append mode
    weight_wr = csv.writer(weight_file)

    acc = []
    rand_acc = []

    for rnd_st in random_states:

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=rnd_st)
        # WARNING: the skf.split returns the indexes
        for train, test in skf.split(data.drop(['Given', 'Asked', 'Family'], axis=1),
                                     data['Family'].astype(str)):
            model = LogisticRegression(random_state=rnd_st).fit(
                data.iloc[train].drop(['Given', 'Asked', 'Family'], axis=1), data.iloc[train]['Family'])

            predicted = model.predict(data.iloc[test].drop(['Given', 'Asked', 'Family'], axis=1))
            labels = data.iloc[test]['Family']

            hits = [int(list(labels.values)[x] == list(predicted)[x]) for x in range(0, len(predicted))]
            acc.append(round(np.sum(hits) * 100 / len(predicted), 2))
            # model weight extraction and saving
            [weight_wr.writerow(x) for x in model.coef_]

            # random shuffle model selected_df.iloc[train]['Asked'].sample(frac = 1)
            model.fit(
                data.iloc[train].drop(['Given', 'Asked', 'Family'], axis=1),
                data.iloc[train]['Family'].sample(frac=1))
            rnd_sc = round(model.score(data.iloc[test].drop(['Given', 'Asked', 'Family'], axis=1),
                                       data.iloc[test]['Family'].sample(frac=1)) * 100, 2)
            rand_acc.append(rnd_sc)

    print("Pre/Abs Mean accuracy for family classification is", round(np.mean(acc), 2), "%. All acc: ",
          [round(x, 2) for x in acc])
    wr = csv.writer(result_file)
    wr.writerow([acc, 'Family'])
    wr.writerow([rand_acc, 'Random'])

    # model weight file close
    weight_file.close()
    result_file.close()


def fam_ep_dur_classification(data):
    ## THIS FUNCTION IS TO CLASSIFY OBJECT FAMILY BASED ON EPs

    result_file = open('./results/EP/accuracy/ep_dur_fam_results_file.csv', 'a')  # Open file in append mode

    random_states = [42, 43, 44]

    weight_filename = './results/EP/weights/w_fam_EP_Dur.csv'
    weight_file = open(weight_filename, 'a')  # Open file in append mode
    weight_wr = csv.writer(weight_file)

    acc = []
    rand_acc = []

    for rnd_st in random_states:

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=rnd_st)
        # WARNING: the skf.split returns the indexes
        for train, test in skf.split(data.drop(['Given', 'Asked', 'Family'], axis=1),
                                     data['Family'].astype(str)):
            model = LogisticRegression(random_state=rnd_st).fit(
                data.iloc[train].drop(['Given', 'Asked', 'Family'], axis=1), data.iloc[train]['Family'])

            predicted = model.predict(data.iloc[test].drop(['Given', 'Asked', 'Family'], axis=1))
            labels = data.iloc[test]['Family']

            hits = [int(list(labels.values)[x] == list(predicted)[x]) for x in range(0, len(predicted))]
            acc.append(round(np.sum(hits) * 100 / len(predicted), 2))
            # model weight extraction and saving
            [weight_wr.writerow(x) for x in model.coef_]

            # random shuffle model selected_df.iloc[train]['Asked'].sample(frac = 1)
            model.fit(
                data.iloc[train].drop(['Given', 'Asked', 'Family'], axis=1),
                data.iloc[train]['Family'].sample(frac=1))
            rnd_sc = round(model.score(data.iloc[test].drop(['Given', 'Asked', 'Family'], axis=1),
                                       data.iloc[test]['Family'].sample(frac=1)) * 100, 2)
            rand_acc.append(rnd_sc)

    print("Dur Mean accuracy for family classification is", round(np.mean(acc), 2), "%. All acc: ",
          [round(x, 2) for x in acc])
    wr = csv.writer(result_file)
    wr.writerow([acc, 'Family'])
    wr.writerow([rand_acc, 'Random'])

    # model weight file close
    weight_file.close()
    result_file.close()


def fam_ep_count_classification(data):
    ## THIS FUNCTION IS TO CLASSIFY OBJECT FAMILY BASED ON EPs

    result_file = open('./results/EP/accuracy/ep_count_fam_results_file.csv', 'a')  # Open file in append mode

    random_states = [42, 43, 44]

    weight_filename = './results/EP/weights/w_fam_EP_Count.csv'
    weight_file = open(weight_filename, 'a')  # Open file in append mode
    weight_wr = csv.writer(weight_file)

    acc = []
    rand_acc = []

    for rnd_st in random_states:

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=rnd_st)
        # WARNING: the skf.split returns the indexes
        for train, test in skf.split(data.drop(['Given', 'Asked', 'Family'], axis=1),
                                     data['Family'].astype(str)):
            model = LogisticRegression(random_state=rnd_st).fit(
                data.iloc[train].drop(['Given', 'Asked', 'Family'], axis=1), data.iloc[train]['Family'])

            predicted = model.predict(data.iloc[test].drop(['Given', 'Asked', 'Family'], axis=1))
            labels = data.iloc[test]['Family']

            hits = [int(list(labels.values)[x] == list(predicted)[x]) for x in range(0, len(predicted))]
            acc.append(round(np.sum(hits) * 100 / len(predicted), 2))
            # model weight extraction and saving
            [weight_wr.writerow(x) for x in model.coef_]

            # random shuffle model selected_df.iloc[train]['Asked'].sample(frac = 1)
            model.fit(
                data.iloc[train].drop(['Given', 'Asked', 'Family'], axis=1),
                data.iloc[train]['Family'].sample(frac=1))
            rnd_sc = round(model.score(data.iloc[test].drop(['Given', 'Asked', 'Family'], axis=1),
                                       data.iloc[test]['Family'].sample(frac=1)) * 100, 2)
            rand_acc.append(rnd_sc)

    print("Count Mean accuracy for family classification is", round(np.mean(acc), 2), "%. All acc: ",
          [round(x, 2) for x in acc])
    wr = csv.writer(result_file)
    wr.writerow([acc, 'Family'])
    wr.writerow([rand_acc, 'Random'])

    # model weight file close
    weight_file.close()
    result_file.close()