import numpy as np
import itertools
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from collections import Counter
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
import csv
from scipy.stats import zscore
import pandas as pd
from multiprocessing.pool import Pool
import random
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def kin_syn_extraction(data):

    kin_scaled = StandardScaler().fit_transform(data)  # Z score
    kin_df = pd.DataFrame(kin_scaled)
    pca_mod = PCA()

    kin_scores = pca_mod.fit_transform(kin_df)
    pd.DataFrame(kin_scores).to_csv('./results/Syn/scores/kin_scores.csv')

    kin_syns = pca_mod.components_  # Each column is a synergy
    pd.DataFrame(kin_syns).to_csv('./results/Syn/synergies/kin_syns.csv')

    kin_var = pca_mod.explained_variance_ratio_
    pd.DataFrame(kin_var).to_csv('./results/Syn/variance/kin_var.csv')



def emg_pca_syn_extraction(data):

    emg_scaled = StandardScaler().fit_transform(data)  # Z score
    emg_df = pd.DataFrame(emg_scaled)
    pca_mod = PCA()

    emg_scores = pca_mod.fit_transform(emg_df)
    pd.DataFrame(emg_scores).to_csv('./results/Syn/scores/emg_pca_scores.csv')

    emg_syns = pca_mod.components_  # Each column is a synergy
    pd.DataFrame(emg_syns).to_csv('./results/Syn/synergies/emg_pca_syns.csv')

    emg_var = pca_mod.explained_variance_ratio_
    pd.DataFrame(emg_var).to_csv('./results/Syn/variance/emg_pca_var.csv')



def emg_nmf_syn_extraction(data):

    perc_syns = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]

    for p in perc_syns:

        num_syn_emg = int(np.ceil(len(data.columns) * p / 100))

        file_name_sc = './results/Syn/scores/emg_nmf' + str(p)
        file_name_sy = './results/Syn/synergies/emg_nmf' + str(p)

        nmf_mod = NMF(n_components=num_syn_emg, max_iter=1500)
        emg_scores = nmf_mod.fit_transform(data)
        pd.DataFrame(emg_scores).to_csv(file_name_sc + '_scores.csv')
        emg_syns = nmf_mod.components_
        pd.DataFrame(emg_syns).to_csv(file_name_sy + '_syns.csv')

        print('Done for', p, '%')



def tact_syn_extraction(data):

    tact_scaled = StandardScaler().fit_transform(data)  # Z score
    tact_df = pd.DataFrame(tact_scaled)
    pca_mod = PCA()

    tact_scores = pca_mod.fit_transform(tact_df)
    pd.DataFrame(tact_scores).to_csv('./results/Syn/scores/tact_scores.csv')

    tact_syns = pca_mod.components_  # Each column is a synergy
    pd.DataFrame(tact_syns).to_csv('./results/Syn/synergies/tact_syns.csv')

    tact_var = pca_mod.explained_variance_ratio_
    pd.DataFrame(tact_var).to_csv('./results/Syn/variance/tact_var.csv')



def syn_extraction(data):

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

    families = np.unique(data['Family'])

    # REMOVE NANs
    data_clean = data.dropna(axis=0, how='any')

    # # NON-NUMERIC DATA EXTRACTION & SAVING
    extra_cols = [col for col in data_clean.columns if (col not in kin_cols) and (col not in emg_cols) and (col not in tact_cols)]
    extra_df = data_clean[extra_cols]
    extra_df.to_csv('./results/Syn/extra_data.csv', index=False)

    ## SYNERGY EXTRACTION AND SAVING
    kin_syn_extraction(data_clean[kin_cols])
    emg_pca_syn_extraction(data_clean[emg_cols])
    emg_nmf_syn_extraction(data_clean[emg_cols])
    tact_syn_extraction(data_clean[tact_cols])



def hierarchical_syn_classification():

    """This code is a piece of the original hierarchical classifier. It does not iterate over C or L1vL2 for the second
    layer classifier and C and L1vL2 parameters are fixed for the single-layer classifier. The idea now is to get the
    best combination for each iteration over the number of synergies"""

    # families = ['Ball', 'Cutlery', 'Geometric', 'Mugs', 'Plates']
    # cv = 3
    #
    # kin_bins = 40
    # kin_l1 = 0.25
    # kin_c = 0.1
    #
    # emg_bins = 10
    # emg_l1 = 0
    # emg_c = 1.5
    #
    # tact_bins = 5
    # tact_l1 = 0.5
    # tact_c = 0.25
    #
    # top_c = 0.5

    # result_file = open('./results/Syn/accuracy/syn_results.csv', 'a')  # Open file in append mode
    # wr = csv.writer(result_file)
    # kin_score_df = pd.read_csv('./results/Syn/scores/kin_scores.csv', index_col=0)
    # emg_score_df = pd.read_csv('./results/Syn/scores/emg_pca_scores.csv', index_col=0)
    # tact_score_df = pd.read_csv('./results/Syn/scores/tact_scores.csv', index_col=0)
    #
    # extra_data = pd.read_csv('./results/Syn/extra_data.csv')
    #
    # perc_syns = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
    #
    # for p in perc_syns:
    #
    #     num_syn_kin = np.ceil(len(kin_score_df.columns) * p / 100)
    #     num_syn_emg = np.ceil(len(emg_score_df.columns) * p / 100)
    #     num_syn_tact = np.ceil(len(tact_score_df.columns) * p / 100)
    #
    #     kin_scores = pd.concat([kin_score_df.iloc[:, :int(num_syn_kin)], extra_data], axis=1, ignore_index=True)
    #     kin_scores.columns = list(kin_score_df.columns[:int(num_syn_kin)]) + list(extra_data.columns)
    #
    #     emg_scores = pd.concat([emg_score_df.iloc[:, :int(num_syn_emg)], extra_data], axis=1, ignore_index=True)
    #     emg_scores.columns = list(emg_score_df.columns[:int(num_syn_emg)]) + list(extra_data.columns)
    #
    #     tact_scores = pd.concat([tact_score_df.iloc[:, :int(num_syn_tact)], extra_data], axis=1, ignore_index=True)
    #     tact_scores.columns = list(tact_score_df.columns[:int(num_syn_tact)]) + list(extra_data.columns)
    #
    #     # print("Kin shape:", kin_scores.shape)
    #     # print("EMG shape:", emg_scores.shape)
    #     # print("Tact shape:", tact_scores.shape)
    #
    #     for family in families:
    #
    #         total_score = []
    #
    #         kin_dat = kin_scores.loc[kin_scores['Family'] == family]
    #         emg_dat = emg_scores.loc[emg_scores['Family'] == family]
    #         tact_dat = tact_scores.loc[tact_scores['Family'] == family]
    #
    #         to_kfold = kin_dat.drop_duplicates(
    #             subset=['Trial num', 'Given Object'])  # only way I found to avoid overlapping
    #
    #         random_states = [42, 43, 44]
    #
    #         for rnd_st in random_states:
    #
    #             skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=rnd_st)
    #             # WARNING: the skf.split returns the indexes
    #             for train, test in skf.split(to_kfold['Trial num'].astype(int), to_kfold['Given Object'].astype(str)):
    #
    #                 train_eps = to_kfold.iloc[train]['Trial num']  # because skf.split returns the indexes
    #                 test_eps = to_kfold.iloc[test]['Trial num']  # because skf.split returns the indexes
    #
    #                 kin_train_data = []
    #                 emg_train_data = []
    #                 tact_train_data = []
    #                 train_labels = []
    #
    #                 trn_dropped = 0  # Number of dropped EPs in training dataset
    #                 tst_dropped = 0  # Number of dropped EPs in test dataset
    #
    #                 for trn_iter in train_eps:
    #
    #                     ep_kin_data = kin_dat.loc[kin_dat['Trial num'] == trn_iter]
    #                     kin_in_bins = np.array_split(ep_kin_data.drop(columns=extra_data.columns, axis=1), kin_bins)
    #
    #                     ep_emg_data = emg_dat.loc[emg_dat['Trial num'] == trn_iter]
    #                     emg_in_bins = np.array_split(ep_emg_data.drop(columns=extra_data.columns, axis=1), emg_bins)
    #
    #                     ep_tact_data = tact_dat.loc[tact_dat['Trial num'] == trn_iter]
    #                     tact_in_bins = np.array_split(ep_tact_data.drop(columns=extra_data.columns, axis=1), tact_bins)
    #
    #                     with warnings.catch_warnings():
    #                         warnings.filterwarnings('error')
    #                         try:
    #
    #                             kin_bin_mean = [np.nanmean(x, axis=0) for x in kin_in_bins]  # size = [num_bins] X [64]
    #                             flat_kin_mean = list(
    #                                 itertools.chain.from_iterable(kin_bin_mean))  # size = [num_bins X 64] (unidimensional)
    #
    #                             emg_bin_mean = [np.nanmean(x, axis=0) for x in emg_in_bins]  # size = [num_bins] X [64]
    #                             flat_emg_mean = list(
    #                                 itertools.chain.from_iterable(
    #                                     emg_bin_mean))  # size = [num_bins X 64] (unidimensional)
    #
    #                             tact_bin_mean = [np.nanmean(x, axis=0) for x in
    #                                              tact_in_bins]  # size = [num_bins] X [64]
    #                             flat_tact_mean = list(
    #                                 itertools.chain.from_iterable(
    #                                     tact_bin_mean))  # size = [num_bins X 64] (unidimensional)
    #
    #                             kin_train_data.append(flat_kin_mean)
    #                             emg_train_data.append(flat_emg_mean)
    #                             tact_train_data.append(flat_tact_mean)
    #                             train_labels.append(np.unique(ep_kin_data['Given Object'])[0])
    #
    #                         except RuntimeWarning:
    #                             # print("Dropped EP", trn_iter, "from family ", family)
    #                             trn_dropped += 1
    #
    #                 kin_test_data = []
    #                 emg_test_data = []
    #                 tact_test_data = []
    #                 test_labels = []
    #
    #                 for tst_iter in test_eps:
    #
    #                     ep_kin_data = kin_dat.loc[kin_dat['Trial num'] == tst_iter]
    #                     kin_in_bins = np.array_split(ep_kin_data.drop(columns=extra_data.columns, axis=1),
    #                                                  kin_bins)
    #
    #                     ep_emg_data = emg_dat.loc[emg_dat['Trial num'] == tst_iter]
    #                     emg_in_bins = np.array_split(ep_emg_data.drop(columns=extra_data.columns, axis=1),
    #                                                  emg_bins)
    #
    #                     ep_tact_data = tact_dat.loc[tact_dat['Trial num'] == tst_iter]
    #                     tact_in_bins = np.array_split(ep_tact_data.drop(columns=extra_data.columns, axis=1),
    #                                                   tact_bins)
    #
    #                     with warnings.catch_warnings():
    #                         warnings.filterwarnings('error')
    #                         try:
    #
    #                             kin_bin_mean = [np.nanmean(x, axis=0) for x in
    #                                             kin_in_bins]  # size = [num_bins] X [64]
    #                             flat_kin_mean = list(
    #                                 itertools.chain.from_iterable(
    #                                     kin_bin_mean))  # size = [num_bins X 64] (unidimensional)
    #
    #                             emg_bin_mean = [np.nanmean(x, axis=0) for x in
    #                                             emg_in_bins]  # size = [num_bins] X [64]
    #                             flat_emg_mean = list(
    #                                 itertools.chain.from_iterable(
    #                                     emg_bin_mean))  # size = [num_bins X 64] (unidimensional)
    #
    #                             tact_bin_mean = [np.nanmean(x, axis=0) for x in
    #                                              tact_in_bins]  # size = [num_bins] X [64]
    #                             flat_tact_mean = list(
    #                                 itertools.chain.from_iterable(
    #                                     tact_bin_mean))  # size = [num_bins X 64] (unidimensional)
    #
    #                             kin_test_data.append(flat_kin_mean)
    #                             emg_test_data.append(flat_emg_mean)
    #                             tact_test_data.append(flat_tact_mean)
    #                             test_labels.append(np.unique(ep_kin_data['Given Object'])[0])
    #
    #                         except RuntimeWarning:
    #                             # print("Dropped EP", tst_iter, "from family ", family)
    #                             tst_dropped += 1
    #
    #                 # compute weights (because unbalanced dataset)
    #                 # weights = compute_sample_weight(class_weight='balanced', y=train_labels)
    #
    #                 # build kinematic model
    #                 kin_log_model = LogisticRegression(penalty='elasticnet', C=kin_c, random_state=rnd_st,
    #                                                    solver='saga', max_iter=25000, multi_class='ovr', n_jobs=-1,
    #                                                    l1_ratio=kin_l1)
    #
    #                 # train kinematic model
    #                 kin_log_model.fit(X=kin_train_data, y=train_labels)
    #
    #                 # build EMG model
    #                 emg_log_model = LogisticRegression(penalty='elasticnet', C=emg_c,
    #                                                    random_state=rnd_st,
    #                                                    solver='saga', max_iter=25000, multi_class='ovr',
    #                                                    n_jobs=-1,
    #                                                    l1_ratio=emg_l1)
    #
    #                 # train EMG model
    #                 emg_log_model.fit(X=emg_train_data, y=train_labels)
    #
    #
    #                 # build Tactile model
    #                 tact_log_model = LogisticRegression(penalty='elasticnet', C=tact_c,
    #                                                     random_state=rnd_st,
    #                                                     solver='saga', max_iter=25000, multi_class='ovr',
    #                                                     n_jobs=-1,
    #                                                     l1_ratio=tact_l1)
    #
    #                 # train EMG model
    #                 tact_log_model.fit(X=tact_train_data, y=train_labels)
    #
    #                 # get prediction probabilities from first layer to train second layer
    #                 kin_model_pred_proba = kin_log_model.predict_proba(X=kin_train_data)
    #                 emg_model_pred_proba = emg_log_model.predict_proba(X=emg_train_data)
    #                 tact_model_pred_proba = tact_log_model.predict_proba(X=tact_train_data)
    #
    #                 # pred_proba = np.concatenate([kin_model_pred_proba, emg_model_pred_proba, tact_model_pred_proba], axis=1)
    #                 pred_proba = np.concatenate([kin_model_pred_proba, emg_model_pred_proba, tact_model_pred_proba], axis=1)
    #
    #                 # build & train top layer classifier
    #                 top_log_model = LogisticRegression(C=top_c, random_state=rnd_st, solver='saga',
    #                                                    max_iter=25000,
    #                                                    multi_class='ovr', n_jobs=-1)
    #                 top_log_model.fit(X=pred_proba, y=train_labels)
    #
    #                 # get probabilities from first layer on test set to feed the second layer
    #                 kin_test_pred = kin_log_model.predict_proba(X=kin_test_data)
    #                 emg_test_pred = emg_log_model.predict_proba(X=emg_test_data)
    #                 tact_test_pred = tact_log_model.predict_proba(X=tact_test_data)
    #                 test_proba = np.concatenate([kin_test_pred, emg_test_pred, tact_test_pred], axis=1)
    #                 # test_proba = np.concatenate([emg_test_pred, tact_test_pred], axis=1)
    #
    #                 # get prediction accuracy from second layer
    #                 sc = round(top_log_model.score(X=test_proba, y=test_labels) * 100, 2)
    #                 # total_score.append(sc)
    #
    #                 res = ['Hierarchical']
    #                 res.extend([family, p])
    #                 res.append(sc)
    #                 # res.append(round(np.mean(total_score), 2))
    #                 wr.writerow(res)
    #                 print(res)
    #
    # result_file.close()
    # print("DONE !!!")



def kin_syn_classif(input_data):

    kin_scores = input_data[0]
    extra_data = input_data[1]
    perc_syns = input_data[2][0]
    family = input_data[2][1]
    l1VSl2 = input_data[2][2]
    c_param = input_data[2][3]
    cv = input_data[3]
    kin_bins = input_data[4]

    total_score = []

    num_syns = np.ceil(len(kin_scores.columns) * perc_syns / 100)
    data_df = pd.concat([kin_scores.iloc[:, 0:int(num_syns)], extra_data], axis=1)
    selected_df = data_df.loc[data_df['Family'] == family]

    to_kfold = selected_df.drop_duplicates(
        subset=['Trial num', 'Given Object'])  # only way I found to avoid overlapping

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
                tr_kin_data = train_tri.drop(columns=extra_data.columns)
                tr_in_bins = np.array_split(tr_kin_data, kin_bins)

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
                tst_kin_data = test_tri.drop(columns=extra_data.columns)
                tst_in_bins = np.array_split(tst_kin_data, kin_bins)

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
            log_model = LogisticRegression(penalty='elasticnet', C=c_param, class_weight='balanced', random_state=rnd_st,
                                           solver='saga', max_iter=25000, multi_class='ovr', n_jobs=-1,
                                           l1_ratio=l1VSl2)
            # train model
            log_model.fit(X=train_data, y=train_labels)
            sc = round(log_model.score(X=test_data, y=test_labels) * 100, 2)
            total_score.append(sc)

    result = ['Kin']
    result.extend(input_data[2])
    result.append(total_score)
    result.append(round(np.mean(total_score), 2))
    # print("RESULT:", result)

    return result



def emg_pca_syn_classif(input_data):

    emg_pca_scores = input_data[0]
    extra_data = input_data[1]
    perc_syns = input_data[2][0]
    family = input_data[2][1]
    l1VSl2 = input_data[2][2]
    c_param = input_data[2][3]
    cv = input_data[3]
    emg_pca_bins = input_data[4]

    total_score = []

    num_syns = np.ceil(len(emg_pca_scores.columns) * perc_syns / 100)
    data_df = pd.concat([emg_pca_scores.iloc[:, 0:int(num_syns)], extra_data], axis=1)
    selected_df = data_df.loc[data_df['Family'] == family]

    to_kfold = selected_df.drop_duplicates(
        subset=['Trial num', 'Given Object'])  # only way I found to avoid overlapping

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
                tr_emg_pca_data = train_tri.drop(columns=extra_data.columns)
                tr_in_bins = np.array_split(tr_emg_pca_data, emg_pca_bins)

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
                tst_emg_pca_data = test_tri.drop(columns=extra_data.columns)
                tst_in_bins = np.array_split(tst_emg_pca_data, emg_pca_bins)

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
            log_model = LogisticRegression(penalty='elasticnet', C=c_param, class_weight='balanced',
                                           random_state=rnd_st,
                                           solver='saga', max_iter=25000, multi_class='ovr', n_jobs=-1,
                                           l1_ratio=l1VSl2)
            # train model
            log_model.fit(X=train_data, y=train_labels)
            sc = round(log_model.score(X=test_data, y=test_labels) * 100, 2)
            total_score.append(sc)

    result = ['EMG PCA']
    result.extend(input_data[2])
    result.append(total_score)
    result.append(round(np.mean(total_score), 2))
    # print("RESULT:", result)

    return result



def emg_nmf_syn_classif(input_data):

    perc_syns = input_data[0][0]
    family = input_data[0][1]
    # l1VSl2 = input_data[0][2]
    c_param = input_data[0][2]
    cv = input_data[1]
    emg_nmf_bins = input_data[2]

    # need to open nmf file
    filepath = './results/Syn/scores/emg_nmf' + str(perc_syns) + '_scores.csv'
    emg_nmf_score_df = pd.read_csv(filepath, index_col=0)
    extra_data = pd.read_csv('./results/Syn/extra_data.csv')

    total_score = []

    data_df = pd.concat([emg_nmf_score_df, extra_data], axis=1)
    selected_df = data_df.loc[data_df['Family'] == family]

    to_kfold = selected_df.drop_duplicates(
        subset=['Trial num', 'Given Object'])  # only way I found to avoid overlapping

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
                tr_emg_nmf_data = train_tri.drop(columns=extra_data.columns)
                tr_in_bins = np.array_split(tr_emg_nmf_data, emg_nmf_bins)

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
                tst_emg_nmf_data = test_tri.drop(columns=extra_data.columns)
                tst_in_bins = np.array_split(tst_emg_nmf_data, emg_nmf_bins)

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
            # log_model = LogisticRegression(penalty='elasticnet', C=c_param, class_weight='balanced',
            #                                random_state=rnd_st,
            #                                solver='saga', max_iter=25000, tol=0.000000001, multi_class='ovr', n_jobs=-1,
            #                                l1_ratio=l1VSl2)
            """PENALTY = L2"""
            log_model = LogisticRegression(penalty='l2', C=c_param, class_weight='balanced',
                                           random_state=rnd_st,
                                           solver='newton-cg', max_iter=25000, tol=0.000000001, multi_class='ovr', n_jobs=-1)
            # train model
            log_model.fit(X=train_data, y=train_labels)
            sc = round(log_model.score(X=test_data, y=test_labels) * 100, 2)
            total_score.append(sc)
            print(log_model.n_iter_)
            print(sc)

    result = ['EMG NMF']
    result.extend(input_data[0])
    result.append(total_score)
    result.append(round(np.mean(total_score), 2))
    # print("RESULT:", result)

    return result



def tact_syn_classif(input_data):

    tact_scores = input_data[0]
    extra_data = input_data[1]
    perc_syns = input_data[2][0]
    family = input_data[2][1]
    l1VSl2 = input_data[2][2]
    c_param = input_data[2][3]
    cv = input_data[3]
    tact_bins = input_data[4]

    total_score = []

    num_syns = np.ceil(len(tact_scores.columns) * perc_syns / 100)
    data_df = pd.concat([tact_scores.iloc[:, 0:int(num_syns)], extra_data], axis=1)
    selected_df = data_df.loc[data_df['Family'] == family]

    to_kfold = selected_df.drop_duplicates(
        subset=['Trial num', 'Given Object'])  # only way I found to avoid overlapping

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
                tr_tact_data = train_tri.drop(columns=extra_data.columns)
                tr_in_bins = np.array_split(tr_tact_data, tact_bins)

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
                tst_tact_data = test_tri.drop(columns=extra_data.columns)
                tst_in_bins = np.array_split(tst_tact_data, tact_bins)

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
            log_model = LogisticRegression(penalty='elasticnet', C=c_param, class_weight='balanced',
                                           random_state=rnd_st,
                                           solver='saga', max_iter=25000, multi_class='ovr', n_jobs=-1,
                                           l1_ratio=l1VSl2)
            # train model
            log_model.fit(X=train_data, y=train_labels)
            sc = round(log_model.score(X=test_data, y=test_labels) * 100, 2)
            total_score.append(sc)

    result = ['Tact']
    result.extend(input_data[2])
    result.append(total_score)
    result.append(round(np.mean(total_score), 2))
    # print("RESULT:", result)

    return result



def syn_single_source_classification():

    cv = 3
    kin_bins = 40
    emg_bins = 10
    tact_bins = 5
    perc_syns = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
    families = ['Ball', 'Cutlery', 'Geometric', 'Mugs', 'Plates']
    l1VSl2 = [0, 0.25, 0.5, 0.75, 1]
    c_param = [0.01, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5]

    # TEST
    # perc_syns = [60, 20]
    # families = ['Ball', 'Plates']
    # l1VSl2 = [0.5, 1]
    # c_param = [0.1, 1.25]

    extra_data = pd.read_csv('./results/Syn/extra_data.csv')

    result_file = open('./results/Syn/accuracy/syn_results.csv', 'a')  # Open file in append mode
    wr = csv.writer(result_file)
    kin_score_df = pd.read_csv('./results/Syn/scores/kin_scores.csv', index_col=0)
    emg_score_df = pd.read_csv('./results/Syn/scores/emg_pca_scores.csv', index_col=0)
    tact_score_df = pd.read_csv('./results/Syn/scores/tact_scores.csv', index_col=0)

    all_param = list(itertools.product(perc_syns, families, l1VSl2, c_param))
    kin_data_and_iter = [[kin_score_df, extra_data, x, cv, kin_bins] for x in all_param]
    emg_pca_data_and_iter = [[emg_score_df, extra_data, x, cv, emg_bins] for x in all_param]
    tact_data_and_iter = [[tact_score_df, extra_data, x, cv, tact_bins] for x in all_param]

    """TEST"""
    # emg_nmf_iter = [[x, cv, emg_bins] for x in all_param]  # NMF data will be opened in the function for each % of synergies
    # nmf_param = list(itertools.product(perc_syns, families, c_param))
    # emg_nmf_iter = [[x, cv, emg_bins] for x in nmf_param]  # NMF data will be opened in the function for each % of synergies

    # multiprocessing
    with Pool() as pool:

        result_kin = pool.map_async(kin_syn_classif, kin_data_and_iter)
        result_emg_pca = pool.map_async(emg_pca_syn_classif, emg_pca_data_and_iter)
        # result_emg_nmf = pool.map_async(emg_nmf_syn_classif, emg_nmf_iter)
        result_tact = pool.map_async(tact_syn_classif, tact_data_and_iter)

        for res_kin in result_kin.get():
            # print(res_kin)
            wr.writerow(res_kin)

        print("Kinematic classification done!")

        for res_emg_pca in result_emg_pca.get():
            # print(res_emg_pca)
            wr.writerow(res_emg_pca)

        print("EMG PCA classification done!")

        # for res_emg_nmf in result_emg_nmf.get():
        #     # print(res_emg_nmf)
        #     wr.writerow(res_emg_nmf)
        #
        for res_tact in result_tact.get():
            # print(res_tact)
            wr.writerow(res_tact)

        print("Tactile classification done!")

    print("Single source classification done!!")
    result_file.close()




def print_syn_results():

    """Very preliminar function. Need to be updated"""

    # plt.close()
    # cols = ['Kind', 'Family', 'Perc', 'Acc']
    # results_df = pd.read_csv('./results/Syn/accuracy/syn_results.csv', header=None)
    # results_df.columns = cols
    # i = sns.pointplot(data=results_df, x="Perc", y="Acc", order=[100, 90, 80, 70, 60, 50, 40, 20, 10], errorbar='ci', errwidth='.75', capsize=.2, color="0")
    # i.set(ylabel="Accuracy (95% ci)")
    # i.set(xlabel="Percentage of Synergies")
    # i.axhline(33, color='b', linestyle='--', label='Chance level')
    # i.axhline(55.52, color='r', linestyle='--', label='Raw Classifier')
    # plt.legend()
    # i.set_ylim([29, 64])
    # sns.move_legend(i, "best")
    # # plt.show()
    # plt.savefig('./results/Syn/plots/drop_syn_acc.png', dpi=600)

    plt.close()
    cols = ['Kind', 'Perc', 'Family', 'L1vsL2', 'C', 'Acc', 'Mean']
    results_df = pd.read_csv('./results/Syn/accuracy/syn_results.csv', header=None)
    results_df.columns = cols

    kin_results_df = results_df.loc[results_df['Kind'] == 'Kin']
    emg_pca_results_df = results_df.loc[results_df['Kind'] == 'EMG PCA']
    # emg_nmf_results_df = results_df.loc[results_df['Kind'] == 'EMG NMF']
    tact_results_df = results_df.loc[results_df['Kind'] == 'Tact']

    ## BEST RESULTS

    perc_values = np.flip(np.unique(results_df['Perc']))
    l1vsl2_values = np.unique(results_df['L1vsL2'])
    c_values = np.unique(results_df['C'])

    kin_best_acc = np.zeros((len(perc_values),5))
    kin_best_params = [[[], []]] * len(perc_values)

    emg_pca_best_acc = np.zeros((len(perc_values),5))
    emg_pca_best_params = [[[], []]] * len(perc_values)

    # emg_nmf_best_acc = np.zeros((len(perc_values),))
    # emg_nmf_best_params = [[[], []]] * len(perc_values)

    tact_best_acc = np.zeros((len(perc_values),5))
    tact_best_params = [[[], []]] * len(perc_values)

    for iter_perc in range(0, len(perc_values)):
        for l1 in l1vsl2_values:
            for c in c_values:

                kin_sel = kin_results_df.loc[
                    (kin_results_df['Perc'] == perc_values[iter_perc]) & (kin_results_df['L1vsL2'] == l1) & (
                            kin_results_df['C'] == c)]
                kin_sel_mean_acc = kin_sel['Mean'].mean()

                if kin_sel_mean_acc > kin_best_acc[iter_perc].mean():
                    kin_best_acc[iter_perc] = kin_sel['Mean']
                    kin_best_params[iter_perc] = [l1, c]

                emg_pca_sel = emg_pca_results_df.loc[
                    (emg_pca_results_df['Perc'] == perc_values[iter_perc]) & (emg_pca_results_df['L1vsL2'] == l1) & (
                                emg_pca_results_df['C'] == c)]
                emg_pca_sel_mean_acc = emg_pca_sel['Mean'].mean()

                if emg_pca_sel_mean_acc > emg_pca_best_acc[iter_perc].mean():
                    emg_pca_best_acc[iter_perc] = emg_pca_sel['Mean']
                    emg_pca_best_params[iter_perc] = [l1, c]

                # emg_nmf_sel = emg_nmf_results_df.loc[
                #     (emg_nmf_results_df['Perc'] == perc_values[iter_perc]) & (emg_nmf_results_df['L1vsL2'] == l1) & (
                #                 emg_nmf_results_df['C'] == c)]
                # emg_nmf_sel_mean_acc = emg_nmf_sel['Mean'].mean()
                # 
                # if emg_nmf_sel_mean_acc > emg_nmf_best_acc[iter_perc].mean():
                #     emg_nmf_best_acc[iter_perc] = emg_nmf_sel['Mean']
                #     emg_nmf_best_params[iter_perc] = [l1, c]

                tact_sel = tact_results_df.loc[
                    (tact_results_df['Perc'] == perc_values[iter_perc]) & (tact_results_df['L1vsL2'] == l1) & (
                                tact_results_df['C'] == c)]
                tact_sel_mean_acc = tact_sel['Mean'].mean()

                if tact_sel_mean_acc > tact_best_acc[iter_perc].mean():
                    tact_best_acc[iter_perc] = tact_sel['Mean']
                    tact_best_params[iter_perc] = [l1, c]

    syn_cols = ["Source"]
    syn_cols.extend(perc_values)
    syn_best_df = pd.DataFrame(columns=syn_cols)

    kin_aux_df = pd.DataFrame(data=kin_best_acc.transpose(), columns=perc_values)
    kin_aux_df.insert(0, "Source", ["Kin"] * 5)

    emg_pca_aux_df = pd.DataFrame(data=emg_pca_best_acc.transpose(), columns=perc_values)
    emg_pca_aux_df.insert(0, "Source", ["EMG PCA"] * 5)

    tact_aux_df = pd.DataFrame(data=tact_best_acc.transpose(), columns=perc_values)
    tact_aux_df.insert(0, "Source", ["Tact"] * 5)

    syn_best_df = syn_best_df.append(kin_aux_df)
    syn_best_df = syn_best_df.append(emg_pca_aux_df)
    syn_best_df = syn_best_df.append(tact_aux_df)

    ## LOAD RAW BEST RESULTS
    raw_cols = ['Kind', 'Family', 'bins', 'L1vsL2', 'C', 'Acc', 'Mean']
    raw_results_df = pd.read_csv('./results/Raw/accuracy/raw_results.csv', header=None)
    raw_results_df.columns = raw_cols

    kin_raw_df = raw_results_df.loc[raw_results_df["Kind"] == "Kin"]
    emg_raw_df = raw_results_df.loc[raw_results_df["Kind"] == "EMG"]
    tact_raw_df = raw_results_df.loc[raw_results_df["Kind"] == "Tactile"]

    best_kin_param = [40, 0.25, 0.1]
    best_emg_param = [10, 0, 1.5]
    best_tact_param = [5, 0.5, 0.25]

    best_raw_kin_results = kin_raw_df.loc[
        (kin_raw_df["bins"] == best_kin_param[0]) & (kin_raw_df["L1vsL2"] == best_kin_param[1]) & (
                kin_raw_df["C"] == best_kin_param[2])]["Mean"]

    best_raw_emg_results = emg_raw_df.loc[
        (emg_raw_df["bins"] == best_emg_param[0]) & (emg_raw_df["L1vsL2"] == best_emg_param[1]) & (
                    emg_raw_df["C"] == best_emg_param[2])]["Mean"]

    best_raw_tact_results = tact_raw_df.loc[
        (tact_raw_df["bins"] == best_tact_param[0]) & (tact_raw_df["L1vsL2"] == best_tact_param[1]) & (
                tact_raw_df["C"] == best_tact_param[2])]["Mean"]


    ## PLOTS
    # kin plot
    i = sns.pointplot(data=syn_best_df.loc[syn_best_df["Source"] == "Kin"], errorbar='ci', errwidth='.75', capsize=.2, color="0")
    i.set(ylabel="Accuracy (95% ci)")
    i.set(xlabel="Percentage of Synergies")
    i.set(title="Kinematic accuracy comparison")
    # i.axhline(33, color='b', linestyle='--', label='Chance level')
    # i.axhline(55.52, color='r', linestyle='--', label='Raw Classifier')
    plt.legend()
    i.set_ylim([0, 100])
    sns.move_legend(i, "best")
    plt.show()
    # plt.savefig('./results/Syn/plots/kin_drop_syn_acc.png', dpi=600)

    # EMG PCA plot
    j = sns.pointplot(data=syn_best_df.loc[syn_best_df["Source"] == "EMG PCA"], errorbar='ci', errwidth='.75', capsize=.2,
                      color="0")
    j.set(ylabel="Accuracy (95% ci)")
    j.set(xlabel="Percentage of Synergies")
    j.set(title="EMG PCA accuracy comparison")
    # j.axhline(33, color='b', linestyle='--', label='Chance level')
    # j.axhline(55.52, color='r', linestyle='--', label='Raw Classifier')
    plt.legend()
    j.set_ylim([0, 100])
    sns.move_legend(j, "best")
    plt.show()
    # plt.savefig('./results/Syn/plots/kin_drop_syn_acc.png', dpi=600)

    # tact plot
    k = sns.pointplot(data=syn_best_df.loc[syn_best_df["Source"] == "Tact"], errorbar='ci', errwidth='.75', capsize=.2,
                      color="0")
    k.set(ylabel="Accuracy (95% ci)")
    k.set(xlabel="Percentage of Synergies")
    k.set(title="Tactile accuracy comparison")
    # k.axhline(33, color='b', linestyle='--', label='Chance level')
    # k.axhline(55.52, color='r', linestyle='--', label='Raw Classifier')
    plt.legend()
    k.set_ylim([0, 100])
    sns.move_legend(k, "best")
    plt.show()
    # plt.savefig('./results/Syn/plots/kin_drop_syn_acc.png', dpi=600)


    a=1


