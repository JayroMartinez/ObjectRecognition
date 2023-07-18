import numpy as np
import itertools
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from collections import Counter
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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



def emg_syn_extraction(data):

    emg_scaled = StandardScaler().fit_transform(data)  # Z score
    emg_df = pd.DataFrame(emg_scaled)
    pca_mod = PCA()

    emg_scores = pca_mod.fit_transform(emg_df)
    pd.DataFrame(emg_scores).to_csv('./results/Syn/scores/emg_scores.csv')

    emg_syns = pca_mod.components_  # Each column is a synergy
    pd.DataFrame(emg_syns).to_csv('./results/Syn/synergies/emg_syns.csv')

    emg_var = pca_mod.explained_variance_ratio_
    pd.DataFrame(emg_var).to_csv('./results/Syn/variance/emg_var.csv')


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


def syn_pip(data):

    result_file = open('./results/Syn/accuracy/syn_results.csv', 'a')  # Open file in append mode
    wr = csv.writer(result_file)

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
    # families = ['Ball', 'Cutlery', 'Geometric', 'Mugs', 'Plates']
    cv = 3

    kin_bins = 40
    kin_l1 = 0.25
    kin_c = 0.1

    emg_bins = 10
    emg_l1 = 0
    emg_c = 1.5

    tact_bins = 5
    tact_l1 = 0.5
    tact_c = 0.25

    top_c = 0.5

    # REMOVE NANs
    data_clean = data.dropna(axis=0, how='any')

    # # NON-NUMERIC DATA EXTRACTION & SAVING
    # extra_cols = [col for col in data_clean.columns if (col not in kin_cols) and (col not in emg_cols) and (col not in tact_cols)]
    # extra_df = data_clean[extra_cols]
    # extra_df.to_csv('./results/Syn/extra_data.csv', index=False)

    ## SYNERGY EXTRACTION AND SAVING
    # kin_syn_extraction(data_clean[kin_cols])
    # emg_syn_extraction(data_clean[emg_cols])
    # tact_syn_extraction(data_clean[tact_cols])

    ## CLASSIFICATION
    kin_score_df = pd.read_csv('./results/Syn/scores/kin_scores.csv', index_col=0)
    emg_score_df = pd.read_csv('./results/Syn/scores/emg_scores.csv', index_col=0)
    tact_score_df = pd.read_csv('./results/Syn/scores/tact_scores.csv', index_col=0)

    extra_data = pd.read_csv('./results/Syn/extra_data.csv')

    perc_syns = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]

    for p in perc_syns:

        num_syn_kin = np.ceil(len(kin_score_df.columns) * p / 100)
        num_syn_emg = np.ceil(len(emg_score_df.columns) * p / 100)
        num_syn_tact = np.ceil(len(tact_score_df.columns) * p / 100)

        kin_scores = pd.concat([kin_score_df.iloc[:, :int(num_syn_kin)], extra_data], axis=1, ignore_index=True)
        kin_scores.columns = list(kin_score_df.columns[:int(num_syn_kin)]) + list(extra_data.columns)

        emg_scores = pd.concat([emg_score_df.iloc[:, :int(num_syn_emg)], extra_data], axis=1, ignore_index=True)
        emg_scores.columns = list(emg_score_df.columns[:int(num_syn_emg)]) + list(extra_data.columns)

        tact_scores = pd.concat([tact_score_df.iloc[:, :int(num_syn_tact)], extra_data], axis=1, ignore_index=True)
        tact_scores.columns = list(tact_score_df.columns[:int(num_syn_tact)]) + list(extra_data.columns)

        # print("Kin shape:", kin_scores.shape)
        # print("EMG shape:", emg_scores.shape)
        # print("Tact shape:", tact_scores.shape)

        for family in families:

            total_score = []

            kin_dat = kin_scores.loc[kin_scores['Family'] == family]
            emg_dat = emg_scores.loc[emg_scores['Family'] == family]
            tact_dat = tact_scores.loc[tact_scores['Family'] == family]

            to_kfold = kin_dat.drop_duplicates(
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
                    tact_train_data = []
                    train_labels = []

                    trn_dropped = 0  # Number of dropped EPs in training dataset
                    tst_dropped = 0  # Number of dropped EPs in test dataset

                    for trn_iter in train_eps:

                        ep_kin_data = kin_dat.loc[kin_dat['Trial num'] == trn_iter]
                        kin_in_bins = np.array_split(ep_kin_data.drop(columns=extra_data.columns, axis=1), kin_bins)

                        ep_emg_data = emg_dat.loc[emg_dat['Trial num'] == trn_iter]
                        emg_in_bins = np.array_split(ep_emg_data.drop(columns=extra_data.columns, axis=1), emg_bins)

                        ep_tact_data = tact_dat.loc[tact_dat['Trial num'] == trn_iter]
                        tact_in_bins = np.array_split(ep_tact_data.drop(columns=extra_data.columns, axis=1), tact_bins)

                        with warnings.catch_warnings():
                            warnings.filterwarnings('error')
                            try:

                                kin_bin_mean = [np.nanmean(x, axis=0) for x in kin_in_bins]  # size = [num_bins] X [64]
                                flat_kin_mean = list(
                                    itertools.chain.from_iterable(kin_bin_mean))  # size = [num_bins X 64] (unidimensional)

                                emg_bin_mean = [np.nanmean(x, axis=0) for x in emg_in_bins]  # size = [num_bins] X [64]
                                flat_emg_mean = list(
                                    itertools.chain.from_iterable(
                                        emg_bin_mean))  # size = [num_bins X 64] (unidimensional)

                                tact_bin_mean = [np.nanmean(x, axis=0) for x in
                                                 tact_in_bins]  # size = [num_bins] X [64]
                                flat_tact_mean = list(
                                    itertools.chain.from_iterable(
                                        tact_bin_mean))  # size = [num_bins X 64] (unidimensional)

                                kin_train_data.append(flat_kin_mean)
                                emg_train_data.append(flat_emg_mean)
                                tact_train_data.append(flat_tact_mean)
                                train_labels.append(np.unique(ep_kin_data['Given Object'])[0])

                            except RuntimeWarning:
                                # print("Dropped EP", trn_iter, "from family ", family)
                                trn_dropped += 1

                    kin_test_data = []
                    emg_test_data = []
                    tact_test_data = []
                    test_labels = []

                    for tst_iter in test_eps:

                        ep_kin_data = kin_dat.loc[kin_dat['Trial num'] == tst_iter]
                        kin_in_bins = np.array_split(ep_kin_data.drop(columns=extra_data.columns, axis=1),
                                                     kin_bins)

                        ep_emg_data = emg_dat.loc[emg_dat['Trial num'] == tst_iter]
                        emg_in_bins = np.array_split(ep_emg_data.drop(columns=extra_data.columns, axis=1),
                                                     emg_bins)

                        ep_tact_data = tact_dat.loc[tact_dat['Trial num'] == tst_iter]
                        tact_in_bins = np.array_split(ep_tact_data.drop(columns=extra_data.columns, axis=1),
                                                      tact_bins)

                        with warnings.catch_warnings():
                            warnings.filterwarnings('error')
                            try:

                                kin_bin_mean = [np.nanmean(x, axis=0) for x in
                                                kin_in_bins]  # size = [num_bins] X [64]
                                flat_kin_mean = list(
                                    itertools.chain.from_iterable(
                                        kin_bin_mean))  # size = [num_bins X 64] (unidimensional)

                                emg_bin_mean = [np.nanmean(x, axis=0) for x in
                                                emg_in_bins]  # size = [num_bins] X [64]
                                flat_emg_mean = list(
                                    itertools.chain.from_iterable(
                                        emg_bin_mean))  # size = [num_bins X 64] (unidimensional)

                                tact_bin_mean = [np.nanmean(x, axis=0) for x in
                                                 tact_in_bins]  # size = [num_bins] X [64]
                                flat_tact_mean = list(
                                    itertools.chain.from_iterable(
                                        tact_bin_mean))  # size = [num_bins X 64] (unidimensional)

                                kin_test_data.append(flat_kin_mean)
                                emg_test_data.append(flat_emg_mean)
                                tact_test_data.append(flat_tact_mean)
                                test_labels.append(np.unique(ep_kin_data['Given Object'])[0])

                            except RuntimeWarning:
                                # print("Dropped EP", tst_iter, "from family ", family)
                                tst_dropped += 1

                    # compute weights (because unbalanced dataset)
                    # weights = compute_sample_weight(class_weight='balanced', y=train_labels)

                    # build kinematic model
                    kin_log_model = LogisticRegression(penalty='elasticnet', C=kin_c, random_state=rnd_st,
                                                       solver='saga', max_iter=25000, multi_class='ovr', n_jobs=-1,
                                                       l1_ratio=kin_l1)

                    # train kinematic model
                    kin_log_model.fit(X=kin_train_data, y=train_labels)

                    # build EMG model
                    emg_log_model = LogisticRegression(penalty='elasticnet', C=emg_c,
                                                       random_state=rnd_st,
                                                       solver='saga', max_iter=25000, multi_class='ovr',
                                                       n_jobs=-1,
                                                       l1_ratio=emg_l1)

                    # train EMG model
                    emg_log_model.fit(X=emg_train_data, y=train_labels)


                    # build Tactile model
                    tact_log_model = LogisticRegression(penalty='elasticnet', C=tact_c,
                                                        random_state=rnd_st,
                                                        solver='saga', max_iter=25000, multi_class='ovr',
                                                        n_jobs=-1,
                                                        l1_ratio=tact_l1)

                    # train EMG model
                    tact_log_model.fit(X=tact_train_data, y=train_labels)

                    # get prediction probabilities from first layer to train second layer
                    kin_model_pred_proba = kin_log_model.predict_proba(X=kin_train_data)
                    emg_model_pred_proba = emg_log_model.predict_proba(X=emg_train_data)
                    tact_model_pred_proba = tact_log_model.predict_proba(X=tact_train_data)

                    # pred_proba = np.concatenate([kin_model_pred_proba, emg_model_pred_proba, tact_model_pred_proba], axis=1)
                    pred_proba = np.concatenate([kin_model_pred_proba, emg_model_pred_proba, tact_model_pred_proba], axis=1)

                    # build & train top layer classifier
                    top_log_model = LogisticRegression(C=top_c, random_state=rnd_st, solver='saga',
                                                       max_iter=25000,
                                                       multi_class='ovr', n_jobs=-1)
                    top_log_model.fit(X=pred_proba, y=train_labels)

                    # get probabilities from first layer on test set to feed the second layer
                    kin_test_pred = kin_log_model.predict_proba(X=kin_test_data)
                    emg_test_pred = emg_log_model.predict_proba(X=emg_test_data)
                    tact_test_pred = tact_log_model.predict_proba(X=tact_test_data)
                    test_proba = np.concatenate([kin_test_pred, emg_test_pred, tact_test_pred], axis=1)
                    # test_proba = np.concatenate([emg_test_pred, tact_test_pred], axis=1)

                    # get prediction accuracy from second layer
                    sc = round(top_log_model.score(X=test_proba, y=test_labels) * 100, 2)
                    # total_score.append(sc)

                    res = ['Hierarchical']
                    res.extend([family, p])
                    res.append(sc)
                    # res.append(round(np.mean(total_score), 2))
                    wr.writerow(res)
                    print(res)

    result_file.close()
    print("DONE !!!")



def print_syn_results():

    plt.close()
    cols = ['Kind', 'Family', 'Perc', 'Acc']
    results_df = pd.read_csv('./results/Syn/accuracy/syn_results.csv', header=None)
    results_df.columns = cols
    i = sns.pointplot(data=results_df, x="Perc", y="Acc", order=[100, 90, 80, 70, 60, 50, 40, 20, 10], errorbar='ci', errwidth='.75', capsize=.2, color="0")
    i.set(ylabel="Accuracy (95% ci)")
    i.set(xlabel="Percentage of Synergies")
    i.axhline(33, color='b', linestyle='--', label='Chance level')
    i.axhline(55.52, color='r', linestyle='--', label='Raw Classifier')
    plt.legend()
    i.set_ylim([29, 64])
    sns.move_legend(i, "best")
    # plt.show()
    plt.savefig('./results/Syn/plots/drop_syn_acc.png', dpi=600)