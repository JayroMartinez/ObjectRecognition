import numpy as np
import os
import itertools
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from collections import Counter
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
import csv
import re
from scipy.stats import zscore
import pandas as pd
pd.DataFrame.iteritems = pd.DataFrame.items
from multiprocessing.pool import Pool
import random
import seaborn as sns
import matplotlib.pyplot as plt
from statannotations.Annotator import Annotator
from statannot import add_stat_annotation, statannot
import glob
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from sklearn.impute import SimpleImputer
import statistics
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDClassifier
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight

from classification import get_raw_best_params

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def get_best_params_hier(type, discard):
    # SET AND READ RESULT FILE
    if type == 'all':
        res_file_name = './results/Syn/accuracy/syn_hier'
    elif type == 'clustering':
        res_file_name = './results/Syn/accuracy/subj_clust_syn_hier'
    else:  # early enclosure
        res_file_name = './results/Early Enclosure/accuracy/alternative_syn_hier'

    if discard == 'less':
        res_file_name += '_results.csv'
    else:
        res_file_name += '_results_decr.csv'

    results_df = pd.read_csv(res_file_name, header=None)
    cols = ['Kind', 'Family', 'Perc', 'C', 'Acc']
    results_df.columns = cols

    perc_values = np.flip(np.unique(results_df['Perc']))
    c_values = np.unique(results_df['C'])

    hier_best_acc = np.zeros((len(perc_values), 5))
    hier_best_params = [[]] * len(perc_values)

    for iter_perc in range(0, len(perc_values)):
        for c in c_values:

            hier_sel = results_df.loc[
                (results_df['Perc'] == perc_values[iter_perc]) & (results_df['C'] == c)]
            hier_sel_mean_acc = hier_sel.groupby('Family')['Acc'].mean()

            if hier_sel_mean_acc.mean() > hier_best_acc[iter_perc].mean():
                hier_best_acc[iter_perc] = hier_sel_mean_acc
                hier_best_params[iter_perc] = c

    syn_best_param_df = pd.DataFrame(data=[hier_best_params], columns=perc_values)
    syn_best_param_df.insert(0, "Source", ["Hier"])

    # SET AND SAVE BEST PARAMS FILE
    if type == 'all':
        best_file_name = './results/Syn/best_syn_params'
    elif type == 'clustering':
        best_file_name = './results/Syn/subj_clust_best_syn_params'
    else:  # early enclosure
        best_file_name = './results/Early Enclosure/alternative_best_syn_params'

    if discard == 'less':
        best_file_name += '.csv'
    else:
        best_file_name += '_decr.csv'

    syn_best_param_df.to_csv(best_file_name, mode='a', index=False, header=False)


def get_best_params_multi(type, discard):

    # SET AND READ RESULT FILE
    if type == 'all':
        res_file_name = './results/Syn/accuracy/syn_multi'
    elif type == 'clustering':
        res_file_name = './results/Syn/accuracy/subj_clust_syn_multi'
    else:  # early enclosure
        res_file_name = './results/Early Enclosure/accuracy/alternative_syn_multi'

    if discard == 'less':
        res_file_name += '_results.csv'
    else:
        res_file_name += '_results_decr.csv'

    results_df = pd.read_csv(res_file_name, header=None)
    cols = ['Kind', 'Family', 'Perc', 'L1vsL2', 'C', 'Acc']
    results_df.columns = cols

    perc_values = np.flip(np.unique(results_df['Perc']))
    l1vsl2_values = np.unique(results_df['L1vsL2'])
    c_values = np.unique(results_df['C'])

    multi_best_acc = np.zeros((len(perc_values), 5))
    multi_best_params = [[[], []]] * len(perc_values)

    for iter_perc in range(0, len(perc_values)):
        for l1 in l1vsl2_values:
            for c in c_values:

                multi_sel = results_df.loc[
                    (results_df['Perc'] == perc_values[iter_perc]) & (results_df['L1vsL2'] == l1) & (
                            results_df['C'] == c)]
                multi_sel_mean_acc = multi_sel.groupby('Family')['Acc'].mean()

                if multi_sel_mean_acc.mean() > multi_best_acc[iter_perc].mean():
                    multi_best_acc[iter_perc] = multi_sel_mean_acc
                    multi_best_params[iter_perc] = [l1, c]

    syn_best_param_df = pd.DataFrame(data=[multi_best_params], columns=perc_values)
    syn_best_param_df.insert(0, "Source", ["Multi"])

    # SET AND SAVE BEST PARAMS FILE
    if type == 'all':
        best_file_name = './results/Syn/best_syn_params'
    elif type == 'clustering':
        best_file_name = './results/Syn/subj_clust_best_syn_params'
    else:  # early enclosure
        best_file_name = './results/Early Enclosure/alternative_best_syn_params'

    if discard == 'less':
        best_file_name += '.csv'
    else:
        best_file_name += '_decr.csv'

    syn_best_param_df.to_csv(best_file_name, mode='a', index=False, header=False)


def get_best_params_single(type, discard):

    # SET AND READ RESULT FILE
    if type == 'all':
        res_file_name = './results/Syn/accuracy/syn'
    elif type == 'clustering':
        res_file_name = './results/Syn/accuracy/subj_clust_syn'
    else:  # early enclosure
        res_file_name = './results/Early Enclosure/accuracy/alternative_syn'

    if discard == 'less':
        res_file_name += '_results.csv'
    else:
        res_file_name += '_results_decr.csv'

    # READ AND SELECT DATA
    results_df = pd.read_csv(res_file_name, header=None)

    cols = ['Kind', 'Perc', 'Family', 'L1vsL2', 'C', 'Acc', 'Mean']
    results_df.columns = cols
    kin_results_df = results_df.loc[results_df['Kind'] == 'Kin']
    emg_pca_results_df = results_df.loc[results_df['Kind'] == 'EMG PCA']
    tact_results_df = results_df.loc[results_df['Kind'] == 'Tact']

    perc_values = np.flip(np.unique(results_df['Perc']))
    c_values = np.unique(results_df['C'])
    l1vsl2_values = np.unique(results_df['L1vsL2'])

    kin_best_acc = np.zeros((len(perc_values), 5))
    kin_best_params = [[[], []]] * len(perc_values)

    emg_pca_best_acc = np.zeros((len(perc_values), 5))
    emg_pca_best_params = [[[], []]] * len(perc_values)

    tact_best_acc = np.zeros((len(perc_values), 5))
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

                tact_sel = tact_results_df.loc[
                    (tact_results_df['Perc'] == perc_values[iter_perc]) & (tact_results_df['L1vsL2'] == l1) & (
                            tact_results_df['C'] == c)]
                tact_sel_mean_acc = tact_sel['Mean'].mean()

                if tact_sel_mean_acc > tact_best_acc[iter_perc].mean():
                    tact_best_acc[iter_perc] = tact_sel['Mean']
                    tact_best_params[iter_perc] = [l1, c]

    syn_cols = ["Source"]
    syn_cols.extend(perc_values)
    syn_best_acc_df = pd.DataFrame(columns=syn_cols)

    # BEST HYPERPARAMETERS
    syn_best_param_df = pd.DataFrame(columns=syn_cols)

    kin_l1c_param = pd.DataFrame(data=[kin_best_params], columns=perc_values)
    kin_l1c_param.insert(0, "Source", ["Kin"])

    emg_pca_l1c_param = pd.DataFrame(data=[emg_pca_best_params], columns=perc_values)
    emg_pca_l1c_param.insert(0, "Source", ["EMG PCA"])

    tact_l1c_param = pd.DataFrame(data=[tact_best_params], columns=perc_values)
    tact_l1c_param.insert(0, "Source", ["Tact"])

    syn_best_param_df = pd.concat([syn_best_param_df, kin_l1c_param])
    syn_best_param_df = pd.concat([syn_best_param_df, emg_pca_l1c_param])
    syn_best_param_df = pd.concat([syn_best_param_df, tact_l1c_param])

    # SET AND SAVE BEST PARAMS FILE
    if type == 'all':
        best_file_name = './results/Syn/best_syn_params'
    elif type == 'clustering':
        best_file_name = './results/Syn/subj_clust_best_syn_params'
    else:  # early enclosure
        best_file_name = './results/Early Enclosure/alternative_best_syn_params'

    if discard == 'less':
        best_file_name += '.csv'
    else:
        best_file_name += '_decr.csv'

    syn_best_param_df.to_csv(best_file_name, mode='a', index=False)


def fam_kin_syn_classif(input_data):

    kin_scores = input_data[0]
    extra_data = input_data[1]
    perc_syns = input_data[2][0]
    l1VSl2 = input_data[2][1]
    c_param = input_data[2][2]
    cv = input_data[3]
    kin_bins = input_data[4]

    discard = input_data[5]

    total_score = []

    num_syns = np.ceil(len(kin_scores.columns) * perc_syns / 100)
    extra_data.reset_index(inplace=True, drop=True)
    kin_scores.reset_index(inplace=True, drop=True)

    if discard == 'less':
        data_df = pd.concat([kin_scores.iloc[:, 0:int(num_syns)], extra_data], axis=1)  # keeps most relevant
    else:
        data_df = pd.concat([kin_scores.iloc[:, -int(num_syns):], extra_data], axis=1) # discards most relevant

    to_kfold = data_df.drop_duplicates(subset=['Trial num', 'Family'])

    random_states = [42, 43, 44]
    for rnd_st in random_states:

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=rnd_st)
        # WARNING: the skf.split returns the indexes
        for train, test in skf.split(to_kfold['Trial num'].astype(int), to_kfold['Family'].astype(str)):

            train_trials = to_kfold.iloc[train]['Trial num']  # because skf.split returns the indexes
            test_trials = to_kfold.iloc[test]['Trial num']  # because skf.split returns the indexes

            train_data = []
            train_labels = []

            dropped = 0  # Number of dropped EPs

            # take each ep, create bins & compute mean
            for trn_iter in train_trials:

                train_tri = data_df.loc[data_df['Trial num'] == trn_iter]
                tr_kin_data = train_tri.drop(columns=extra_data.columns)
                tr_in_bins = np.array_split(tr_kin_data, kin_bins)

                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    try:
                        tr_bin_mean = [np.nanmean(x, axis=0) for x in tr_in_bins]  # size = [num_bins] X [64]
                        flat_tr_mean = list(
                            itertools.chain.from_iterable(tr_bin_mean))  # size = [num_bins X 64] (unidimensional)
                        train_data.append(flat_tr_mean)
                        train_labels.append(np.unique(train_tri['Family'])[0])
                    except RuntimeWarning:
                        # print("Dropped EP", trn_iter, "from family ", family)
                        dropped += 1

            test_data = []
            test_labels = []

            for tst_iter in test_trials:

                test_tri = data_df.loc[data_df['Trial num'] == tst_iter]
                tst_kin_data = test_tri.drop(columns=extra_data.columns)
                tst_in_bins = np.array_split(tst_kin_data, kin_bins)

                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    try:
                        tst_bin_mean = [np.nanmean(x, axis=0) for x in tst_in_bins]  # size = [num_bins] X [64]
                        flat_tst_mean = list(
                            itertools.chain.from_iterable(tst_bin_mean))  # size = [num_bins X 64] (unidimensional)
                        test_data.append(flat_tst_mean)
                        test_labels.append(np.unique(test_tri['Family'])[0])
                    except RuntimeWarning:
                        # print("Dropped EP", tst_iter, "from family ", family)
                        dropped += 1

            # # build model
            # weights = compute_sample_weight(class_weight='balanced', y=train_labels)
            # log_model = LogisticRegression(penalty='elasticnet', C=c_param, class_weight='balanced', random_state=rnd_st,
            #                                solver='saga', max_iter=25000, tol=0.000001, multi_class='multinomial', n_jobs=-1,
            #                                l1_ratio=l1VSl2)
            # # log_model = SGDClassifier(loss="log_loss", penalty='elasticnet', alpha=c_param, l1_ratio=l1VSl2, max_iter=25000, n_jobs=-1, random_state=rnd_st, class_weight='balanced')
            # # train model
            # log_model.fit(X=train_data, y=train_labels, sample_weight=weights)
            # weights_test = compute_sample_weight(class_weight='balanced', y=test_labels)
            # sc = round(log_model.score(X=test_data, y=test_labels, sample_weight=weights_test) * 100, 2)
            # total_score.append(sc)

            alpha_param = 1 / c_param  # Alpha is the inverse of regularization strength (C)
            batch_size = 50  # Define your batch size here

            # Create the SGDClassifier model with logistic regression
            sgd_model = SGDClassifier(loss='log_loss', penalty='elasticnet', alpha=alpha_param, l1_ratio=l1VSl2,
                                      random_state=rnd_st, max_iter=10000, warm_start=True, learning_rate='optimal',
                                      eta0=0.01)

            # Compute the sample weights for the entire dataset
            classes = np.unique(train_labels)
            # Compute class weights
            class_weights = compute_class_weight('balanced', classes=classes, y=train_labels)
            # Convert class weights to dictionary format
            class_weight_dict = {classes[i]: class_weights[i] for i in range(len(classes))}

            # Mini-batch training
            for _ in range(100000 // batch_size):  # Assuming 100000 iterations as max, adjust as needed
                # Randomly sample a batch of data
                batch_indices = np.random.choice(range(len(train_data)), size=batch_size, replace=False)
                batch_indices_list = batch_indices.tolist()

                batch_data = [train_data[i] for i in batch_indices_list]
                batch_labels = [train_labels[i] for i in batch_indices_list]
                # batch_weights = [trn_weights[i] for i in batch_indices_list]
                batch_weights = np.array([class_weight_dict[label] for label in batch_labels])

                # Partial fit on the batch
                sgd_model.partial_fit(batch_data, batch_labels, classes=classes, sample_weight=batch_weights)

            # Evaluate the model
            sc = round(sgd_model.score(X=test_data, y=test_labels) * 100, 2)
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
    l1VSl2 = input_data[2][1]
    c_param = input_data[2][2]
    cv = input_data[3]
    emg_pca_bins = input_data[4]

    discard = input_data[5]

    total_score = []

    num_syns = np.ceil(len(emg_pca_scores.columns) * perc_syns / 100)
    extra_data.reset_index(inplace=True, drop=True)
    emg_pca_scores.reset_index(inplace=True, drop=True)

    if discard == 'less':
        data_df = pd.concat([emg_pca_scores.iloc[:, 0:int(num_syns)], extra_data], axis=1)  # keeps most relevant
    else:
        data_df = pd.concat([emg_pca_scores.iloc[:, -int(num_syns):], extra_data], axis=1)  # discards most relevant

    selected_df = data_df
    to_kfold = selected_df.drop_duplicates(subset=['Trial num', 'Family'])

    random_states = [42, 43, 44]
    for rnd_st in random_states:

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=rnd_st)
        # WARNING: the skf.split returns the indexes
        for train, test in skf.split(to_kfold['Trial num'].astype(int), to_kfold['Family'].astype(str)):

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
                        #train_labels.append(np.unique(train_tri['Family'])[0])
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
                        test_labels.append(np.unique(test_tri['Family'])[0])
                    except RuntimeWarning:
                        # print("Dropped EP", tst_iter, "from family ", family)
                        dropped += 1

            # build model
            log_model = LogisticRegression(penalty='elasticnet', C=c_param, class_weight='balanced',
                                           random_state=rnd_st,
                                           solver='saga', max_iter=25000, multi_class='multinomial', n_jobs=-1,
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


def fam_tact_syn_classif(input_data):

    tact_scores = input_data[0]
    extra_data = input_data[1]
    perc_syns = input_data[2][0]
    l1VSl2 = input_data[2][1]
    c_param = input_data[2][2]
    cv = input_data[3]
    tact_bins = input_data[4]

    discard = input_data[5]

    total_score = []

    num_syns = np.ceil(len(tact_scores.columns) * perc_syns / 100)
    extra_data.reset_index(inplace=True, drop=True)
    tact_scores.reset_index(inplace=True, drop=True)

    if discard == 'less':
        data_df = pd.concat([tact_scores.iloc[:, 0:int(num_syns)], extra_data], axis=1) # keeps most relevant
    else:
        data_df = pd.concat([tact_scores.iloc[:, -int(num_syns):], extra_data], axis=1)  # discards most relevant

    selected_df = data_df
    to_kfold = selected_df.drop_duplicates(subset=['Trial num', 'Family'])

    random_states = [42, 43, 44]
    for rnd_st in random_states:

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=rnd_st)
        # WARNING: the skf.split returns the indexes
        for train, test in skf.split(to_kfold['Trial num'].astype(int), to_kfold['Family'].astype(str)):

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
                        train_labels.append(np.unique(train_tri['Family'])[0])
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
                        test_labels.append(np.unique(test_tri['Family'])[0])
                    except RuntimeWarning:
                        # print("Dropped EP", tst_iter, "from family ", family)
                        dropped += 1

            # build model
            weights = compute_sample_weight(class_weight='balanced', y=train_labels)
            log_model = LogisticRegression(penalty='elasticnet', C=c_param, class_weight='balanced',
                                           random_state=rnd_st,
                                           solver='saga', max_iter=25000, multi_class='multinomial', n_jobs=-1,
                                           l1_ratio=l1VSl2)
            # train model
            log_model.fit(X=train_data, y=train_labels, sample_weight=weights)
            sc = round(log_model.score(X=test_data, y=test_labels) * 100, 2)
            total_score.append(sc)

    result = ['Tact']
    result.extend(input_data[2])
    result.append(total_score)
    result.append(round(np.mean(total_score), 2))
    # print("RESULT:", result)

    return result


def fam_syn_single_source_classification(type, discard):

    cv = 3
    [kin_params, emg_params, tact_params] = get_raw_best_params()
    kin_bins = kin_params[0]
    emg_bins = emg_params[0]
    tact_bins = tact_params[0]
    perc_syns = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
    l1VSl2 = [0, 0.25, 0.5, 0.75, 1]
    # c_param = [0.01, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5]
    c_param = [0.01, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5]

    # LOAD EXTRA DATA
    if type == 'early':
        extra_data = pd.read_csv('./results/Early Enclosure/alternative_early_enclosure_extra_data.csv')
    else:
        extra_data = pd.read_csv('./results/Syn/extra_data.csv')

    # SET AND OPEN RESULT FILE
    if type == 'all':
        res_file_name = './results/Syn/accuracy/fam_syn_results'
    elif type == 'clustering':
        res_file_name = './results/Syn/accuracy/fam_subj_clust_syn_results'
    elif type == 'alternative':
        res_file_name = './results/Syn/accuracy/fam_alternative_syn_results'
    else:  # early enclosure
        res_file_name = './results/Early Enclosure/accuracy/fam_alternative_syn_results'

    if discard == 'less':
        res_file_name += '.csv'
    else:
        res_file_name += '_decr.csv'

    result_file = open(res_file_name, 'a')
    wr = csv.writer(result_file)

    # GET SCORES
    if type == 'all':
        kin_score_df = pd.read_csv('./results/Syn/scores/kin_scores.csv', index_col=0)
        # emg_score_df = pd.read_csv('./results/Syn/scores/emg_pca_scores.csv', index_col=0)
        # tact_score_df = pd.read_csv('./results/Syn/scores/tact_scores.csv', index_col=0)
    elif type == 'clustering':
        kin_score_df = pd.read_csv('./results/Syn/scores/reordered_kin_scores.csv', index_col=0)
        # emg_score_df = pd.read_csv('./results/Syn/scores/reordered_emg_pca_scores.csv', index_col=0)
        # tact_score_df = pd.read_csv('./results/Syn/scores/reordered_tact_scores.csv', index_col=0)
    elif type == 'alternative':
        kin_score_df = pd.read_csv('./results/Syn/scores/reordered_alternative_kin_scores.csv', index_col=0)
    else:  # early enclosure
        kin_score_df = pd.read_csv('./results/Early Enclosure/scores/alternative_reordered_kin_scores.csv', index_col=0)
        # emg_score_df = pd.read_csv('./results/Early Enclosure/scores/alternative_reordered_emg_pca_scores.csv', index_col=0)
        # tact_score_df = pd.read_csv('./results/Early Enclosure/scores/alternative_reordered_tact_scores.csv', index_col=0)

    # BUILD ITERABLE STRUCTURES
    all_param = list(itertools.product(perc_syns, l1VSl2, c_param))
    kin_data_and_iter = [[kin_score_df, extra_data, x, cv, kin_bins, discard] for x in all_param]
    # emg_pca_data_and_iter = [[emg_score_df, extra_data, x, cv, emg_bins, discard] for x in all_param]
    # tact_data_and_iter = [[tact_score_df, extra_data, x, cv, tact_bins, discard] for x in all_param]

    # multiprocessing
    with Pool() as pool:

        result_kin = pool.map_async(fam_kin_syn_classif, kin_data_and_iter)
        # result_emg_pca = pool.map_async(emg_pca_syn_classif, emg_pca_data_and_iter)
        # result_tact = pool.map_async(fam_tact_syn_classif, tact_data_and_iter)

        for res_kin in result_kin.get():
            wr.writerow(res_kin)
        print("Kinematic classification done!")

        # for res_emg_pca in result_emg_pca.get():
        #     wr.writerow(res_emg_pca)
        # # print("EMG PCA classification done!")

        # for res_tact in result_tact.get():
        #     wr.writerow(res_tact)
        # # print("Tactile classification done!")

    # print("Single source classification done!!")
    result_file.close()


def hierarchical_syn_classification(type, discard):

    cv = 3

    [[kin_bins, kin_l1, kin_c], [emg_bins, emg_l1, emg_c], [tact_bins, tact_l1, tact_c]] = get_raw_best_params()

    c_values = [0.01, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5]
    perc_syns = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]

    # SET AND OPEN RESULT FILE
    if type == 'all':
        res_file_name = './results/Syn/accuracy/fam_syn_hier_results'
        best_params_file = './results/Syn/fam_best_syn_params'
    elif type == 'clustering':
        res_file_name = './results/Syn/accuracy/fam_subj_clust_syn_hier_results'
        best_params_file = './results/Syn/fam_subj_clust_best_syn_params'
    else:  # early enclosure
        res_file_name = './results/Early Enclosure/accuracy/fam_alternative_syn_hier_results'
        best_params_file = './results/Early Enclosure/fam_alternative_best_syn_params'

    if discard == 'less':
        res_file_name += '.csv'
        best_params_file += '.csv'
    else:
        res_file_name += '_decr.csv'
        best_params_file += '_decr.csv'

    result_file = open(res_file_name, 'a')
    wr = csv.writer(result_file)

    # GET SCORES
    if type == 'all':
        kin_score_df = pd.read_csv('./results/Syn/scores/kin_scores.csv', index_col=0)
        emg_score_df = pd.read_csv('./results/Syn/scores/emg_pca_scores.csv', index_col=0)
        tact_score_df = pd.read_csv('./results/Syn/scores/tact_scores.csv', index_col=0)
    elif type == 'clustering':
        kin_score_df = pd.read_csv('./results/Syn/scores/reordered_kin_scores.csv', index_col=0)
        emg_score_df = pd.read_csv('./results/Syn/scores/reordered_emg_pca_scores.csv', index_col=0)
        tact_score_df = pd.read_csv('./results/Syn/scores/reordered_tact_scores.csv', index_col=0)
    else:  # early enclosure
        kin_score_df = pd.read_csv('./results/Early Enclosure/scores/alternative_reordered_kin_scores.csv', index_col=0)
        emg_score_df = pd.read_csv('./results/Early Enclosure/scores/alternative_reordered_emg_pca_scores.csv',
                                   index_col=0)
        tact_score_df = pd.read_csv('./results/Early Enclosure/scores/alternative_reordered_tact_scores.csv',
                                    index_col=0)
    kin_score_df.reset_index(inplace=True, drop=True)
    emg_score_df.reset_index(inplace=True, drop=True)
    tact_score_df.reset_index(inplace=True, drop=True)

    # LOAD EXTRA DATA
    if type == 'early':
        extra_data = pd.read_csv('./results/Early Enclosure/alternative_early_enclosure_extra_data.csv')
    else:
        extra_data = pd.read_csv('./results/Syn/extra_data.csv')
    extra_data.reset_index(inplace=True, drop=True)

    for top_c in c_values:
        for p in perc_syns:

            num_syn_kin = np.ceil(len(kin_score_df.columns) * p / 100)
            num_syn_emg = np.ceil(len(emg_score_df.columns) * p / 100)
            num_syn_tact = np.ceil(len(tact_score_df.columns) * p / 100)

            # SELECT SYNERGIES
            if discard == 'less':
                kin_scores = pd.concat([kin_score_df.iloc[:, :int(num_syn_kin)], extra_data], axis=1, ignore_index=True)
                kin_scores.columns = list(kin_score_df.columns[:int(num_syn_kin)]) + list(extra_data.columns)
                emg_scores = pd.concat([emg_score_df.iloc[:, :int(num_syn_emg)], extra_data], axis=1, ignore_index=True)
                emg_scores.columns = list(emg_score_df.columns[:int(num_syn_emg)]) + list(extra_data.columns)
                tact_scores = pd.concat([tact_score_df.iloc[:, :int(num_syn_tact)], extra_data], axis=1, ignore_index=True)
                tact_scores.columns = list(tact_score_df.columns[:int(num_syn_tact)]) + list(extra_data.columns)
            else:
                kin_scores = pd.concat([kin_score_df.iloc[:, -int(num_syn_kin):], extra_data], axis=1, ignore_index=True)
                kin_scores.columns = list(kin_score_df.columns[-int(num_syn_kin):]) + list(extra_data.columns)
                emg_scores = pd.concat([emg_score_df.iloc[:, -int(num_syn_emg):], extra_data], axis=1, ignore_index=True)
                emg_scores.columns = list(emg_score_df.columns[-int(num_syn_emg):]) + list(extra_data.columns)
                tact_scores = pd.concat([tact_score_df.iloc[:, -int(num_syn_tact):], extra_data], axis=1, ignore_index=True)
                tact_scores.columns = list(tact_score_df.columns[:int(num_syn_tact)]) + list(extra_data.columns)


            total_score = []

            # kin_dat = kin_scores.loc[kin_scores['Family'] == family]
            # emg_dat = emg_scores.loc[emg_scores['Family'] == family]
            # tact_dat = tact_scores.loc[tact_scores['Family'] == family]

            kin_dat = kin_scores
            emg_dat = emg_scores
            tact_dat = tact_scores

            # to_kfold = kin_dat.drop_duplicates(subset=['Trial num', 'Given Object'])
            to_kfold = kin_dat.drop_duplicates(subset=['Trial num', 'Family'])

            random_states = [42, 43, 44]

            for rnd_st in random_states:

                skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=rnd_st)
                # WARNING: the skf.split returns the indexes
                # for train, test in skf.split(to_kfold['Trial num'].astype(int), to_kfold['Given Object'].astype(str)):
                for train, test in skf.split(to_kfold['Trial num'].astype(int), to_kfold['Family'].astype(str)):

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
                                # train_labels.append(np.unique(ep_kin_data['Given Object'])[0])
                                train_labels.append(np.unique(ep_kin_data['Family'])[0])

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
                                # test_labels.append(np.unique(ep_kin_data['Given Object'])[0])
                                train_labels.append(np.unique(ep_kin_data['Family'])[0])

                            except RuntimeWarning:
                                # print("Dropped EP", tst_iter, "from family ", family)
                                tst_dropped += 1

                    # compute weights (because unbalanced dataset)
                    # weights = compute_sample_weight(class_weight='balanced', y=train_labels)

                    # build kinematic model
                    kin_log_model = LogisticRegression(penalty='elasticnet', C=kin_c, random_state=rnd_st,
                                                       solver='saga', max_iter=50000, multi_class='multinomial', n_jobs=-1,
                                                       l1_ratio=kin_l1)

                    # train kinematic model
                    kin_log_model.fit(X=kin_train_data, y=train_labels)

                    # build EMG model
                    emg_log_model = LogisticRegression(penalty='elasticnet', C=emg_c,
                                                       random_state=rnd_st,
                                                       solver='saga', max_iter=50000, multi_class='multinomial',
                                                       n_jobs=-1,
                                                       l1_ratio=emg_l1)

                    # train EMG model
                    emg_log_model.fit(X=emg_train_data, y=train_labels)


                    # build Tactile model
                    tact_log_model = LogisticRegression(penalty='elasticnet', C=tact_c,
                                                        random_state=rnd_st,
                                                        solver='saga', max_iter=50000, multi_class='multinomial',
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
                                                       max_iter=50000,
                                                       multi_class='multinomial', n_jobs=-1)
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
                    res.extend([p, top_c])
                    res.append(sc)
                    # res.append(round(np.mean(total_score), 2))
                    # wr.writerow(res)
                    # print(res)

    result_file.close()
    # print("HIERARCHICAL DONE !!!")


def multi_aux_classification(input_data):

    cv = 3
    family = input_data[0][0]
    l1_param = input_data[0][1]
    c_param = input_data[0][2]
    perc = input_data[0][3]
    rnd_st = input_data[0][4]

    type = input_data[1]
    discard = input_data[2]

    [[kin_bins, kin_l1, kin_c], [emg_bins, emg_l1, emg_c], [tact_bins, tact_l1, tact_c]] = get_raw_best_params()

    # LOAD EXTRA DATA
    if type == 'early':
        extra_data = pd.read_csv('./results/Early Enclosure/alternative_early_enclosure_extra_data.csv')
    else:
        extra_data = pd.read_csv('./results/Syn/extra_data.csv')
    extra_data.reset_index(inplace=True, drop=True)

    # GET SCORES
    if type == 'all':
        kin_score_df = pd.read_csv('./results/Syn/scores/kin_scores.csv', index_col=0)
        emg_score_df = pd.read_csv('./results/Syn/scores/emg_pca_scores.csv', index_col=0)
        tact_score_df = pd.read_csv('./results/Syn/scores/tact_scores.csv', index_col=0)
    elif type == 'clustering':
        kin_score_df = pd.read_csv('./results/Syn/scores/reordered_kin_scores.csv', index_col=0)
        emg_score_df = pd.read_csv('./results/Syn/scores/reordered_emg_pca_scores.csv', index_col=0)
        tact_score_df = pd.read_csv('./results/Syn/scores/reordered_tact_scores.csv', index_col=0)
    else:  # early enclosure
        kin_score_df = pd.read_csv('./results/Early Enclosure/scores/alternative_reordered_kin_scores.csv', index_col=0)
        emg_score_df = pd.read_csv('./results/Early Enclosure/scores/alternative_reordered_emg_pca_scores.csv', index_col=0)
        tact_score_df = pd.read_csv('./results/Early Enclosure/scores/alternative_reordered_tact_scores.csv', index_col=0)
    kin_score_df.reset_index(inplace=True, drop=True)
    emg_score_df.reset_index(inplace=True, drop=True)
    tact_score_df.reset_index(inplace=True, drop=True)

    # DEFINE NUMBER OF SYNERGIES
    num_syn_kin = np.ceil(len(kin_score_df.columns) * perc / 100)
    num_syn_emg = np.ceil(len(emg_score_df.columns) * perc / 100)
    num_syn_tact = np.ceil(len(tact_score_df.columns) * perc / 100)

    # SELECT SYNERGIES
    if discard == 'less':
        kin_scores = pd.concat([kin_score_df.iloc[:, :int(num_syn_kin)], extra_data], axis=1, ignore_index=True)
        kin_scores.columns = list(kin_score_df.columns[:int(num_syn_kin)]) + list(extra_data.columns)
        emg_scores = pd.concat([emg_score_df.iloc[:, :int(num_syn_emg)], extra_data], axis=1, ignore_index=True)
        emg_scores.columns = list(emg_score_df.columns[:int(num_syn_emg)]) + list(extra_data.columns)
        tact_scores = pd.concat([tact_score_df.iloc[:, :int(num_syn_tact)], extra_data], axis=1, ignore_index=True)
        tact_scores.columns = list(tact_score_df.columns[:int(num_syn_tact)]) + list(extra_data.columns)
    else:
        kin_scores = pd.concat([kin_score_df.iloc[:, -int(num_syn_kin):], extra_data], axis=1, ignore_index=True)
        kin_scores.columns = list(kin_score_df.columns[-int(num_syn_kin):]) + list(extra_data.columns)
        emg_scores = pd.concat([emg_score_df.iloc[:, -int(num_syn_emg):], extra_data], axis=1, ignore_index=True)
        emg_scores.columns = list(emg_score_df.columns[-int(num_syn_emg):]) + list(extra_data.columns)
        tact_scores = pd.concat([tact_score_df.iloc[:, -int(num_syn_tact):], extra_data], axis=1, ignore_index=True)
        tact_scores.columns = list(tact_score_df.columns[:int(num_syn_tact)]) + list(extra_data.columns)

    total_score = []

    # kin_dat = kin_scores.loc[kin_scores['Family'] == family]
    # emg_dat = emg_scores.loc[emg_scores['Family'] == family]
    # tact_dat = tact_scores.loc[tact_scores['Family'] == family]

    kin_dat = kin_scores
    emg_dat = emg_scores
    tact_dat = tact_scores

    # to_kfold = kin_dat.drop_duplicates(subset=['Trial num', 'Given Object'])
    to_kfold = kin_dat.drop_duplicates(subset=['Trial num', 'Family'])

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=rnd_st)
    # WARNING: the skf.split returns the indexes
    # for train, test in skf.split(to_kfold['Trial num'].astype(int),to_kfold['Given Object'].astype(str)):
    for train, test in skf.split(to_kfold['Trial num'].astype(int), to_kfold['Family'].astype(str)):

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
            kin_in_bins = np.array_split(ep_kin_data.drop(columns=extra_data.columns, axis=1),
                                         kin_bins)

            ep_emg_data = emg_dat.loc[emg_dat['Trial num'] == trn_iter]
            emg_in_bins = np.array_split(ep_emg_data.drop(columns=extra_data.columns, axis=1),
                                         emg_bins)

            ep_tact_data = tact_dat.loc[tact_dat['Trial num'] == trn_iter]
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

                    kin_train_data.append(flat_kin_mean)
                    emg_train_data.append(flat_emg_mean)
                    tact_train_data.append(flat_tact_mean)
                    # train_labels.append(np.unique(ep_kin_data['Given Object'])[0])
                    train_labels.append(np.unique(ep_kin_data['Family'])[0])

                except RuntimeWarning:
                    # print("Dropped EP", trn_iter, "from family ", family)
                    trn_dropped += 1

        train_kin_df = pd.DataFrame(kin_train_data)
        train_emg_df = pd.DataFrame(emg_train_data)
        train_tact_df = pd.DataFrame(tact_train_data)
        train_df = pd.concat([train_kin_df, train_emg_df, train_tact_df], axis=1)
        train_df.apply(zscore)

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
                    # test_labels.append(np.unique(ep_kin_data['Given Object'])[0])
                    test_labels.append(np.unique(ep_kin_data['Family'])[0])

                except RuntimeWarning:
                    # print("Dropped EP", tst_iter, "from family ", family)
                    tst_dropped += 1

        test_kin_df = pd.DataFrame(kin_test_data)
        test_emg_df = pd.DataFrame(emg_test_data)
        test_tact_df = pd.DataFrame(tact_test_data)
        test_df = pd.concat([test_kin_df, test_emg_df, test_tact_df], axis=1)
        test_df.apply(zscore)

        log_model = LogisticRegression(penalty='elasticnet', C=c_param, class_weight='balanced',
                                       random_state=rnd_st, solver='saga', max_iter=50000,
                                       multi_class='multinomial',
                                       n_jobs=-1, l1_ratio=l1_param)
        # train model
        log_model.fit(X=train_df, y=train_labels)
        sc = round(log_model.score(X=test_df, y=test_labels) * 100, 2)
        total_score.append(sc)

    res = ['Multimodal']
    res.extend([family, perc, l1_param, c_param])
    # need to add weights
    res.append(total_score)

    # print(res)
    return res


def multisource_syn_classification(type, discard):

    perc_syns = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
    families = ['Ball', 'Cutlery', 'Geometric', 'Mugs', 'Plates']
    l1VSl2 = [0, 0.25, 0.5, 0.75, 1]
    c_values = [0.01, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5]

    # TO TEST
    # l1VSl2 = 0.25
    # c_values = 1.5

    # SET AND OPEN RESULT FILE
    if type == 'all':
        res_file_name = './results/Syn/accuracy/syn_multi_results'
    elif type == 'clustering':
        res_file_name = './results/Syn/accuracy/subj_clust_syn_multi_results'
    else:  # early enclosure
        res_file_name = './results/Early Enclosure/accuracy/alternative_syn_multi_results'

    if discard == 'less':
        res_file_name += '.csv'
    else:
        res_file_name += '_decr.csv'

    result_file = open(res_file_name, 'a')
    wr = csv.writer(result_file)

    random_states = [42, 43, 44]

    # we need to build the object to be iterated in the multiprocessing pool
    all_param = list(itertools.product(families, l1VSl2, c_values, perc_syns, random_states))
    data_and_iter = [[x, type, discard] for x in all_param]

    # TO TEST
    # all_param = list(itertools.product(families, perc_syns, random_states))
    # data_and_iter = [[[x[0], l1VSl2, c_values, x[1], x[2]], type, discard] for x in all_param]

    # multiprocessing
    with Pool() as pool:

        result = pool.map_async(multi_aux_classification, data_and_iter)

        for res in result.get():

            wr.writerow([res[0], res[1], res[2], res[3], res[4], res[5][0]])
            wr.writerow([res[0], res[1], res[2], res[3], res[4], res[5][1]])
            wr.writerow([res[0], res[1], res[2], res[3], res[4], res[5][2]])

            # print(res)

    result_file.close()
    # print("MULTIMODAL DONE !!!")
