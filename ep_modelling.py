import numpy as np
import os
import itertools
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression, SGDClassifier
from collections import Counter
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
from sklearn.preprocessing import OneHotEncoder
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
from itertools import combinations
from statsmodels.multivariate.manova import MANOVA
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import mannwhitneyu
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.distance import cdist
from numpy import std, mean, sqrt
import ast
from scipy.stats import f_oneway
from scipy.spatial import procrustes

from classification import get_raw_best_params
from load_subject import load
from synergy_pipeline import kin_syn_extraction
from split_data import split
from synergy_pipeline import silhouette_scores_custom

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def ep_from_scores_classif(include_suj):
    """
    This function is used as an interface to build SGD classifiers
    They are based on the EPs bins from synergy scores
    The classifier targets the EP label
    """

    discard = 'least'

    cv = 3
    [kin_params, emg_params, tact_params] = get_raw_best_params()
    kin_bins = kin_params[0]
    # emg_bins = emg_params[0]
    # tact_bins = tact_params[0]
    # perc_syns = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
    # families = ['Ball', 'Cutlery', 'Geometric', 'Mugs', 'Plates']
    l1VSl2 = [0, 0.25, 0.5, 0.75, 1]
    # c_param = [0.01, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5]
    # c_param = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5]
    c_param = [0.01, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5]

    # LOAD EXTRA DATA
    extra_data = pd.read_csv('./results/Syn/extra_data.csv')

    # SET AND OPEN RESULT FILE
    if include_suj:
        res_file_name = './results/Syn/accuracy/ep_alternative_syn_suj_results.csv'
    else:
        # res_file_name = './results/Syn/accuracy/ep_alternative_syn_results.csv'
        res_file_name = './results/Syn/accuracy/ep_all_suj_syn_results.csv'

    result_file = open(res_file_name, 'a')
    wr = csv.writer(result_file)

    # GET SCORES
    # sources = ['kin', 'emg_pca', 'tact']
    # sources = ['kin']
    source = 'kin'

    kin_score_df = pd.DataFrame()
    # emg_score_df = pd.DataFrame()
    # tact_score_df = pd.DataFrame()

    # score_files = glob.glob('./results/Syn/scores/sub*' + source + '_scores.csv')
    # score_files = glob.glob('./results/Syn/scores/reordered_alternative_' + source + '_scores.csv')
    score_files = glob.glob('./results/Syn/scores/' + source + '_scores.csv')
    score_files.sort()

    for iter_file in range(0, len(score_files)):

        subj_dat = pd.read_csv(score_files[iter_file])
        subj_dat.drop(subj_dat.columns[0], axis=1, inplace=True)

        if source == 'kin':
            kin_score_df = pd.concat([kin_score_df, subj_dat])
        # elif source == 'emg_pca':
        #     emg_score_df = pd.concat([emg_score_df, subj_dat])
        # else:
        #     tact_score_df = pd.concat([tact_score_df, subj_dat])

    # BUILD ITERABLE STRUCTURES
    # all_param = list(itertools.product(perc_syns, families, l1VSl2, c_param))
    all_param = list(itertools.product(l1VSl2, c_param))
    kin_data_and_iter = [[kin_score_df, extra_data, x, cv, kin_bins, discard, include_suj] for x in all_param]
    # emg_pca_data_and_iter = [[emg_score_df, extra_data, x, cv, emg_bins, discard] for x in all_param]
    # tact_data_and_iter = [[tact_score_df, extra_data, x, cv, tact_bins, discard] for x in all_param]

    # multiprocessing
    with Pool() as pool:

        # result_kin = pool.map_async(kin_ep_classif, kin_data_and_iter)
        result_kin = pool.map_async(kin_ep_classif_sgdb, kin_data_and_iter)

        # result_emg_pca = pool.map_async(emg_pca_syn_classif, emg_pca_data_and_iter)
        # result_tact = pool.map_async(tact_syn_classif, tact_data_and_iter)

        for res_kin in result_kin.get():
            wr.writerow(res_kin)
        # print("Kinematic classification done!")

        # for res_emg_pca in result_emg_pca.get():
        #     wr.writerow(res_emg_pca)
        # # print("EMG PCA classification done!")
        #
        # for res_tact in result_tact.get():
        #     wr.writerow(res_tact)
        # # print("Tactile classification done!")

    # print("Single source classification done!!")
    result_file.close()


def kin_ep_classif(input_data):
    """
    THIS FUNCTION IS DEPRECATED
    """
    kin_scores = input_data[0]
    extra_data = input_data[1]
    # perc_syns = input_data[2][0]
    perc_syns = 100
    # family = input_data[2][1]
    l1VSl2 = input_data[2][0]
    c_param = input_data[2][1]
    cv = input_data[3]
    kin_bins = input_data[4]

    discard = input_data[5]

    include_suj = input_data[6]

    total_score = []

    num_syns = np.ceil(len(kin_scores.columns) * perc_syns / 100)
    extra_data.reset_index(inplace=True, drop=True)
    kin_scores.reset_index(inplace=True, drop=True)

    if discard == 'least':
        data_df = pd.concat([kin_scores.iloc[:, 0:int(num_syns)], extra_data], axis=1)  # keeps most relevant
    else:
        data_df = pd.concat([kin_scores.iloc[:, -int(num_syns):], extra_data], axis=1)  # discards most relevant

    # selected_df = data_df.loc[data_df['Family'] == family]
    to_kfold = data_df.drop_duplicates(subset=['EP total', 'EP'])

    if include_suj:
        encoder = OneHotEncoder(sparse_output=False)
        encoded_suj = encoder.fit(to_kfold[['Subject']])

    random_states = [42, 43, 44]
    for rnd_st in random_states:

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=rnd_st)
        # WARNING: the skf.split returns the indexes
        for train, test in skf.split(to_kfold['EP total'].astype(int), to_kfold['EP'].astype(str)):

            train_trials = to_kfold.iloc[train]['EP total']  # because skf.split returns the indexes
            test_trials = to_kfold.iloc[test]['EP total']  # because skf.split returns the indexes

            train_data = []
            train_labels = []

            dropped = 0  # Number of dropped EPs

            # take each ep, create bins & compute mean
            for trn_iter in train_trials:

                train_tri = data_df.loc[data_df['EP total'] == trn_iter]
                tr_kin_data = train_tri.drop(columns=extra_data.columns)
                tr_in_bins = np.array_split(tr_kin_data, kin_bins)

                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    try:
                        tr_bin_mean = [np.nanmean(x, axis=0) for x in tr_in_bins]  # size = [num_bins] X [64]
                        flat_tr_mean = list(
                            itertools.chain.from_iterable(tr_bin_mean))  # size = [num_bins X 64] (unidimensional)
                        if include_suj:
                            transformed_sample = encoder.transform(train_tri[['Subject']])[0]
                            flat_tr_mean.extend(transformed_sample)
                        train_data.append(flat_tr_mean)
                        train_labels.append(np.unique(train_tri['EP'])[0])
                    except RuntimeWarning:
                        # print("Dropped EP", trn_iter, "from family ", family)
                        dropped += 1

            test_data = []
            test_labels = []

            for tst_iter in test_trials:

                test_tri = data_df.loc[data_df['EP total'] == tst_iter]
                tst_kin_data = test_tri.drop(columns=extra_data.columns)
                tst_in_bins = np.array_split(tst_kin_data, kin_bins)

                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    try:
                        tst_bin_mean = [np.nanmean(x, axis=0) for x in tst_in_bins]  # size = [num_bins] X [64]
                        flat_tst_mean = list(
                            itertools.chain.from_iterable(tst_bin_mean))  # size = [num_bins X 64] (unidimensional)
                        if include_suj:
                            transformed_sample = encoder.transform(test_tri[['Subject']])[0]
                            flat_tst_mean.extend(transformed_sample)
                        test_data.append(flat_tst_mean)
                        test_labels.append(np.unique(test_tri['EP'])[0])
                    except RuntimeWarning:
                        # print("Dropped EP", tst_iter, "from family ", family)
                        dropped += 1

            # build model
            log_model = LogisticRegression(penalty='elasticnet', C=c_param, class_weight='balanced',
                                           random_state=rnd_st,
                                           solver='saga', max_iter=100000, multi_class='ovr', n_jobs=-1,
                                           l1_ratio=l1VSl2)
            # train model
            trn_weights = compute_sample_weight(class_weight='balanced', y=train_labels)
            log_model.fit(X=train_data, y=train_labels, sample_weight=trn_weights)
            # tst_weights = compute_sample_weight(class_weight='balanced', y=test_labels)
            # sc = round(log_model.score(X=test_data, y=test_labels, sample_weight=tst_weights) * 100, 2)
            sc = round(log_model.score(X=test_data, y=test_labels) * 100, 2)
            total_score.append(sc)

    result = ['Kin']
    result.extend(input_data[2])
    result.append(total_score)
    result.append(round(np.mean(total_score), 2))
    # print("RESULT:", result)

    return result


def kin_ep_classif_sgdb(input_data):
    """
    This function is used to build SGD classifiers
    They are based on the EPs bins from raw data or synergies
    The classifier targets the EP label
    """

    kin_scores = input_data[0]
    extra_data = input_data[1]
    # perc_syns = input_data[2][0]
    perc_syns = 100
    # family = input_data[2][1]
    l1VSl2 = input_data[2][0]
    c_param = input_data[2][1]
    cv = input_data[3]
    kin_bins = input_data[4]

    discard = input_data[5]

    include_suj = input_data[6]

    total_score = []

    num_syns = np.ceil(len(kin_scores.columns) * perc_syns / 100)
    extra_data.reset_index(inplace=True, drop=True)
    kin_scores.reset_index(inplace=True, drop=True)

    if discard == 'least':
        data_df = pd.concat([kin_scores.iloc[:, 0:int(num_syns)], extra_data], axis=1)  # keeps most relevant
    else:
        data_df = pd.concat([kin_scores.iloc[:, -int(num_syns):], extra_data], axis=1)  # discards most relevant

    # selected_df = data_df.loc[data_df['Family'] == family]
    to_kfold = data_df.drop_duplicates(subset=['EP total', 'EP'])

    if include_suj:
        encoder = OneHotEncoder(sparse_output=False)
        encoded_suj = encoder.fit(to_kfold[['Subject']])

    random_states = [42, 43, 44]
    for rnd_st in random_states:

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=rnd_st)
        # WARNING: the skf.split returns the indexes
        for train, test in skf.split(to_kfold['EP total'].astype(int), to_kfold['EP'].astype(str)):

            train_trials = to_kfold.iloc[train]['EP total']  # because skf.split returns the indexes
            test_trials = to_kfold.iloc[test]['EP total']  # because skf.split returns the indexes

            train_data = []
            train_labels = []

            dropped = 0  # Number of dropped EPs

            # take each ep, create bins & compute mean
            for trn_iter in train_trials:

                train_tri = data_df.loc[data_df['EP total'] == trn_iter]
                tr_kin_data = train_tri.drop(columns=extra_data.columns)
                tr_in_bins = np.array_split(tr_kin_data, kin_bins)

                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    try:
                        tr_bin_mean = [np.nanmean(x, axis=0) for x in tr_in_bins]  # size = [num_bins] X [64]
                        flat_tr_mean = list(
                            itertools.chain.from_iterable(tr_bin_mean))  # size = [num_bins X 64] (unidimensional)
                        if include_suj:
                            transformed_sample = encoder.transform(train_tri[['Subject']])[0]
                            flat_tr_mean.extend(transformed_sample)
                        train_data.append(flat_tr_mean)
                        train_labels.append(np.unique(train_tri['EP'])[0])
                    except RuntimeWarning:
                        # print("Dropped EP", trn_iter, "from family ", family)
                        dropped += 1

            test_data = []
            test_labels = []

            for tst_iter in test_trials:

                test_tri = data_df.loc[data_df['EP total'] == tst_iter]
                tst_kin_data = test_tri.drop(columns=extra_data.columns)
                tst_in_bins = np.array_split(tst_kin_data, kin_bins)

                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    try:
                        tst_bin_mean = [np.nanmean(x, axis=0) for x in tst_in_bins]  # size = [num_bins] X [64]
                        flat_tst_mean = list(
                            itertools.chain.from_iterable(tst_bin_mean))  # size = [num_bins X 64] (unidimensional)
                        if include_suj:
                            transformed_sample = encoder.transform(test_tri[['Subject']])[0]
                            flat_tst_mean.extend(transformed_sample)
                        test_data.append(flat_tst_mean)
                        test_labels.append(np.unique(test_tri['EP'])[0])
                    except RuntimeWarning:
                        # print("Dropped EP", tst_iter, "from family ", family)
                        dropped += 1

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


def ep_from_raw_classif(df, include_suj):
    """
    This function is used as an interface to build SGD classifiers
    They are based on the EPs bins from raw data
    The classifier targets the EP label
    """

    discard = 'least'

    kin_cols = ['ThumbRotate', 'ThumbMPJ', 'ThumbIj', 'ThumbAb', 'IndexMPJ', 'IndexPIJ',
                'MiddleMPJ', 'MiddlePIJ', 'MiddleIndexAb', 'RingMPJ', 'RingPIJ',
                'RingMiddleAb', 'PinkieMPJ', 'PinkiePIJ', 'PinkieRingAb', 'PalmArch',
                'WristPitch', 'WristYaw']

    extra_cols = ['Task', 'EP', 'Subject', 'Trial num', 'EP num', 'EP total',
       'Given Object', 'Asked Object', 'Family']

    kin_df = df[kin_cols]
    extra_data = df[extra_cols]

    cv = 3
    [kin_params, emg_params, tact_params] = get_raw_best_params()
    kin_bins = kin_params[0]
    # emg_bins = emg_params[0]
    # tact_bins = tact_params[0]
    perc_syns = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
    families = ['Ball', 'Cutlery', 'Geometric', 'Mugs', 'Plates']
    l1VSl2 = [0, 0.25, 0.5, 0.75, 1]
    # c_param = [0.01, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5]
    c_param = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5]

    # SET AND OPEN RESULT FILE
    if include_suj:
        res_file_name = './results/Raw/accuracy/ep_alternative_raw_suj_results.csv'
    else:
        res_file_name = './results/Raw/accuracy/ep_alternative_raw_results.csv'

    result_file = open(res_file_name, 'a')
    wr = csv.writer(result_file)

    # BUILD ITERABLE STRUCTURES
    all_param = list(itertools.product(l1VSl2, c_param))
    kin_data_and_iter = [[kin_df, extra_data, x, cv, kin_bins, discard, include_suj] for x in all_param]

    with Pool() as pool:

        # result_kin = pool.map_async(kin_ep_classif, kin_data_and_iter)
        result_kin = pool.map_async(kin_ep_classif_sgdb, kin_data_and_iter)

        for res_kin in result_kin.get():
            wr.writerow(res_kin)
        # print("Kinematic classification done!")

        # for res_emg_pca in result_emg_pca.get():
        #     wr.writerow(res_emg_pca)
        # # print("EMG PCA classification done!")
        #
        # for res_tact in result_tact.get():
        #     wr.writerow(res_tact)
        # # print("Tactile classification done!")

    # print("Single source classification done!!")
    result_file.close()


def ep_classification_plots(type):
    """
    This function generates the plots associated to the classifiers targeting EPs
    It is build to compare between paradigms
    THIS FUNCTION IS DEPRECATED
    """

    plt.close('all')  # to clean the screen

    # ['syn', 'raw', 'syn_raw_suj', 'syn_raw_no_suj]
    if type == 'syn':
        # a_file = './results/Syn/accuracy/ep_alternative_syn_results.csv'
        a_file = './results/Syn/accuracy/ep_all_suj_syn_results.csv'
        b_file = './results/Syn/accuracy/ep_alternative_syn_suj_results.csv'
    elif type == 'raw':
        a_file = './results/Raw/accuracy/ep_alternative_raw_results.csv'
        b_file = './results/Raw/accuracy/ep_alternative_raw_suj_results.csv'
    elif type == 'syn_raw_suj':
        a_file = './results/Syn/accuracy/ep_alternative_syn_suj_results.csv'
        b_file = './results/Raw/accuracy/ep_alternative_raw_suj_results.csv'
    else: # syn_raw_no_suj
        # a_file = './results/Syn/accuracy/ep_alternative_syn_results.csv'
        a_file = './results/Syn/accuracy/ep_all_suj_syn_results.csv'
        b_file = './results/Raw/accuracy/ep_alternative_raw_results.csv'

    a_df = pd.read_csv(a_file, header=None)
    b_df = pd.read_csv(b_file, header=None)

    values_col = a_df.columns[-2]
    best_score_col = a_df.columns[-1]

    # Fix: Use 'a_df' consistently for 'a_best_values'
    a_best_values = a_df.loc[a_df[best_score_col].idxmax()][values_col]
    # Ensure the string is correctly formatted and convert it to a list
    a_floats_list = ast.literal_eval(a_best_values.strip("'"))

    # Fix: Properly create a DataFrame from the list
    a_best_values_df = pd.DataFrame({'Group 1': a_floats_list})

    # Similar corrections for 'b_df'
    b_best_values = b_df.loc[b_df[best_score_col].idxmax()][values_col]
    b_floats_list = ast.literal_eval(b_best_values.strip("'"))
    b_best_values_df = pd.DataFrame({'Group 2': b_floats_list})

    # Combining the two DataFrames side by side
    plot_data = pd.concat([a_best_values_df, b_best_values_df], axis=1)
    plot_data_melted = plot_data.melt(var_name='Group', value_name='Values')

    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x='Group', y='Values', data=plot_data_melted, width=0.5)
    _, p = mannwhitneyu(plot_data['Group 1'], plot_data['Group 2'])
    ax.text(0.5, 0.95, f'p={p:.3f}', fontsize=14, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    ax.set_ylim(0, 100)
    plt.axhline(12.5, color='r', linestyle='--')
    current_ticks = ax.get_yticks()
    updated_ticks = np.unique(np.append(current_ticks, 12.5))
    ax.set_yticks(updated_ticks)
    ax.set_xlabel('')
    ax.set_ylabel('Values', fontsize=14)  # Ajusta según sea necesario
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    if type == 'syn':
        ax.set_xticklabels(['Syn scores', 'Syn scores + Subject'])
        plt.title('Comparison of accuracies for classifiers targeting EP label\nSynergy scores and syn scores + subject')
        # save_file = './results/ep_comp_syn.png'
        save_file = './results/ep_comp_syn.svg'
    elif type == 'raw':
        ax.set_xticklabels(['Raw data', 'Raw data + Subject'])
        plt.title('Comparison of accuracies for classifiers targeting EP label\nRaw data and raw data + subject')
        # save_file = './results/ep_comp_raw.png'
        save_file = './results/ep_comp_raw.svg'
    elif type == 'syn_raw_suj':
        ax.set_xticklabels(['Syn scores + Subject', 'Raw Data + Subject'])
        plt.title('Comparison of accuracies for classifiers targeting EP label\nSyn scores and Raw data, both including subject')
        # save_file = './results/ep_comp_syn_raw_suj.png'
        save_file = './results/ep_comp_syn_raw_suj.svg'
    else:  # syn_raw_no_suj
        ax.set_xticklabels(['Syn scores', 'Raw Data'])
        plt.title('Comparison of accuracies for classifiers targeting EP label\nSyn scores and Raw data without subject')
        # save_file = './results/ep_comp_syn_raw_no_suj.png'
        save_file = './results/ep_comp_syn_raw_no_suj.svg'

    # plt.show()
    plt.savefig(save_file, format='svg', dpi=600)


def ep_clust_suj_syn_one_subject_out():
    """
    THIS FUNCTION IS DEPRECATED
    """

    clusters = [
        {'sub-01', 'sub-03', 'sub-05', 'sub-07', 'sub-08'},
        {'sub-09', 'sub-10'},
        {'sub-02', 'sub-04', 'sub-06', 'sub-11'}
    ]
    # Defining clusters

    [kin_params, emg_params, tact_params] = get_raw_best_params()
    kin_bins = kin_params[0]
    l1VSl2 = [0, 0.25, 0.5, 0.75, 1]
    c_param = [0.01, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5]

    res_file_name = './results/Syn/accuracy/ep_clust_suj_leave_one_syn_results.csv'
    result_file = open(res_file_name, 'a')
    wr = csv.writer(result_file)

    data_folder = '/BIDSData'
    subject_folders = sorted([f.name for f in os.scandir(os.getcwd() + data_folder) if f.is_dir()])

    # Iterate over each cluster
    for cluster in clusters:
        cluster_subjects = [subj for subj in subject_folders if subj in cluster]
        for target_subject in cluster_subjects:

            new_subjects = [x for x in cluster_subjects if x != target_subject]
            to_pca = pd.DataFrame()

            for n_subject in new_subjects:
                n_subject_data = load(n_subject)
                to_pca = pd.concat([to_pca, n_subject_data], ignore_index=True)

            data_clean = to_pca.dropna(axis=0, how='any')
            kin_cols = ['ThumbRotate', 'ThumbMPJ', 'ThumbIj', 'ThumbAb', 'IndexMPJ', 'IndexPIJ',
                        'MiddleMPJ', 'MiddlePIJ', 'MiddleIndexAb', 'RingMPJ', 'RingPIJ',
                        'RingMiddleAb', 'PinkieMPJ', 'PinkiePIJ', 'PinkieRingAb', 'PalmArch',
                        'WristPitch', 'WristYaw']

            train_split_df = split(data_clean)
            train_split_df['Trial num'] = train_split_df['Trial num'].astype('str')
            train_split_df['EP num'] = train_split_df['EP num'].astype('str')
            to_remove = [x for x in train_split_df['EP'].unique() if '+' in x]
            train_split_df = train_split_df[~train_split_df['EP'].isin(to_remove)]
            train_split_df.loc[train_split_df['EP'] == 'contour following', 'EP'] = 'edge following'

            [kin_scores, kin_syns, kin_var, kin_var_tot, kin_mean, kin_scale] = kin_syn_extraction(
                train_split_df[kin_cols])

            target_subject_data = load(target_subject)
            target_split_df = split(target_subject_data)
            target_split_df['Trial num'] = target_split_df['Trial num'].astype('str')
            target_split_df['EP num'] = target_split_df['EP num'].astype('str')
            to_remove = [x for x in target_split_df['EP'].unique() if '+' in x]
            target_split_df = target_split_df[~target_split_df['EP'].isin(to_remove)]
            target_split_df.loc[target_split_df['EP'] == 'contour following', 'EP'] = 'edge following'

            target_scaled = (target_split_df[kin_cols] - kin_mean) / kin_scale
            target_transformed = np.dot(target_scaled, kin_syns)

            extra_cols = [col for col in data_clean.columns if col not in kin_cols]
            train_extra_data = train_split_df[extra_cols]
            target_extra_data = target_split_df[extra_cols]

            all_param = list(itertools.product(l1VSl2, c_param))
            kin_data_and_iter = [[kin_scores, target_transformed, train_extra_data, target_extra_data, x, kin_bins] for
                                 x in all_param]

            with Pool() as pool:
                result_kin = pool.map_async(kin_ep_classif_sgdb_subject, kin_data_and_iter)
                for res_kin in result_kin.get():
                    wr.writerow(res_kin)

    result_file.close()


def ep_all_suj_syn_one_subject_out():
    """
    This function is used as an interface to build SGD classifiers using one-subject-out paradigm
    They are based on the EPs bins from raw data
    The classifier targets the EP label
    """

    [kin_params, emg_params, tact_params] = get_raw_best_params()
    kin_bins = kin_params[0]
    # emg_bins = emg_params[0]
    # tact_bins = tact_params[0]
    # perc_syns = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
    # families = ['Ball', 'Cutlery', 'Geometric', 'Mugs', 'Plates']
    l1VSl2 = [0, 0.25, 0.5, 0.75, 1]
    # c_param = [0.01, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5]
    # c_param = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5]
    c_param = [0.01, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5]

    # LOAD EXTRA DATA
    # extra_data = pd.read_csv('./results/Syn/extra_data.csv')

    res_file_name = './results/Syn/accuracy/ep_all_suj_leave_one_syn_results.csv'

    result_file = open(res_file_name, 'a')
    wr = csv.writer(result_file)

    data_folder = '/BIDSData'
    subject_folders = sorted([f.name for f in os.scandir(os.getcwd() + data_folder) if f.is_dir()])

    # for subjects: pca and pool over parameters
    for target_subject in subject_folders:

        new_subjects = [x for x in subject_folders if x != target_subject]

        to_pca = pd.DataFrame()

        for n_subject in new_subjects:

            n_subject_data = load(n_subject)
            to_pca = pd.concat([to_pca, n_subject_data], ignore_index=True)

        # REMOVE NANs
        data_clean = to_pca.dropna(axis=0, how='any')

        kin_cols = ['ThumbRotate', 'ThumbMPJ', 'ThumbIj', 'ThumbAb', 'IndexMPJ', 'IndexPIJ',
                    'MiddleMPJ', 'MiddlePIJ', 'MiddleIndexAb', 'RingMPJ', 'RingPIJ',
                    'RingMiddleAb', 'PinkieMPJ', 'PinkiePIJ', 'PinkieRingAb', 'PalmArch',
                    'WristPitch', 'WristYaw']

        # emg_cols = [col for col in data_clean.columns if ('flexion' in col) or ('extension' in col)]
        # tact_cols = ['rmo', 'mdo', 'rmi', 'mmo', 'pcim', 'ldd', 'rmm', 'rp', 'rdd', 'lmi', 'rdo', 'lmm', 'lp', 'rdm',
        #              'ldm', 'ptip', 'idi', 'mdi', 'ido', 'mmm', 'ipi', 'mdm', 'idd', 'idm', 'imo', 'pdi', 'mmi', 'pdm',
        #              'imm', 'mdd', 'pdii', 'mp', 'ptod', 'ptmd', 'tdo', 'pcid', 'imi', 'tmm', 'tdi', 'tmi', 'ptop',
        #              'ptid', 'ptmp', 'tdm', 'tdd', 'tmo', 'pcip', 'ip', 'pcmp', 'rdi', 'ldi', 'lmo', 'pcmd', 'ldo',
        #              'pdl', 'pdr', 'pdlo', 'lpo']

        train_split_df = split(data_clean)
        train_split_df['Trial num'] = train_split_df['Trial num'].astype('str')
        train_split_df['EP num'] = train_split_df['EP num'].astype('str')
        to_remove = [x for x in train_split_df['EP'].unique() if '+' in x]
        train_split_df = train_split_df[~train_split_df['EP'].isin(to_remove)]
        train_split_df.loc[train_split_df['EP'] == 'contour following', 'EP'] = 'edge following'

        [kin_scores, kin_syns, kin_var, kin_var_tot, kin_mean, kin_scale] = kin_syn_extraction(train_split_df[kin_cols])

        target_subject_data = load(target_subject)
        target_split_df = split(target_subject_data)
        target_split_df['Trial num'] = target_split_df['Trial num'].astype('str')
        target_split_df['EP num'] = target_split_df['EP num'].astype('str')
        to_remove = [x for x in target_split_df['EP'].unique() if '+' in x]
        target_split_df = target_split_df[~target_split_df['EP'].isin(to_remove)]
        target_split_df.loc[target_split_df['EP'] == 'contour following', 'EP'] = 'edge following'

        target_scaled = (target_split_df[kin_cols] - kin_mean) / kin_scale

        # Project the new data onto the PCA components (kin_syns)
        target_transformed = np.dot(target_scaled, kin_syns)

        # # NON-NUMERIC DATA EXTRACTION & SAVING
        # extra_cols = [col for col in data_clean.columns if
        #               (col not in kin_cols) and (col not in emg_cols) and (col not in tact_cols)]
        extra_cols = [col for col in data_clean.columns if col not in kin_cols]

        train_extra_data = train_split_df[extra_cols]
        target_extra_data = target_split_df[extra_cols]

        all_param = list(itertools.product(l1VSl2, c_param))
        kin_data_and_iter = [[kin_scores, target_transformed, train_extra_data, target_extra_data, x, kin_bins] for x in all_param]

        with Pool() as pool:

            result_kin = pool.map_async(kin_ep_classif_sgdb_subject, kin_data_and_iter)

            for res_kin in result_kin.get():
                wr.writerow(res_kin)

    result_file.close()

def kin_ep_classif_sgdb_subject(input_data):
    """
    This function is used to build SGD classifiers
    They are based  on the EPs bins from raw data
    The classifier targets the EP label
    """

    train_scores = input_data[0]
    target_scores = input_data[1]

    train_extra_data = input_data[2]
    target_extra_data = input_data[3]

    l1VSl2 = input_data[4][0]
    c_param = input_data[4][1]
    kin_bins = input_data[5]

    train_data = pd.concat([pd.DataFrame(train_scores), train_extra_data], axis=1)
    target_data = pd.concat([pd.DataFrame(target_scores), target_extra_data], axis=1)

    train_trials = pd.unique(train_data['EP total'])
    target_trials = pd.unique(target_data['EP total'])

    dropped = 0

    tr_data = []
    tr_labels = []
    tst_data = []
    tst_labels = []

    for train_tr in train_trials:

        train_tri = train_data.loc[train_data['EP total'] == train_tr]
        tr_kin_data = train_tri.drop(columns=train_extra_data.columns)
        tr_in_bins = np.array_split(tr_kin_data, kin_bins)

        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                tr_bin_mean = [np.nanmean(x, axis=0) for x in tr_in_bins]  # size = [num_bins] X [64]
                flat_tr_mean = list(
                    itertools.chain.from_iterable(tr_bin_mean))  # size = [num_bins X 64] (unidimensional)
                tr_data.append(flat_tr_mean)
                tr_labels.append(np.unique(train_tri['EP'])[0])
            except RuntimeWarning:
                # print("Dropped EP", trn_iter, "from family ", family)
                dropped += 1

    for test_tr in target_trials:

        test_tri = target_data.loc[target_data['EP total'] == test_tr]
        tst_kin_data = test_tri.drop(columns=target_extra_data.columns)
        tst_in_bins = np.array_split(tst_kin_data, kin_bins)

        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                tst_bin_mean = [np.nanmean(x, axis=0) for x in tst_in_bins]  # size = [num_bins] X [64]
                flat_tst_mean = list(
                    itertools.chain.from_iterable(tst_bin_mean))  # size = [num_bins X 64] (unidimensional)
                tst_data.append(flat_tst_mean)
                tst_labels.append(np.unique(test_tri['EP'])[0])
            except RuntimeWarning:
                # print("Dropped EP", trn_iter, "from family ", family)
                dropped += 1

    alpha_param = 1 / c_param  # Alpha is the inverse of regularization strength (C)
    batch_size = 50  # Define your batch size here

    # Create the SGDClassifier model with logistic regression
    sgd_model = SGDClassifier(loss='log_loss', penalty='elasticnet', alpha=alpha_param, l1_ratio=l1VSl2,
                              max_iter=10000, warm_start=True, learning_rate='optimal',
                              eta0=0.01)

    # Compute the sample weights for the entire dataset
    classes = np.unique(tr_labels)
    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=classes, y=tr_labels)
    # Convert class weights to dictionary format
    class_weight_dict = {classes[i]: class_weights[i] for i in range(len(classes))}

    # Mini-batch training
    for _ in range(100000 // batch_size):  # Assuming 100000 iterations as max, adjust as needed
        # Randomly sample a batch of data
        batch_indices = np.random.choice(range(len(tr_data)), size=batch_size, replace=False)
        batch_indices_list = batch_indices.tolist()

        batch_data = [tr_data[i] for i in batch_indices_list]
        batch_labels = [tr_labels[i] for i in batch_indices_list]
        # batch_weights = [trn_weights[i] for i in batch_indices_list]
        batch_weights = np.array([class_weight_dict[label] for label in batch_labels])

        # Partial fit on the batch
        sgd_model.partial_fit(batch_data, batch_labels, classes=classes, sample_weight=batch_weights)

    # Evaluate the model
    sc = round(sgd_model.score(X=tst_data, y=tst_labels) * 100, 2)

    result = ['Kin']
    result.extend(input_data[4])
    result.extend(pd.unique(target_extra_data['Subject']))
    result.append(sc)
    # print("RESULT:", result)

    return result


def ep_clust_suj_plots():
    """
    This function generates the plots associated to classifiers targeting EPs
    They compare the results between raw data, all subjects together,  leave-one-subject-out
    and leave-one-subject-out for each subject cluster
    (these clusters are generated externally to this code and hardcoded here)
    """

    plt.close('all')  # to clean the screen

    # File paths
    raw_file = './results/Raw/accuracy/ep_alternative_raw_results.csv'
    leave_one_out_file = './results/Syn/accuracy/ep_all_suj_leave_one_syn_results.csv'
    all_suj_file = './results/Syn/accuracy/ep_all_suj_syn_results.csv'
    leave_one_out_cluster_file = './results/Syn/accuracy/ep_clust_suj_leave_one_syn_results.csv'

    # Load data
    raw_data = pd.read_csv(raw_file, header=None)
    leave_one_out_data = pd.read_csv(leave_one_out_file, header=None)
    all_suj_data = pd.read_csv(all_suj_file, header=None)
    leave_one_out_cluster_data = pd.read_csv(leave_one_out_cluster_file, header=None)

    # Extract best hyperparameter set results
    raw_best_values = ast.literal_eval(raw_data.loc[raw_data.iloc[:, -1].idxmax(), raw_data.columns[-2]].strip("'"))
    all_suj_best_values = ast.literal_eval(all_suj_data.loc[all_suj_data.iloc[:, -1].idxmax(), all_suj_data.columns[-2]].strip("'"))
    leave_one_out_best_values = leave_one_out_data.groupby(3)[4].max().tolist()

    # Define clusters mapping
    clusters = {
        'sub-01': 'Cluster 0', 'sub-03': 'Cluster 0', 'sub-05': 'Cluster 0', 'sub-07': 'Cluster 0', 'sub-08': 'Cluster 0',
        'sub-09': 'Cluster 1', 'sub-10': 'Cluster 1',
        'sub-02': 'Cluster 2', 'sub-04': 'Cluster 2', 'sub-06': 'Cluster 2', 'sub-11': 'Cluster 2'
    }
    leave_one_out_cluster_data['Cluster'] = leave_one_out_cluster_data[3].map(clusters)

    # Extract best scores per subject within each cluster
    leave_one_out_cluster_data['Best Score'] = leave_one_out_cluster_data.groupby([3, 'Cluster'])[4].transform('max')
    best_cluster_values = leave_one_out_cluster_data.drop_duplicates(subset=[3, 'Cluster', 'Best Score'])

    # Flatten values into lists for each cluster
    cluster_values = best_cluster_values.groupby('Cluster')['Best Score'].apply(list)

    # Prepare dataframes for plotting
    raw_df = pd.DataFrame({'Group': 'Raw', 'Values': [raw_best_values]})
    all_suj_df = pd.DataFrame({'Group': 'All Subjects', 'Values': [all_suj_best_values]})
    leave_one_out_df = pd.DataFrame({'Group': 'Leave One Subject Out', 'Values': [leave_one_out_best_values]})
    cluster_dfs = pd.DataFrame({'Group': ['Cluster 0', 'Cluster 1', 'Cluster 2'],
                                'Values': cluster_values.reindex(['Cluster 0', 'Cluster 1', 'Cluster 2']).values})

    # Combine data for plotting
    plot_data = pd.concat([raw_df.explode('Values'), all_suj_df.explode('Values'), leave_one_out_df.explode('Values'), cluster_dfs.explode('Values')])

    # Ensure data is numeric
    plot_data['Values'] = pd.to_numeric(plot_data['Values'], errors='coerce')

    # Plotting
    plt.figure(figsize=(12, 8))
    ax = sns.violinplot(x='Group', y='Values', data=plot_data)
    add_stat_annotation(ax, data=plot_data, x='Group', y='Values',
                        box_pairs=[("Raw", "All Subjects"),
                                   ("Raw", "Leave One Subject Out"),
                                   ("All Subjects", "Leave One Subject Out"),
                                   ("Cluster 0", "Leave One Subject Out"),
                                   ("Cluster 1", "Leave One Subject Out"),
                                   ("Cluster 2", "Leave One Subject Out")],
                        test='Mann-Whitney', text_format='simple', loc='inside', verbose=2)
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels[0] = 'Raw\nData'
    labels[1] = 'All\nSubjects'
    labels[2] = 'Leave One\nSubject Out'
    ax.set_xticklabels(labels)
    ax.set_xlabel('')
    plt.ylim(0, 110)
    ax.tick_params(axis='both', labelsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.title('Comparison of accuracies for classifiers targeting EP label', fontsize=14)
    plt.axhline(12.5, color='r', linestyle='--', label='Chance Level')  # Add chance level line
    ax.legend(loc='lower left')
    plt.savefig('./results/ep_comp_syn_clust.svg', format='svg', dpi=600)
    # plt.show()


def ep_all_suj_plots():
    """
    This function generates the plots associated to classifiers targeting EPs
    They compare the results between raw data, all subjects together and leave-one-subject-out
    """

    plt.close('all')  # to clean the screen

    raw_file = './results/Raw/accuracy/ep_alternative_raw_results.csv'
    leave_one_out_file = './results/Syn/accuracy/ep_all_suj_leave_one_syn_results.csv'
    all_suj_file = './results/Syn/accuracy/ep_all_suj_syn_results.csv'

    raw_data = pd.read_csv(raw_file, header=None)
    leave_one_out_data = pd.read_csv(leave_one_out_file, header=None)
    all_suj_data = pd.read_csv(all_suj_file, header=None)

    raw_best_values = raw_data.loc[raw_data[raw_data.columns[-1]].idxmax()][raw_data.columns[-2]]
    raw_floats_list = ast.literal_eval(raw_best_values.strip("'"))

    leave_one_out_best_values = leave_one_out_data.groupby(3)[4].max().values

    all_suj_best_values = all_suj_data.loc[all_suj_data[all_suj_data.columns[-1]].idxmax()][all_suj_data.columns[-2]]
    all_suj_floats_list = ast.literal_eval(all_suj_best_values.strip("'"))

    raw_mean_value = np.asarray(raw_floats_list).mean()
    full_raw_values = np.pad(raw_floats_list, (0, len(leave_one_out_best_values) - len(raw_floats_list)), 'constant',
                         constant_values=(0, raw_mean_value))

    all_suj_mean_value = np.asarray(all_suj_floats_list).mean()
    full_all_suj_values = np.pad(all_suj_floats_list, (0, len(leave_one_out_best_values) - len(all_suj_floats_list)), 'constant',
                             constant_values=(0, all_suj_mean_value))

    raw_best_values_df = pd.DataFrame({'Raw': full_raw_values})
    all_suj_best_values_df = pd.DataFrame({'All Subj': full_all_suj_values})
    leave_one_out_best_values_df = pd.DataFrame({'Leave One Subject Out': leave_one_out_best_values})

    plot_data = pd.concat([raw_best_values_df, all_suj_best_values_df, leave_one_out_best_values_df], axis=1)
    plot_data_melted = plot_data.melt(var_name='Group', value_name='Values')

    plt.figure(figsize=(10, 6))
    ax = sns.violinplot(x='Group', y='Values', data=plot_data_melted, width=0.5)
    add_stat_annotation(ax, data=plot_data_melted, x='Group', y='Values',
                        box_pairs=[("Raw", "All Subj"), ("Raw", "Leave One Subject Out"), ("All Subj", "Leave One Subject Out")],
                        test='Mann-Whitney', text_format='simple', loc='inside', verbose=2)

    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels[0] = 'Raw\nData'
    labels[1] = 'All\nSubjects'
    labels[2] = 'Leave One\nSubject Out'
    ax.set_xticklabels(labels)

    ax.set_ylim(0, 100)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_xlabel('')
    plt.axhline(12.5, color='r', linestyle='--')
    plt.title('Comparison of accuracies for classifiers targeting EP label', fontsize=14)
    # save_file = './results/ep_comp_syn.png'
    save_file = './results/ep_comp_syn.svg'
    plt.savefig(save_file, format='svg', dpi=600)
    # plt.show()


def build_subject_clusters():
    """
    THIS FUNCTION IS DEPRECATED
    It was used to build subject clusters based on different distance methods
    """

    plt.close('all')  # to clean the screen

    number_clusters = [2, 3, 4, 5, 6, 7, 8]

    selected_files = glob.glob('./results/Syn/synergies/*kin_syns.csv')
    selected_files = [x for x in selected_files if 'sub' in x]  # selects only subject files
    selected_files.sort()

    all_data = pd.DataFrame()

    for file in selected_files:
        suj_dat = pd.read_csv(file)
        components = suj_dat.pop('Unnamed: 0').to_list()
        suj_dat = suj_dat.T
        suj_dat['Component'] = components

        all_data = pd.concat([all_data, pd.DataFrame(suj_dat)], ignore_index=True)

    syns = all_data.iloc[:, :-1].T  # corr() performs correlation between columns
    syns_corr = syns.corr().abs()
    distances = 1 - syns_corr.to_numpy()

    extra_data = pd.read_csv('./results/Syn/extra_data.csv')
    subjects = extra_data['Subject'].unique()

    num_subjects = len(subjects)
    synergies_per_subject = len(syns.index)

    mean_distances = np.zeros((num_subjects, num_subjects))

    """ENTIRE DISTANCE FROM SUBJECT TO SUBJECT"""
    # for i in range(num_subjects):
    #     for j in range(num_subjects):
    #
    #         if i == j:
    #             mean_distances[i, j] = 0
    #         else:
    #             start_i = i * synergies_per_subject
    #             end_i = start_i + synergies_per_subject
    #             start_j = j * synergies_per_subject
    #             end_j = start_j + synergies_per_subject
    #
    #             block = distances[start_i:end_i, start_j:end_j]
    #             mean_distances[i, j] = block.mean()

    """DISTANCE COMPONENT TO COMPONENT OF EACH SUBJECT"""
    # for i in range(num_subjects):
    #     for j in range(num_subjects):
    #         if i == j:
    #             mean_distances[i, j] = 0
    #         else:
    #             total_distance = 0
    #             for k in range(synergies_per_subject):
    #                 comp_i_index = i * synergies_per_subject + k
    #                 comp_j_index = j * synergies_per_subject + k
    #                 total_distance += distances[comp_i_index, comp_j_index]
    #
    #             mean_distances[i, j] = total_distance
    #
    # mean_distances /= synergies_per_subject

    """FIND MINIMAL DISTANCE BETWEEN COMPONENTS OF EACH SUBJECT"""
    for i in range(num_subjects):
        for j in range(num_subjects):
            if i == j:
                mean_distances[i, j] = 0
            else:
                min_distances = []
                for k in range(synergies_per_subject):
                    comp_i_index = i * synergies_per_subject + k
                    min_distance = np.min(
                        distances[comp_i_index, j * synergies_per_subject:(j + 1) * synergies_per_subject])
                    min_distances.append(min_distance)
                mean_distances[i, j] = np.mean(min_distances)

    """PROCRUSTES DISTANCE"""
    # for i in range(num_subjects):
    #     for j in range(num_subjects):
    #
    #         if i == j:
    #             mean_distances[i, j] = 0
    #         else:
    #             start_i = i * synergies_per_subject
    #             end_i = start_i + synergies_per_subject
    #             start_j = j * synergies_per_subject
    #             end_j = start_j + synergies_per_subject
    #
    #             suj_a = all_data.iloc[start_i:end_i, :-1]
    #             suj_b = all_data.iloc[start_j:end_j, :-1]
    #             _, _, dist = procrustes(suj_a, suj_b)
    #
    #             mean_distances[i, j] = dist

    silhouette_scores = []

    for n_clust in number_clusters:
        clustering_model = AgglomerativeClustering(metric='precomputed', linkage='average', n_clusters=n_clust)
        # clustering_model = AgglomerativeClustering(linkage='average', n_clusters=n_clust)
        cluster_labels = clustering_model.fit_predict(mean_distances)
        print(cluster_labels)
        silh_score = metrics.silhouette_score(mean_distances, cluster_labels)
        # silh_score = silhouette_scores_custom(mean_distances, cluster_labels)
        silhouette_scores.append(silh_score)

    plt.plot(number_clusters, silhouette_scores)
    plt.xticks(number_clusters, fontsize=14)  # Setting x-ticks to the exact values in number_clusters
    plt.yticks(np.linspace(0, 0.11, 12), ['{:.2f}'.format(x) for x in np.linspace(0, 0.11, 12)], fontsize=14)  # Setting y-ticks
    plt.ylim(0, 0.11)  # Setting the y-axis limits to 0 to 0.11
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores for subject clusters')
    # plt.show()
    # plt.savefig('./results/silhouette_subject_clusters.png', dpi=600)
    plt.savefig('./results/silhouette_subject_clusters.svg', format='svg', dpi=600)


def build_ep_clusters(data):
    """
    THIS FUNCTION IS DEPRECATED
    It was used to build EP clusters based on different distance methods
    """

    plt.close('all')  # to clean the screen

    kin_cols = ['ThumbRotate', 'ThumbMPJ', 'ThumbIj', 'ThumbAb', 'IndexMPJ', 'IndexPIJ',
                'MiddleMPJ', 'MiddlePIJ', 'MiddleIndexAb', 'RingMPJ', 'RingPIJ',
                'RingMiddleAb', 'PinkieMPJ', 'PinkiePIJ', 'PinkieRingAb', 'PalmArch',
                'WristPitch', 'WristYaw']

    number_clusters = [2, 3, 4, 5, 6]

    eps = data['EP'].unique()

    syn_df = pd.DataFrame()

    for ep in eps:

        selected_ep = data.loc[data['EP'] == ep]
        selected_data = selected_ep[kin_cols]
        clean_data = selected_data.apply(lambda col: col.fillna(col.mean()), axis=0)

        [kin_scores, kin_syns, kin_var, kin_var_tot, kin_mean, kin_scale] = kin_syn_extraction(clean_data) # Each column is a synergy

        ep_labels = np.repeat(ep, len(kin_syns))

        syns = pd.DataFrame(kin_syns.T)  # each row in syns is a synergy
        syns['EP'] = ep_labels

        syn_df = pd.concat([syn_df, syns])

    syns = syn_df.iloc[:, :-1].T  # corr() performs correlation between columns
    syns_corr = syns.corr().abs()
    distances = 1 - syns_corr.to_numpy()

    num_eps = len(eps)
    synergies_per_ep = len(kin_cols)

    mean_distances = np.zeros((num_eps, num_eps))

    """ENTIRE DISTANCE FROM EP TO EP"""
    # for i in range(num_eps):
    #     for j in range(num_eps):
    #
    #         if i == j:
    #             mean_distances[i, j] = 0
    #         else:
    #             start_i = i * synergies_per_ep
    #             end_i = start_i + synergies_per_ep
    #             start_j = j * synergies_per_ep
    #             end_j = start_j + synergies_per_ep
    #
    #             block = distances[start_i:end_i, start_j:end_j]
    #             mean_distances[i, j] = block.mean()

    """DISTANCE COMPONENT TO COMPONENT OF EACH EP"""
    # for i in range(num_eps):
    #     for j in range(num_eps):
    #         if i == j:
    #             mean_distances[i, j] = 0
    #         else:
    #             total_distance = 0
    #             for k in range(synergies_per_ep):
    #                 comp_i_index = i * synergies_per_ep + k
    #                 comp_j_index = j * synergies_per_ep + k
    #                 total_distance += distances[comp_i_index, comp_j_index]
    #
    #             mean_distances[i, j] = total_distance
    #
    # mean_distances /= synergies_per_ep

    """FIND MINIMAL DISTANCE BETWEEN COMPONENTS OF EACH EP"""
    for iter_i in range(num_eps):
        for iter_j in range(num_eps):

            if iter_i == iter_j:
                mean_distances[iter_i, iter_j] = 0
            else:
                min_distances = []
                for k in range(synergies_per_ep):
                    comp_i_index = iter_i * synergies_per_ep + k
                    min_distance = np.min(
                        distances[comp_i_index, iter_j * synergies_per_ep:(iter_j + 1) * synergies_per_ep])
                    min_distances.append(min_distance)
                mean_distances[iter_i, iter_j] = np.mean(min_distances)

    """PROCRUSTES DISTANCE"""
    # for i in range(num_eps):
    #     for j in range(num_eps):
    #
    #         if i == j:
    #             mean_distances[i, j] = 0
    #         else:
    #             start_i = i * synergies_per_ep
    #             end_i = start_i + synergies_per_ep
    #             start_j = j * synergies_per_ep
    #             end_j = start_j + synergies_per_ep
    #
    #             suj_a = syn_df.iloc[start_i:end_i, :-1]
    #             suj_b = syn_df.iloc[start_j:end_j, :-1]
    #             _, _, dist = procrustes(suj_a, suj_b)
    #
    #             mean_distances[i, j] = dist

    """SILHOUETTE SCORES"""
    silhouette_scores = []
    clusters = []

    for n_clust in number_clusters:
        clustering_model = AgglomerativeClustering(metric='precomputed', linkage='average', n_clusters=n_clust)
        # clustering_model = AgglomerativeClustering(linkage='average', n_clusters=n_clust)
        cluster_labels = clustering_model.fit_predict(mean_distances)
        # print(cluster_labels)
        clusters.append(cluster_labels)
        silh_score = metrics.silhouette_score(mean_distances, cluster_labels)
        # silh_score = silhouette_scores_custom(mean_distances, cluster_labels)
        silhouette_scores.append(silh_score)

    """SHOW CLUSTER COMPONENTS FOR BEST SILHOUETTE SCORE"""
    best_score_idx = silhouette_scores.index(max(silhouette_scores))
    best_cluster = clusters[best_score_idx]

    print('Best score', round(max(silhouette_scores), 3), 'with', number_clusters[best_score_idx], 'clusters')

    for cl in np.unique(best_cluster):

        cl_idx = [i for i in range(len(best_cluster)) if best_cluster[i] == cl]
        print('Cluster', cl, ':')
        print([eps[x] for x in cl_idx])

    resulting_clusters = pd.concat([pd.Series(eps), pd.Series(best_cluster)], axis=1)
    resulting_clusters.to_csv('./results/EP/resulting_components.csv', header=False, index=False)
    a=1

    """PLOT"""
    plt.plot(number_clusters, silhouette_scores)
    plt.xticks(number_clusters, fontsize=14)  # Setting x-ticks to the exact values in number_clusters
    plt.yticks(np.linspace(0, 0.11, 12), ['{:.2f}'.format(x) for x in np.linspace(0, 0.11, 12)], fontsize=14)  # Setting y-ticks
    plt.ylim(0, 0.11)  # Setting the y-axis limits to 0 to 0.11
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores for EP clusters')
    # plt.show()
    # plt.savefig('./results/silhouette_ep_clusters.png', dpi=600)
    plt.savefig('./results/silhouette_ep_clusters.svg', format='svg', dpi=600)
    # plt.savefig('./results/test_silhouette_subject_clusters.png', dpi=600)
