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
from itertools import combinations
from statsmodels.multivariate.manova import MANOVA
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import mannwhitneyu
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.distance import cdist
from numpy import std, mean, sqrt

from classification import get_raw_best_params
from synergy_pipeline import kin_syn_classif
from synergy_pipeline import emg_pca_syn_classif
from synergy_pipeline import tact_syn_classif

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def ep_from_scores_classif():

    discard = 'less'

    cv = 3
    [kin_params, emg_params, tact_params] = get_raw_best_params()
    kin_bins = kin_params[0]
    emg_bins = emg_params[0]
    tact_bins = tact_params[0]
    # perc_syns = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
    # families = ['Ball', 'Cutlery', 'Geometric', 'Mugs', 'Plates']
    l1VSl2 = [0, 0.25, 0.5, 0.75, 1]
    c_param = [0.01, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5]

    # LOAD EXTRA DATA
    extra_data = pd.read_csv('./results/Syn/extra_data.csv')

    # SET AND OPEN RESULT FILE
    res_file_name = './results/Syn/accuracy/ep_alternative_syn_results.csv'
    result_file = open(res_file_name, 'a')
    wr = csv.writer(result_file)

    # GET SCORES
    # sources = ['kin', 'emg_pca', 'tact']
    sources = ['kin']

    kin_score_df = pd.DataFrame()
    emg_score_df = pd.DataFrame()
    tact_score_df = pd.DataFrame()

    for source in sources:

        # score_files = glob.glob('./results/Syn/scores/sub*' + source + '_scores.csv')
        score_files = glob.glob('./results/Syn/scores/reordered_alternative_' + source + '_scores.csv')
        score_files.sort()

        for iter_file in range(0, len(score_files)):

            subj_dat = pd.read_csv(score_files[iter_file])
            subj_dat.drop(subj_dat.columns[0], axis=1, inplace=True)

            if source == 'kin':
                kin_score_df = pd.concat([kin_score_df, subj_dat])
            elif source == 'emg_pca':
                emg_score_df = pd.concat([emg_score_df, subj_dat])
            else:
                tact_score_df = pd.concat([tact_score_df, subj_dat])

    # BUILD ITERABLE STRUCTURES
    # all_param = list(itertools.product(perc_syns, families, l1VSl2, c_param))
    all_param = list(itertools.product(l1VSl2, c_param))
    kin_data_and_iter = [[kin_score_df, extra_data, x, cv, kin_bins, discard] for x in all_param]
    # emg_pca_data_and_iter = [[emg_score_df, extra_data, x, cv, emg_bins, discard] for x in all_param]
    # tact_data_and_iter = [[tact_score_df, extra_data, x, cv, tact_bins, discard] for x in all_param]

    # multiprocessing
    with Pool() as pool:

        """NOPE"""

        result_kin = pool.map_async(kin_ep_classif, kin_data_and_iter)
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

    total_score = []

    num_syns = np.ceil(len(kin_scores.columns) * perc_syns / 100)
    extra_data.reset_index(inplace=True, drop=True)
    kin_scores.reset_index(inplace=True, drop=True)

    if discard == 'less':
        data_df = pd.concat([kin_scores.iloc[:, 0:int(num_syns)], extra_data], axis=1)  # keeps most relevant
    else:
        data_df = pd.concat([kin_scores.iloc[:, -int(num_syns):], extra_data], axis=1)  # discards most relevant

    # selected_df = data_df.loc[data_df['Family'] == family]
    to_kfold = data_df.drop_duplicates(subset=['Trial num', 'EP'])

    random_states = [42, 43, 44]
    for rnd_st in random_states:

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=rnd_st)
        # WARNING: the skf.split returns the indexes
        for train, test in skf.split(to_kfold['Trial num'].astype(int), to_kfold['EP'].astype(str)):

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
                        train_labels.append(np.unique(train_tri['EP'])[0])
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
                        test_labels.append(np.unique(test_tri['EP'])[0])
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

    result = ['Kin']
    result.extend(input_data[2])
    result.append(total_score)
    result.append(round(np.mean(total_score), 2))
    # print("RESULT:", result)

    return result