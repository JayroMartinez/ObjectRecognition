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
from sklearn.metrics.pairwise import cosine_similarity

from classification import get_raw_best_params

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def silhouette_scores_custom(distances, cluster_labels):
    n_samples = distances.shape[0]
    silhouette_scores = np.zeros(n_samples)

    # Iterate through each sample
    for i in range(n_samples):
        # Current sample's cluster
        own_cluster = cluster_labels[i]

        # Intra-cluster distances (a)
        intra_cluster_distances = distances[i, cluster_labels == own_cluster]
        a = np.mean(intra_cluster_distances[intra_cluster_distances != 0])  # Exclude distance to itself

        # Nearest cluster calculation (b)
        inter_cluster_distances = []
        for other_cluster in set(cluster_labels):
            if other_cluster != own_cluster:
                # Mean distance to all samples in the other cluster
                inter_cluster_distances.append(np.mean(distances[i, cluster_labels == other_cluster]))
        b = np.min(inter_cluster_distances)  # Minimum mean distance to any other cluster

        # Silhouette score for this sample
        silhouette_scores[i] = (b - a) / max(a, b)

    return silhouette_scores


def all_subjects_comp():

    sources = ['kin', 'emg_pca', 'tact']

    for source in sources:

        score_file = './results/Syn/scores/' + source + '_scores.csv'
        score_df = pd.read_csv(score_file)
        score_df = score_df.drop(score_df.columns[0], axis=1)

        extra_data = pd.read_csv('./results/Syn/extra_data.csv')

        score_df.columns = ['Syn' + x for x in score_df.columns]
        # selected_data = pd.concat([score_df, extra_data['EP'], extra_data['Subject'], extra_data['Family']], axis=1)
        # selected_data = pd.concat([score_df, extra_data['EP'], extra_data['Subject'], extra_data['Given Object']], axis=1)
        selected_data = pd.concat([score_df, extra_data['EP'], extra_data['Subject'], extra_data['Given Object'], extra_data['Family']], axis=1)

        """MANOVA"""
        # cols = list(selected_data.columns)
        # cols.remove('EP')
        # cols.remove('Subject')
        # # cols.remove('Family')
        # cols.remove('Given Object')
        # # form = f"{' + '.join(cols)} ~ EP + Subject + Family"
        # form = f"{' + '.join(cols)} ~ EP + Subject + Q('Given Object')"
        # manova = MANOVA.from_formula(formula=form, data=selected_data)
        # result = manova.mv_test()
        # print(result.summary())


        """MIXED EFFECTS MODEL"""
        # cols = list(selected_data.columns)
        # cols.remove('EP')
        # cols.remove('Subject')
        # cols.remove('Family')
        # cols.remove('Given Object')
        #
        # # Reshape the data to have each synergy score component as a separate row
        # melted_data = pd.melt(selected_data, id_vars=['Subject', 'EP', 'Given Object', 'Family'],
        #                       value_vars=cols, var_name='Synergy_Component', value_name='Synergy_Score')
        #
        # # Define the formula string with the reshaped data
        # form = "Synergy_Score ~ EP + Family + Q('Given Object')"
        #
        # # Define and fit the mixed effects model
        # # model = smf.mixedlm(form, data=melted_data, groups=melted_data['Subject']).fit()
        # model = smf.mixedlm(form, data=melted_data, groups=melted_data['Subject']).fit()
        #
        # # Print model summary
        # print(model.summary())

        a=1

        # i = sns.boxplot(data=selected_data, x='EP', y='Syn')
        # pairs = list(combinations(extra_data['EP'].unique(), 2))
        # test_results = add_stat_annotation(i,
        #                                    data=selected_data,
        #                                    x='EP',
        #                                    y='Syn',
        #                                    box_pairs=pairs,
        #                                    test='Kruskal',
        #                                    text_format='star',
        #                                    loc='inside',
        #                                    verbose=2,
        #                                    line_offset=0.1,
        #                                    fontsize='x-small')
        # i.set(title=source.upper() + ' Syn ' + str(iter))
        # i.set(ylabel='Scores')
        # _, xlabels = plt.xticks()
        # i.set_xticklabels(xlabels, size=5)
        # plt.xticks(rotation=45)
        # plt.tight_layout()
        # # plt.show()
        # plt.savefig('./results/Syn/plots/all_subj_syns/' + source + '_synergy_' + str(iter)+'.png', dpi=600, bbox_inches='tight')
        # plt.close()
        # plt.clf()


def clustered_comp():

    sources = ['kin', 'emg_pca', 'tact']

    for source in sources:

        score_file = './results/Syn/scores/reordered_' + source + '_scores.csv'
        score_df = pd.read_csv(score_file)
        score_df = score_df.drop(score_df.columns[0], axis=1)

        extra_data = pd.read_csv('./results/Syn/extra_data.csv')

        for iter in range(0, len(score_df.columns)):
            selected_syn = score_df[str(iter)]
            selected_data = pd.DataFrame({'Syn': selected_syn, 'EP': extra_data['EP']})

            i = sns.boxplot(data=selected_data, x='EP', y='Syn')
            pairs = list(combinations(extra_data['EP'].unique(), 2))
            test_results = add_stat_annotation(i,
                                               data=selected_data,
                                               x='EP',
                                               y='Syn',
                                               box_pairs=pairs,
                                               test='Kruskal',
                                               text_format='star',
                                               loc='inside',
                                               verbose=2,
                                               line_offset=0.1,
                                               fontsize='x-small')
            i.set(title=source.upper() + ' Syn ' + str(iter))
            i.set(ylabel='Scores')
            _, xlabels = plt.xticks()
            i.set_xticklabels(xlabels, size=5)
            plt.xticks(rotation=45)
            plt.tight_layout()
            # plt.show()
            plt.savefig('./results/Syn/plots/clustered_syns/' + source + '_synergy_' + str(iter) + '.png',
                        dpi=600, bbox_inches='tight')
            plt.close()
            plt.clf()


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
    elif type == 'family':
        res_file_name = './results/Syn/accuracy/fam_subj_clust'
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


def extract_early_enclosure_alt():

    """When computing the variance for the selected datapoints (early enclosure) we get that the order of synergies
    should change (e.g.: the variance of Syn5 in the selected data is higher than variance for Syn4). We are going to
    discard this change in the order since we want to compare how the same synergies behave in these two different
    conditions"""

    sources = ['kin', 'emg_pca', 'tact']
    extra_data = pd.read_csv('./results/Syn/extra_data.csv')
    early_enclosure = extra_data[(extra_data['EP num'].astype('string').isin(['0', '1'])) & (extra_data['EP'].isin(['enclosure', 'enclosure part']))]
    selected_extra_data = extra_data.iloc[early_enclosure.index]
    selected_extra_data.to_csv('./results/Early Enclosure/alternative_early_enclosure_extra_data.csv')

    for source in sources:

        source_data = pd.read_csv('./results/Syn/scores/reordered_' + source + '_scores.csv')
        source_data = source_data.drop(source_data.columns[0], axis=1)
        var_source = source_data.var(axis='index') * len(source_data)
        tot_var_source = sum(source_data.var(axis='index')) * len(source_data)

        selected_data = source_data.iloc[early_enclosure.index]
        var_selected = selected_data.var(axis='index') * len(selected_data)
        # new_order = var_selected.sort_values(ascending=False)
        # reordered_selected = selected_data[new_order.index]
        # reordered_selected.columns = source_data.columns
        selected_data.to_csv('./results/Early Enclosure/scores/alternative_reordered_' + source + '_scores.csv')
        # reordered_var = var_selected[new_order.index]
        tot_var_sel = sum(var_selected)
        ratio = tot_var_sel / tot_var_source
        ratio_var = [(x * ratio)/tot_var_sel for x in var_selected]
        ratio_var_df = pd.Series(ratio_var)
        ratio_var_df.to_csv('./results/Early Enclosure/variance/alternative_reordered_' + source + '_var_tot.csv')


def score_reordering(type):

    # sources = ['kin', 'emg_pca', 'tact']
    # sources = ['kin', 'tact']
    sources = ['kin']

    for source in sources:

        if type == 'agglomerative':
            source_clusters = pd.read_csv('./results/Syn/resulting_components/agglomerative_' + source + '.csv')
        else:  # alternative
            source_clusters = pd.read_csv('./results/Syn/resulting_components/alternative_' + source + '.csv')

        source_clusters.drop(source_clusters.columns[0], axis=1, inplace=True)

        score_files = glob.glob('./results/Syn/scores/sub*' + source + '_scores.csv')
        score_files.sort()

        all_data = pd.DataFrame()
        all_var = pd.DataFrame()

        source_tot_var = pd.read_csv('./results/Syn/variance/' + source + '_var_tot.csv')
        source_tot_var.drop(source_tot_var.columns[0], axis=1, inplace=True)

        number_datapoints = 0

        for iter_file in range(0, len(score_files)):

            subj_dat = pd.read_csv(score_files[iter_file])
            subj_dat.drop(subj_dat.columns[0], axis=1, inplace=True)

            number_datapoints += len(subj_dat)

            # var_file = score_files[iter_file].replace('_scores', '_var')
            # var_file = var_file.replace('scores', 'variance')
            # subj_var = pd.read_csv(var_file)
            # subj_var.drop(subj_var.columns[0], axis=1, inplace=True)

            var_file = score_files[iter_file].replace('_scores', '_var_tot')
            var_file = var_file.replace('scores', 'variance')
            subj_var = pd.read_csv(var_file)
            subj_var.drop(subj_var.columns[0], axis=1, inplace=True)
            subj_var = subj_var * len(subj_dat)

            new_subj_dat = np.empty(subj_dat.shape)
            new_subj_dat[:] = np.nan

            new_subj_var = np.empty([len(subj_dat.columns)])
            new_subj_var[:] = np.nan

            subj_order = source_clusters.iloc[iter_file, :]

            for iter_cols in range(0, len(subj_order)):

                if ~np.isnan(subj_order[iter_cols]):

                    new_subj_dat[:, iter_cols] = subj_dat.iloc[:, int(subj_order[iter_cols])]
                    new_subj_var[iter_cols] = subj_var.iloc[int(subj_order[iter_cols])]
                    a=1

            all_data = pd.concat([all_data, pd.DataFrame(new_subj_dat)])
            all_var = pd.concat([all_var, pd.DataFrame(new_subj_var).T], ignore_index=True)

        """ AFTER FILLING ALL SUBJECTS WE NEED TO PERFORM IMPUTATION """
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp_data = imp.fit_transform(all_data)

        """ WE ALSO NEED TO COMPUTE VARIANCE FOR EACH SYNERGY AND REORDER SYNS """
        # entire_dataset_var = source_tot_var.sum() * number_datapoints
        rescaled_var = all_var / number_datapoints
        sum_var = rescaled_var.sum(axis=0, skipna=True)
        variance = [x / source_tot_var.sum() for x in sum_var]
        var = pd.DataFrame(variance)
        var_sort = var.sort_values(by='0', ascending=False)

        if type == 'agglomerative':
            var_sort.to_csv('./results/Syn/variance/overall_var_' + source + '.csv')
        else:  # alternative
            var_sort.to_csv('./results/Syn/variance/alternative_overall_var_' + source + '.csv')

        source_clusters.columns = [int(x) for x in source_clusters.columns]
        reordered_clusters = source_clusters.iloc[:, var_sort.index]
        reordered_scores = pd.DataFrame(imp_data[:, var_sort.index])

        if type == 'agglomerative':
            reordered_clusters.to_csv('./results/Syn/resulting_components/reordered_' + source + '.csv')
            reordered_scores.to_csv('./results/Syn/scores/reordered_' + source + '_scores.csv')
        else:  # alternative
            reordered_clusters.to_csv('./results/Syn/resulting_components/reordered_alternative_' + source + '.csv')
            reordered_scores.to_csv('./results/Syn/scores/reordered_alternative_' + source + '_scores.csv')


def score_reordering_early_enclosure():

    """This function is used only for early enclosure when we select the data and then extract the synergies, the old version which is no longer used"""
    sources = ['kin', 'emg_pca', 'tact']

    for source in sources:

        source_clusters = pd.read_csv('./results/Early Enclosure/resulting_components/agglomerative_' + source + '.csv')
        source_clusters.drop(source_clusters.columns[0], axis=1, inplace=True)

        score_files = glob.glob('./results/Early Enclosure/scores/*' + source + '_scores.csv')
        score_files.sort()

        all_data = pd.DataFrame()
        all_var = pd.DataFrame()

        for iter_file in range(0, len(score_files)):

            subj_dat = pd.read_csv(score_files[iter_file])
            subj_dat.drop(subj_dat.columns[0], axis=1, inplace=True)

            var_file = score_files[iter_file].replace('_scores', '_var')
            var_file = var_file.replace('scores', 'variance')
            subj_var = pd.read_csv(var_file)
            subj_var.drop(subj_var.columns[0], axis=1, inplace=True)

            new_subj_dat = np.empty(subj_dat.shape)
            new_subj_dat[:] = np.nan

            new_subj_var = np.empty([len(subj_dat.columns)])
            new_subj_var[:] = np.nan

            subj_order = source_clusters.iloc[iter_file, :]

            for iter_cols in range(0, len(subj_order)):

                if ~np.isnan(subj_order[iter_cols]):
                    new_subj_dat[:, iter_cols] = subj_dat.iloc[:, int(subj_order[iter_cols])]
                    new_subj_var[iter_cols] = subj_var.iloc[int(subj_order[iter_cols])]
                    a = 1

            all_data = pd.concat([all_data, pd.DataFrame(new_subj_dat)])
            all_var = pd.concat([all_var, pd.DataFrame(new_subj_var).T], ignore_index=True)

        """ AFTER FILLING ALL SUBJECTS WE NEED TO PERFORM IMPUTATION """
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp_data = imp.fit_transform(all_data)

        """ WE ALSO NEED TO COMPUTE VARIANCE FOR EACH SYNERGY AND REORDER SYNS """
        variance = all_var.mean(axis=0, skipna=True)
        variance = variance.sort_values(ascending=False)
        variance.to_csv('./results/Early Enclosure/variance/overall_var_' + source + '.csv')

        reordered_scores = pd.DataFrame(imp_data[:, variance.index])
        reordered_scores.to_csv('./results/Early Enclosure/scores/reordered_' + source + '_scores.csv')

    a=1


def syn_clustering_early_enclosure():

    """This function is created to pair synergies among subjects"""

    sources = ['kin', 'emg_pca', 'tact']

    for source in sources:

        selected_files = glob.glob('./results/Early Enclosure/synergies/*' + source + '_syns.csv')
        selected_files = [x for x in selected_files if 'sub' in x]  # selects only subject files
        selected_files.sort()

        all_data = pd.DataFrame()

        for file in selected_files:

            suj_dat = pd.read_csv(file)
            components = suj_dat.pop('Unnamed: 0').to_list()
            suj_dat = suj_dat.T
            suj_dat['Component'] = components
            aux_sub = file.replace('./results/Early Enclosure/synergies/', '')
            subject = aux_sub.replace('_' + source + '_syns.csv', '')
            suj_dat['Subject'] = np.repeat(subject, suj_dat.shape[0])

            all_data = pd.concat([all_data, pd.DataFrame(suj_dat)], ignore_index=True)

        numerical_data = all_data.select_dtypes(include='float64')
        num_clust = len(numerical_data.columns) + 1  # we include an extra cluster to drop "garbage" components

        resulting_clusters = np.empty((len(selected_files), len(numerical_data.columns)))
        resulting_clusters[:] = np.nan  # fill array with NaNs

        for iter_clust in range(0, num_clust - 1):  # -1 because makes no sense to ask for 1 cluster

            remaining_clusters = num_clust - iter_clust

            # model = KMeans(n_clusters=remaining_clusters, init='k-means++', algorithm='lloyd', n_init=1000, max_iter=100000)

            model = AgglomerativeClustering(n_clusters=remaining_clusters, metric='cosine', linkage='average')

            numerical_data = all_data.select_dtypes(include='float64')
            model.fit(numerical_data)
            # syns = pd.DataFrame(model.labels_.reshape((-1, len(numerical_data.columns))))
            # print(metrics.silhouette_score(numerical_data, model.labels_, metric='euclidean'))
            silh_score = metrics.silhouette_samples(numerical_data, model.labels_)
            # silh_df = silh_score
            # silh_df = pd.DataFrame(silh_score.reshape((-1, len(numerical_data.columns))))
            # print(sum(silh_score))
            # clus_scores_df = pd.DataFrame({'label': model.labels_, 'score': silh_score})
            # score_per_clust = clus_scores_df.groupby(by=['label']).median()

            components_df = all_data[['Component', 'Subject']]
            components_df['Label'] = model.labels_
            components_df['Score'] = silh_score

            subjects = np.unique(components_df['Subject'])

            """checking for best cluster after removing duplicates [subject-label]"""
            aux_df = components_df.sort_values(by='Score', ascending=False)
            aux_df.drop_duplicates(subset=['Subject', 'Label'], keep='first', inplace=True)
            score_per_clust = aux_df.groupby(by=['Label'])['Score'].mean()
            best_clust = score_per_clust.idxmax()

            for it_subj in range(len(subjects)):

                select_components = components_df.loc[(components_df['Subject'] == str(subjects[it_subj])) & (components_df['Label'] == int(best_clust))]

                if (len(select_components.index) > 0) & (select_components['Score'].max() > 0):  # If there are components for that subject in the cluster and the sample is well clustered

                    best_suj_component = select_components.loc[select_components['Score'].idxmax()]['Component']

                    """SAVE THE CLUSTERED SYNERGIES [rows=subjects, columns=this cluster {clusters still not ordered}]"""
                    resulting_clusters[it_subj, iter_clust] = int(best_suj_component)

                    """WE DROP THE ALREADY CLUSTERED SYNERGIES"""
                    all_data.drop(all_data.loc[(all_data["Subject"] == subjects[it_subj]) & (all_data["Component"] == best_suj_component)].index, inplace=True)

        resulting_clusters_df = pd.DataFrame(resulting_clusters)
        resulting_clusters_df.to_csv('./results/Early Enclosure/resulting_components/agglomerative_' + source + '.csv', mode='a')


def syn_clustering():

    """This function is created to pair synergies among subjects"""

    sources = ['kin', 'emg_pca', 'tact']

    for source in sources:

        selected_files = glob.glob('./results/Syn/synergies/*' + source + '_syns.csv')
        selected_files = [x for x in selected_files if 'sub' in x]  # selects only subject files
        selected_files.sort()

        all_data = pd.DataFrame()

        for file in selected_files:

            suj_dat = pd.read_csv(file)
            components = suj_dat.pop('Unnamed: 0').to_list()
            suj_dat = suj_dat.T
            suj_dat['Component'] = components
            aux_sub = file.replace('./results/Syn/synergies/', '')
            subject = aux_sub.replace('_' + source + '_syns.csv', '')
            suj_dat['Subject'] = np.repeat(subject, suj_dat.shape[0])

            all_data = pd.concat([all_data, pd.DataFrame(suj_dat)], ignore_index=True)

        numerical_data = all_data.select_dtypes(include='float64')
        num_clust = len(numerical_data.columns) + 1  # we include an extra cluster to drop "garbage" components

        resulting_clusters = np.empty((len(selected_files), len(numerical_data.columns)))
        resulting_clusters[:] = np.nan  # fill array with NaNs

        for iter_clust in range(0, num_clust - 1):  # -1 because makes no sense to ask for 1 cluster

            remaining_clusters = num_clust - iter_clust

            # model = KMeans(n_clusters=remaining_clusters, init='k-means++', algorithm='lloyd', n_init=1000, max_iter=100000)

            model = AgglomerativeClustering(n_clusters=remaining_clusters, metric='cosine', linkage='average', compute_distances=True)
            # model = AgglomerativeClustering(n_clusters=remaining_clusters, affinity=custom_metric, linkage='average')

            numerical_data = all_data.select_dtypes(include='float64')
            model.fit(numerical_data)
            # syns = pd.DataFrame(model.labels_.reshape((-1, len(numerical_data.columns))))
            # print(metrics.silhouette_score(numerical_data, model.labels_, metric='euclidean'))
            silh_score = metrics.silhouette_samples(numerical_data, model.labels_)
            # silh_df = silh_score
            # silh_df = pd.DataFrame(silh_score.reshape((-1, len(numerical_data.columns))))
            # print(sum(silh_score))
            # clus_scores_df = pd.DataFrame({'label': model.labels_, 'score': silh_score})
            # score_per_clust = clus_scores_df.groupby(by=['label']).median()

            components_df = all_data[['Component', 'Subject']]
            components_df['Label'] = model.labels_
            components_df['Score'] = silh_score

            subjects = np.unique(components_df['Subject'])

            """checking for best cluster after removing duplicates [subject-label]"""
            aux_df = components_df.sort_values(by='Score', ascending=False)
            aux_df.drop_duplicates(subset=['Subject', 'Label'], keep='first', inplace=True)
            score_per_clust = aux_df.groupby(by=['Label'])['Score'].mean()
            best_clust = score_per_clust.idxmax()

            for it_subj in range(len(subjects)):

                select_components = components_df.loc[(components_df['Subject'] == str(subjects[it_subj])) & (components_df['Label'] == int(best_clust))]

                if (len(select_components.index) > 0) & (select_components['Score'].max() > 0):  # If there are components for that subject in the cluster and the sample is well clustered

                    best_suj_component = select_components.loc[select_components['Score'].idxmax()]['Component']

                    """SAVE THE CLUSTERED SYNERGIES [rows=subjects, columns=this cluster {clusters still not ordered}]"""
                    resulting_clusters[it_subj, iter_clust] = int(best_suj_component)

                    """WE DROP THE ALREADY CLUSTERED SYNERGIES"""
                    all_data.drop(all_data.loc[(all_data["Subject"] == subjects[it_subj]) & (all_data["Component"] == best_suj_component)].index, inplace=True)

        resulting_clusters_df = pd.DataFrame(resulting_clusters)
        resulting_clusters_df.to_csv('./results/Syn/resulting_components/agglomerative_' + source + '.csv', mode='a')


def syn_clustering_alternative():

    # sources = ['kin', 'emg_pca', 'tact']
    sources = ['kin', 'tact']

    for source in sources:

        selected_files = glob.glob('./results/Syn/synergies/*' + source + '_syns.csv')
        selected_files = [x for x in selected_files if 'sub' in x]  # selects only subject files
        selected_files.sort()

        all_data = pd.DataFrame()

        for file in selected_files:
            suj_dat = pd.read_csv(file)
            components = suj_dat.pop('Unnamed: 0').to_list()
            suj_dat = suj_dat.T
            suj_dat['Component'] = components
            aux_sub = file.replace('./results/Syn/synergies/', '')
            subject = aux_sub.replace('_' + source + '_syns.csv', '')
            suj_dat['Subject'] = np.repeat(subject, suj_dat.shape[0])

            all_data = pd.concat([all_data, pd.DataFrame(suj_dat)], ignore_index=True)


        results_index = np.unique(all_data['Subject'])
        cols = np.unique(all_data['Component'])
        # results_columns = ['Syn ' + str(x) for x in cols]
        empty = np.empty((len(results_index), len(cols),))
        empty[:] = np.nan
        results = pd.DataFrame(data=empty)
        results.index = results_index
        results.columns = cols

        numerical_data = all_data.select_dtypes(include='float64')
        num_clust = len(numerical_data.columns) + 1  # we include an extra cluster to drop "garbage" components

        resulting_clusters = np.empty((len(selected_files), len(numerical_data.columns)))
        resulting_clusters[:] = np.nan  # fill array with NaNs

        for iter_clust in range(0, num_clust - 1):  # -1 because makes no sense to ask for 1 cluster

            remaining_clusters = num_clust - iter_clust
            numerical_data = all_data.select_dtypes(include='float64')

            syns = numerical_data.T
            syns_corr = syns.corr().abs()
            # mask = np.tril(np.ones_like(syns_corr, dtype=bool))
            # tri_syns_corr = syns_corr.where(mask)
            # np.fill_diagonal(tri_syns_corr.values, 0)

            distances = 1 - syns_corr.to_numpy()
            clustering_model = AgglomerativeClustering(affinity='precomputed', linkage='average', n_clusters=remaining_clusters)
            cluster_labels = clustering_model.fit_predict(distances)
            # reshaped_clusters = np.reshape(cluster_labels, (-1, remaining_clusters-1))
            # silh_score = metrics.silhouette_samples(numerical_data, cluster_labels)
            silh_score = silhouette_scores_custom(distances, cluster_labels)


            components_df = all_data[['Component', 'Subject']]
            components_df['Label'] = cluster_labels
            components_df['Score'] = silh_score

            """checking for best cluster after removing duplicates [subject-label]"""
            aux_df = components_df.sort_values(by='Score', ascending=False)
            aux_df.drop_duplicates(subset=['Subject', 'Label'], keep='first', inplace=True)
            score_per_clust = aux_df.groupby(by=['Label'])['Score'].mean()
            best_clust = score_per_clust.idxmax()

            subjects = np.unique(components_df['Subject'])

            for it_subj in range(len(subjects)):

                select_components = components_df.loc[
                    (components_df['Subject'] == str(subjects[it_subj])) & (components_df['Label'] == int(best_clust))]

                if (len(select_components.index) > 0) & (select_components['Score'].max() > 0):  # If there are components for that subject in the cluster and the sample is well clustered
                # if (len(select_components.index) > 0):  # If there are components for that subject in the cluster

                    best_suj_component = select_components.loc[select_components['Score'].idxmax()]['Component']

                    """SAVE THE CLUSTERED SYNERGIES [rows=subjects, columns=this cluster {clusters still not ordered}]"""
                    resulting_clusters[it_subj, iter_clust] = int(best_suj_component)

                    """WE DROP THE ALREADY CLUSTERED SYNERGIES"""
                    all_data.drop(all_data.loc[(all_data["Subject"] == subjects[it_subj]) & (all_data["Component"] == best_suj_component)].index, inplace=True)
                # else:
                #     a=1

        resulting_clusters_df = pd.DataFrame(resulting_clusters)
        resulting_clusters_df.to_csv('./results/Syn/resulting_components/alternative_' + source + '.csv', mode='a')

        """OLD CODE"""
        # for iter in cols:
        #
        #     subjects = np.unique(all_data['Subject'])
        #
        #     max_pos = tri_syns_corr.stack().idxmax()  # Get coordinates with max value
        #
        #     # That coordinates correspond to (SujA-CompX vs SujB-CompY)
        #     suj_a = all_data.iloc[[max_pos[0]]]['Subject']
        #     comp_a = all_data.iloc[[max_pos[0]]]['Component']
        #
        #     suj_b = all_data.iloc[[max_pos[1]]]['Subject']
        #     comp_b = all_data.iloc[[max_pos[1]]]['Component']
        #
        #     # Remove these two subjects from the list to search for the other subjects
        #     subjects = np.delete(subjects, np.where(subjects == suj_a.values))
        #     subjects = np.delete(subjects, np.where(subjects == suj_b.values))
        #
        #     # Save the selected components in a synergy
        #     results.loc[suj_a.values, iter] = comp_a.values
        #     results.loc[suj_b.values, iter] = comp_b.values
        #
        #     a_comps = []
        #     b_comps = []
        #
        #     a_corr = []
        #     b_corr = []
        #
        #     # Select correlations for both selected subjects
        #     suj_a_idx = all_data.loc[all_data['Subject'] == suj_a.values[0]].index
        #     suj_a_data = tri_syns_corr.iloc[suj_a_idx]
        #
        #     suj_b_idx = all_data.loc[all_data['Subject'] == suj_b.values[0]].index
        #     suj_b_data = tri_syns_corr.iloc[suj_b_idx]
        #
        #     # for each subject find best match to each component
        #     for suj in subjects:
        #
        #         sel_idx = all_data.loc[all_data['Subject'] == suj].index
        #
        #         sel_data_suj_a = suj_a_data.iloc[:, sel_idx]
        #
        #         sel_data_suj_b = suj_b_data.iloc[:, sel_idx]
        #
        #         # # Now we check the correlation of SujA-CompA to the components of the selected subject
        #         # # If the best match is also the best match for the selected component of the selected subject we save it
        #         # column_a = sel_data_suj_a.columns[comp_a.values]
        #         # if sel_data_suj_a.loc[max_pos[0], column_a].item() >= sel_data_suj_a.loc[:, column_a].max().item():
        #         #     a_comps.append(column_a)  # Save the component
        #         #     a_corr.append(sel_data_suj_a.loc[max_pos[0], column_a])  # Save the correlation value
        #         #
        #         # column_b = sel_data_suj_b.columns[comp_b.values]
        #         # if sel_data_suj_b.loc[max_pos[1], column_b].item() >= sel_data_suj_b.loc[:, column_b].max().item():
        #         #     b_comps.append(column_b)  # Save the component
        #         #     b_corr.append(sel_data_suj_b.loc[max_pos[1], column_b])  # Save the correlation value
        #
        #         column_a = sel_data_suj_a.columns[comp_a.values]
        #         a_comps.append(column_a)  # Save the component
        #         a_corr.append(sel_data_suj_a.loc[max_pos[0], column_a])  # Save the correlation value
        #         column_b = sel_data_suj_b.columns[comp_b.values]
        #         b_comps.append(column_b)  # Save the component
        #         b_corr.append(sel_data_suj_b.loc[max_pos[1], column_b])  # Save the correlation value
        #
        #     if np.mean(a_corr) > np.mean(b_corr):
        #         comps = a_comps
        #     else:
        #         comps = b_comps
        #
        #     for comp in comps:
        #         subject = all_data.loc[comp]['Subject']
        #         component = all_data.loc[comp]['Component']
        #
        #         results.loc[subject.values, iter] = component.values
        #
        #         tri_syns_corr.loc[comp] = np.nan
        #         tri_syns_corr.loc[:, comp] = np.nan
        #
        #     tri_syns_corr.loc[max_pos[0]] = np.nan
        #     tri_syns_corr.loc[:, max_pos[0]] = np.nan
        #
        #     tri_syns_corr.loc[max_pos[1]] = np.nan
        #     tri_syns_corr.loc[:, max_pos[1]] = np.nan

        a = 1


def kin_syn_extraction(data):

    kin_scaled = StandardScaler().fit_transform(data)  # Z score
    kin_df = pd.DataFrame(kin_scaled)
    pca_mod = PCA()

    kin_scores = pca_mod.fit_transform(kin_df)
    kin_syns = pca_mod.components_.T  # Each column is a synergy
    kin_var = pca_mod.explained_variance_ratio_
    kin_tot_var = pca_mod.explained_variance_

    return [kin_scores, kin_syns, kin_var, kin_tot_var]


def emg_pca_syn_extraction(data):

    emg_scaled = StandardScaler().fit_transform(data)  # Z score
    emg_df = pd.DataFrame(emg_scaled)
    pca_mod = PCA()

    emg_scores = pca_mod.fit_transform(emg_df)
    emg_syns = pca_mod.components_.T  # Each column is a synergy
    emg_var = pca_mod.explained_variance_ratio_
    emg_tot_var = pca_mod.explained_variance_

    return [emg_scores, emg_syns, emg_var, emg_tot_var]


def emg_nmf_syn_extraction(data):
    """Probably useless"""
    perc_syns = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]

    for p in perc_syns:

        num_syn_emg = int(np.ceil(len(data.columns) * p / 100))

        file_name_sc = './results/Syn/scores/emg_nmf' + str(p)
        file_name_sy = './results/Syn/synergies/emg_nmf' + str(p)

        nmf_mod = NMF(n_components=num_syn_emg, max_iter=1500)
        emg_scores = nmf_mod.fit_transform(data)
        pd.DataFrame(emg_scores).to_csv(file_name_sc + '_scores.csv')
        emg_syns = nmf_mod.components_.T
        pd.DataFrame(emg_syns).to_csv(file_name_sy + '_syns.csv')

        print('Done for', p, '%')


def tact_syn_extraction(data):

    tact_scaled = StandardScaler().fit_transform(data)  # Z score
    tact_df = pd.DataFrame(tact_scaled)
    pca_mod = PCA()

    tact_scores = pca_mod.fit_transform(tact_df)
    tact_syns = pca_mod.components_.T  # Each column is a synergy
    tact_var = pca_mod.explained_variance_ratio_
    tact_tot_var = pca_mod.explained_variance_

    return [tact_scores, tact_syns, tact_var, tact_tot_var]


def syn_extraction(data):
    """Synergy extraction for each source using all subjects together"""

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

    # REMOVE NANs
    data_clean = data.dropna(axis=0, how='any')

    # # NON-NUMERIC DATA EXTRACTION & SAVING
    extra_cols = [col for col in data_clean.columns if (col not in kin_cols) and (col not in emg_cols) and (col not in tact_cols)]
    extra_df = data_clean[extra_cols]
    extra_df.to_csv('./results/Syn/extra_data.csv', index=False)

    ## SYNERGY EXTRACTION AND SAVING
    [kin_scores, kin_syns, kin_var, kin_var_tot] = kin_syn_extraction(data_clean[kin_cols])
    pd.DataFrame(kin_scores).to_csv('./results/Syn/scores/kin_scores.csv')
    pd.DataFrame(kin_syns).to_csv('./results/Syn/synergies/kin_syns.csv')
    pd.DataFrame(kin_var).to_csv('./results/Syn/variance/kin_var.csv')
    pd.DataFrame(kin_var_tot).to_csv('./results/Syn/variance/kin_var_tot.csv')

    [emg_scores, emg_syns, emg_var, emg_var_tot] = emg_pca_syn_extraction(data_clean[emg_cols])
    pd.DataFrame(emg_scores).to_csv('./results/Syn/scores/emg_pca_scores.csv')
    pd.DataFrame(emg_syns).to_csv('./results/Syn/synergies/emg_pca_syns.csv')
    pd.DataFrame(emg_var).to_csv('./results/Syn/variance/emg_pca_var.csv')
    pd.DataFrame(emg_var_tot).to_csv('./results/Syn/variance/emg_pca_var_tot.csv')

    # emg_nmf_syn_extraction(data_clean[emg_cols]) # NEEDS TO BE REFACTORED

    [tact_scores, tact_syns, tact_var, tact_var_tot] = tact_syn_extraction(data_clean[tact_cols])
    pd.DataFrame(tact_scores).to_csv('./results/Syn/scores/tact_scores.csv')
    pd.DataFrame(tact_syns).to_csv('./results/Syn/synergies/tact_syns.csv')
    pd.DataFrame(tact_var).to_csv('./results/Syn/variance/tact_var.csv')
    pd.DataFrame(tact_var_tot).to_csv('./results/Syn/variance/tact_var_tot.csv')


def syn_extraction_subj(data):

    """Synergy extraction for each source and each subject"""

    subjects = np.unique(data['Subject'])

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
    extra_cols = [col for col in data_clean.columns if
                  (col not in kin_cols) and (col not in emg_cols) and (col not in tact_cols)]
    extra_df = data_clean[extra_cols]
    extra_df.to_csv('./results/Syn/extra_data.csv', index=False)

    for subj in subjects:

        subj_data = data_clean.loc[data_clean['Subject'] == subj]

        ## SYNERGY EXTRACTION AND SAVING
        [kin_subj_scores, kin_subj_syns, kin_subj_var, kin_subj_var_tot] = kin_syn_extraction(subj_data[kin_cols])
        pd.DataFrame(kin_subj_scores).to_csv('./results/Syn/scores/' + subj + '_kin_scores.csv')
        pd.DataFrame(kin_subj_syns).to_csv('./results/Syn/synergies/' + subj + '_kin_syns.csv')
        pd.DataFrame(kin_subj_var).to_csv('./results/Syn/variance/' + subj + '_kin_var.csv')
        pd.DataFrame(kin_subj_var_tot).to_csv('./results/Syn/variance/' + subj + '_kin_var_tot.csv')

        [emg_pca_subj_scores, emg_pca_subj_syns, emg_pca_subj_var, emg_pca_subj_var_tot] = emg_pca_syn_extraction(subj_data[emg_cols])
        pd.DataFrame(emg_pca_subj_scores).to_csv('./results/Syn/scores/' + subj + '_emg_pca_scores.csv')
        pd.DataFrame(emg_pca_subj_syns).to_csv('./results/Syn/synergies/' + subj + '_emg_pca_syns.csv')
        pd.DataFrame(emg_pca_subj_var).to_csv('./results/Syn/variance/' + subj + '_emg_pca_var.csv')
        pd.DataFrame(emg_pca_subj_var_tot).to_csv('./results/Syn/variance/' + subj + '_emg_pca_var_tot.csv')

        # emg_nmf_syn_extraction(data_clean[emg_cols])

        [tact_subj_scores, tact_subj_syns, tact_subj_var, tact_subj_var_tot] = tact_syn_extraction(subj_data[tact_cols])
        pd.DataFrame(tact_subj_scores).to_csv('./results/Syn/scores/' + subj + '_tact_scores.csv')
        pd.DataFrame(tact_subj_syns).to_csv('./results/Syn/synergies/' + subj + '_tact_syns.csv')
        pd.DataFrame(tact_subj_var).to_csv('./results/Syn/variance/' + subj + '_tact_var.csv')
        pd.DataFrame(tact_subj_var_tot).to_csv('./results/Syn/variance/' + subj + '_tact_var_tot.csv')

    # print("Individual synergy extraction done!")


def syn_extraction_early_enclosure():

    data = pd.read_csv('./results/Early Enclosure/early_enclosure_data.csv')

    subjects = np.unique(data['Subject'])

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
    extra_cols = [col for col in data_clean.columns if
                  (col not in kin_cols) and (col not in emg_cols) and (col not in tact_cols)]
    extra_df = data_clean[extra_cols]
    extra_df.to_csv('./results/Early Enclosure/early_enclosure_extra_data.csv', index=False)

    for subj in subjects:
        subj_data = data_clean.loc[data_clean['Subject'] == subj]

        ## SYNERGY EXTRACTION AND SAVING
        [kin_subj_scores, kin_subj_syns, kin_subj_var] = kin_syn_extraction(subj_data[kin_cols])
        pd.DataFrame(kin_subj_scores).to_csv('./results/Early Enclosure/scores/' + subj + '_kin_scores.csv')
        pd.DataFrame(kin_subj_syns).to_csv('./results/Early Enclosure/synergies/' + subj + '_kin_syns.csv')
        pd.DataFrame(kin_subj_var).to_csv('./results/Early Enclosure/variance/' + subj + '_kin_var.csv')

        [emg_pca_subj_scores, emg_pca_subj_syns, emg_pca_subj_var] = emg_pca_syn_extraction(subj_data[emg_cols])
        pd.DataFrame(emg_pca_subj_scores).to_csv('./results/Early Enclosure/scores/' + subj + '_emg_pca_scores.csv')
        pd.DataFrame(emg_pca_subj_syns).to_csv('./results/Early Enclosure/synergies/' + subj + '_emg_pca_syns.csv')
        pd.DataFrame(emg_pca_subj_var).to_csv('./results/Early Enclosure/variance/' + subj + '_emg_pca_var.csv')

        # emg_nmf_syn_extraction(data_clean[emg_cols])

        [tact_subj_scores, tact_subj_syns, tact_subj_var] = tact_syn_extraction(subj_data[tact_cols])
        pd.DataFrame(tact_subj_scores).to_csv('./results/Early Enclosure/scores/' + subj + '_tact_scores.csv')
        pd.DataFrame(tact_subj_syns).to_csv('./results/Early Enclosure/synergies/' + subj + '_tact_syns.csv')
        pd.DataFrame(tact_subj_var).to_csv('./results/Early Enclosure/variance/' + subj + '_tact_var.csv')

    # print("Early enclosure individual synergy extraction done!")


    a=1


def kin_syn_classif(input_data):

    kin_scores = input_data[0]
    extra_data = input_data[1]
    perc_syns = input_data[2][0]
    family = input_data[2][1]
    l1VSl2 = input_data[2][2]
    c_param = input_data[2][3]
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

    selected_df = data_df.loc[data_df['Family'] == family]
    to_kfold = selected_df.drop_duplicates(subset=['Trial num', 'Given Object'])

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

    discard = input_data[5]

    total_score = []

    num_syns = np.ceil(len(emg_pca_scores.columns) * perc_syns / 100)
    extra_data.reset_index(inplace=True, drop=True)
    emg_pca_scores.reset_index(inplace=True, drop=True)

    if discard == 'less':
        data_df = pd.concat([emg_pca_scores.iloc[:, 0:int(num_syns)], extra_data], axis=1)  # keeps most relevant
    else:
        data_df = pd.concat([emg_pca_scores.iloc[:, -int(num_syns):], extra_data], axis=1)  # discards most relevant

    selected_df = data_df.loc[data_df['Family'] == family]
    to_kfold = selected_df.drop_duplicates(subset=['Trial num', 'Given Object'])

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
    """WE ARE NOT USING THIS FUNCTION"""
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
            # print(log_model.n_iter_)
            # print(sc)

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

    discard = input_data[5]

    total_score = []

    num_syns = np.ceil(len(tact_scores.columns) * perc_syns / 100)
    extra_data.reset_index(inplace=True, drop=True)
    tact_scores.reset_index(inplace=True, drop=True)

    if discard == 'less':
        data_df = pd.concat([tact_scores.iloc[:, 0:int(num_syns)], extra_data], axis=1) # keeps most relevant
    else:
        data_df = pd.concat([tact_scores.iloc[:, -int(num_syns):], extra_data], axis=1)  # discards most relevant

    selected_df = data_df.loc[data_df['Family'] == family]
    to_kfold = selected_df.drop_duplicates(subset=['Trial num', 'Given Object'])

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


def syn_single_source_classification(type, discard):

    cv = 3
    [kin_params, emg_params, tact_params] = get_raw_best_params()
    kin_bins = kin_params[0]
    emg_bins = emg_params[0]
    tact_bins = tact_params[0]
    perc_syns = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
    families = ['Ball', 'Cutlery', 'Geometric', 'Mugs', 'Plates']
    l1VSl2 = [0, 0.25, 0.5, 0.75, 1]
    c_param = [0.01, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5]

    # TEST
    # perc_syns = [100, 90]
    # l1VSl2 = [0, 0.25]
    # c_param = [0.01, 0.1]

    # LOAD EXTRA DATA
    if type == 'early':
        extra_data = pd.read_csv('./results/Early Enclosure/alternative_early_enclosure_extra_data.csv')
    else:
        extra_data = pd.read_csv('./results/Syn/extra_data.csv')

    # SET AND OPEN RESULT FILE
    if type == 'all':
        res_file_name = './results/Syn/accuracy/syn_results'
    elif type == 'clustering':
        res_file_name = './results/Syn/accuracy/subj_clust_syn_results'
    else:  # early enclosure
        res_file_name = './results/Early Enclosure/accuracy/alternative_syn_results'

    if discard == 'less':
        res_file_name += '.csv'
    else:
        res_file_name += '_decr.csv'

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
        emg_score_df = pd.read_csv('./results/Early Enclosure/scores/alternative_reordered_emg_pca_scores.csv', index_col=0)
        tact_score_df = pd.read_csv('./results/Early Enclosure/scores/alternative_reordered_tact_scores.csv', index_col=0)

    # BUILD ITERABLE STRUCTURES
    all_param = list(itertools.product(perc_syns, families, l1VSl2, c_param))
    kin_data_and_iter = [[kin_score_df, extra_data, x, cv, kin_bins, discard] for x in all_param]
    emg_pca_data_and_iter = [[emg_score_df, extra_data, x, cv, emg_bins, discard] for x in all_param]
    tact_data_and_iter = [[tact_score_df, extra_data, x, cv, tact_bins, discard] for x in all_param]

    # multiprocessing
    with Pool() as pool:

        result_kin = pool.map_async(kin_syn_classif, kin_data_and_iter)
        result_emg_pca = pool.map_async(emg_pca_syn_classif, emg_pca_data_and_iter)
        result_tact = pool.map_async(tact_syn_classif, tact_data_and_iter)

        for res_kin in result_kin.get():
            wr.writerow(res_kin)
        # print("Kinematic classification done!")

        for res_emg_pca in result_emg_pca.get():
            wr.writerow(res_emg_pca)
        # print("EMG PCA classification done!")

        for res_tact in result_tact.get():
            wr.writerow(res_tact)
        # print("Tactile classification done!")

    # print("Single source classification done!!")
    result_file.close()


def hierarchical_syn_classification(type, discard):

    families = ['Ball', 'Cutlery', 'Geometric', 'Mugs', 'Plates']
    cv = 3

    [[kin_bins, kin_l1, kin_c], [emg_bins, emg_l1, emg_c], [tact_bins, tact_l1, tact_c]] = get_raw_best_params()

    c_values = [0.01, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5]
    perc_syns = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]

    # SET AND OPEN RESULT FILE
    if type == 'all':
        res_file_name = './results/Syn/accuracy/syn_hier_results'
        best_params_file = './results/Syn/best_syn_params'
    elif type == 'clustering':
        res_file_name = './results/Syn/accuracy/subj_clust_syn_hier_results'
        best_params_file = './results/Syn/subj_clust_best_syn_params'
    else:  # early enclosure
        res_file_name = './results/Early Enclosure/accuracy/alternative_syn_hier_results'
        best_params_file = './results/Early Enclosure/alternative_best_syn_params'

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

            for family in families:

                total_score = []

                kin_dat = kin_scores.loc[kin_scores['Family'] == family]
                emg_dat = emg_scores.loc[emg_scores['Family'] == family]
                tact_dat = tact_scores.loc[tact_scores['Family'] == family]

                to_kfold = kin_dat.drop_duplicates(subset=['Trial num', 'Family'])

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
                        res.extend([family, p, top_c])
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

    kin_dat = kin_scores.loc[kin_scores['Family'] == family]
    emg_dat = emg_scores.loc[emg_scores['Family'] == family]
    tact_dat = tact_scores.loc[tact_scores['Family'] == family]

    to_kfold = kin_dat.drop_duplicates(subset=['Trial num', 'Given Object'])

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=rnd_st)
    # WARNING: the skf.split returns the indexes
    for train, test in skf.split(to_kfold['Trial num'].astype(int),to_kfold['Given Object'].astype(str)):

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
                    train_labels.append(np.unique(ep_kin_data['Given Object'])[0])

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
                    test_labels.append(np.unique(ep_kin_data['Given Object'])[0])

                except RuntimeWarning:
                    # print("Dropped EP", tst_iter, "from family ", family)
                    tst_dropped += 1

        test_kin_df = pd.DataFrame(kin_test_data)
        test_emg_df = pd.DataFrame(emg_test_data)
        test_tact_df = pd.DataFrame(tact_test_data)
        test_df = pd.concat([test_kin_df, test_emg_df, test_tact_df], axis=1)
        test_df.apply(zscore)

        log_model = LogisticRegression(penalty='elasticnet', C=c_param, class_weight='balanced',
                                       random_state=rnd_st, solver='saga', max_iter=25000,
                                       multi_class='ovr',
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


def print_syn_results(type, discard):

    plt.close()
    plt.clf()
    cols = ['Kind', 'Perc', 'Family', 'L1vsL2', 'C', 'Acc', 'Mean']

    # VARIANCE
    if type == 'all':
        kin_var = pd.read_csv('./results/Syn/variance/kin_var.csv')
        kin_var.drop(kin_var.columns[0], axis=1, inplace=True)
        emg_pca_var = pd.read_csv('./results/Syn/variance/emg_pca_var.csv')
        emg_pca_var.drop(emg_pca_var.columns[0], axis=1, inplace=True)
        tact_var = pd.read_csv('./results/Syn/variance/tact_var.csv')
        tact_var.drop(tact_var.columns[0], axis=1, inplace=True)
    elif type == 'clustering':
        kin_var = pd.read_csv('./results/Syn/variance/overall_var_kin.csv')
        kin_var.drop(kin_var.columns[0], axis=1, inplace=True)
        emg_pca_var = pd.read_csv('./results/Syn/variance/overall_var_emg_pca.csv')
        emg_pca_var.drop(emg_pca_var.columns[0], axis=1, inplace=True)
        tact_var = pd.read_csv('./results/Syn/variance/overall_var_tact.csv')
        tact_var.drop(tact_var.columns[0], axis=1, inplace=True)
    elif type == 'alternative':
        kin_var = pd.read_csv('./results/Syn/variance/alternative_overall_var_kin.csv')
        kin_var.drop(kin_var.columns[0], axis=1, inplace=True)
        emg_pca_var = []
        # emg_pca_var.drop(emg_pca_var.columns[0], axis=1, inplace=True)
        tact_var = []
        # tact_var.drop(tact_var.columns[0], axis=1, inplace=True)
    else:  # early enclosure
        kin_var = pd.read_csv('./results/Early Enclosure/variance/alternative_reordered_kin_var_tot.csv')
        kin_var.drop(kin_var.columns[0], axis=1, inplace=True)
        emg_pca_var = pd.read_csv('./results/Early Enclosure/variance/alternative_reordered_emg_pca_var_tot.csv')
        emg_pca_var.drop(emg_pca_var.columns[0], axis=1, inplace=True)
        tact_var = pd.read_csv('./results/Early Enclosure/variance/alternative_reordered_tact_var_tot.csv')
        tact_var.drop(tact_var.columns[0], axis=1, inplace=True)

    # RESUL FILE LOADING
    if type == 'all':
        res_file_name = './results/Syn/accuracy/syn_results'
        multi_file_name = './results/Syn/accuracy/syn_multi_results'
        hier_file_name = './results/Syn/accuracy/syn_hier_results'
    elif type == 'clustering':
        res_file_name = './results/Syn/accuracy/subj_clust_syn_results'
        multi_file_name = './results/Syn/accuracy/subj_clust_syn_multi_results'
        hier_file_name = './results/Syn/accuracy/subj_clust_syn_hier_results'
    elif type == 'alternative':
        res_file_name = './results/Syn/accuracy/fam_alternative_syn_results'
        multi_file_name = []
        hier_file_name = []
    else:  # early enclosure
        res_file_name = './results/Early Enclosure/accuracy/alternative_syn_results'
        multi_file_name = './results/Early Enclosure/accuracy/alternative_syn_multi_results'
        hier_file_name = './results/Early Enclosure/accuracy/alternative_syn_hier_results'

    if discard == 'less':
        res_file_name += '.csv'
        multi_file_name += '.csv'
        hier_file_name += '.csv'
    else:
        res_file_name += '_decr.csv'
        multi_file_name += '_decr.csv'
        hier_file_name += '_decr.csv'

    results_df = pd.read_csv(res_file_name, header=None)
    multi_res_df = pd.read_csv(multi_file_name, header=None)
    hier_res_df = pd.read_csv(hier_file_name, header=None)

    results_df.columns = cols
    multi_cols = ['Kind', 'Family', 'Perc', 'L1vsL2', 'C', 'Acc']
    multi_res_df.columns = multi_cols
    hier_cols = ['Kind', 'Family', 'Perc', 'C', 'Acc']
    hier_res_df.columns = hier_cols

    kin_results_df = results_df.loc[results_df['Kind'] == 'Kin']
    emg_pca_results_df = results_df.loc[results_df['Kind'] == 'EMG PCA']
    tact_results_df = results_df.loc[results_df['Kind'] == 'Tact']
    multi_results_df = multi_res_df.loc[multi_res_df['Kind'] == 'Multimodal']
    hier_results_df = hier_res_df.loc[hier_res_df['Kind'] == 'Hierarchical']

    """VARIANCE CALCULATIONS"""
    perc = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]

    kin_cum_var = []
    emg_pca_cum_var = []
    tact_cum_var = []
    combined_cum_var = []

    for p in perc:
        kin_syns = int(np.ceil(len(kin_var) * p / 100))
        emg_pca_syns = int(np.ceil(len(emg_pca_var) * p / 100))
        tact_syns = int(np.ceil(len(tact_var) * p / 100))

        if discard == 'less':
            kin_cum_var.append(kin_var.iloc[:kin_syns].sum()[0] * 100)
            emg_pca_cum_var.append(emg_pca_var.iloc[:emg_pca_syns].sum()[0] * 100)
            tact_cum_var.append(tact_var.iloc[:tact_syns].sum()[0] * 100)
        else:  # discard most relevant
            kin_cum_var.append(kin_var.iloc[-kin_syns:].sum()[0] * 100)
            emg_pca_cum_var.append(emg_pca_var.iloc[-emg_pca_syns:].sum()[0]*100)
            tact_cum_var.append(tact_var.iloc[-tact_syns:].sum()[0]*100)

    combined_cum_var = [statistics.mean(k) for k in zip(kin_cum_var, emg_pca_cum_var, tact_cum_var)]
    kin_cum_var = pd.DataFrame(kin_cum_var)
    emg_pca_cum_var = pd.DataFrame(emg_pca_cum_var)
    tact_cum_var = pd.DataFrame(tact_cum_var)
    combined_cum_var = pd.DataFrame(combined_cum_var)

    # not sure what have done here
    extended_kin_cum_var = kin_cum_var[:]
    extended_kin_cum_var = np.insert(extended_kin_cum_var, 0, extended_kin_cum_var[0][0])
    extended_emg_pca_cum_var = emg_pca_cum_var[:]
    extended_emg_pca_cum_var = np.insert(extended_emg_pca_cum_var, 0, extended_emg_pca_cum_var[0][0])
    extended_tact_cum_var = tact_cum_var[:]
    extended_tact_cum_var = np.insert(extended_tact_cum_var, 0, extended_tact_cum_var[0][0])
    extended_combined_cum_var = combined_cum_var[:]
    extended_combined_cum_var = np.insert(extended_combined_cum_var, 0, extended_combined_cum_var[0][0])

    ## GET SYNERGIES BEST RESULTS

    perc_values = np.flip(np.unique(results_df['Perc']))
    l1vsl2_values = np.unique(results_df['L1vsL2'])
    c_values = np.unique(results_df['C'])

    kin_best_acc = np.zeros((len(perc_values),5))
    kin_best_params = [[[], []]] * len(perc_values)

    emg_pca_best_acc = np.zeros((len(perc_values),5))
    emg_pca_best_params = [[[], []]] * len(perc_values)

    tact_best_acc = np.zeros((len(perc_values),5))
    tact_best_params = [[[], []]] * len(perc_values)

    multi_best_acc = np.zeros((len(perc_values), 5))
    multi_best_params = [[[], []]] * len(perc_values)

    hier_best_acc = np.zeros((len(perc_values), 5))
    hier_best_params = [[]] * len(perc_values)

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

                multi_sel = multi_results_df.loc[
                    (multi_results_df['Perc'] == perc_values[iter_perc]) & (multi_results_df['L1vsL2'] == l1) & (
                            multi_results_df['C'] == c)]
                multi_sel_mean_acc = multi_sel.groupby('Family')['Acc'].mean()

                if multi_sel_mean_acc.mean() > multi_best_acc[iter_perc].mean():
                    multi_best_acc[iter_perc] = multi_sel_mean_acc
                    multi_best_params[iter_perc] = [l1, c]

                hier_sel = hier_results_df.loc[
                    (hier_results_df['Perc'] == perc_values[iter_perc]) & (hier_results_df['C'] == c)]
                hier_sel_mean_acc = hier_sel.groupby('Family')['Acc'].mean()

                if hier_sel_mean_acc.mean() > hier_best_acc[iter_perc].mean():
                    hier_best_acc[iter_perc] = hier_sel_mean_acc
                    hier_best_params[iter_perc] = c

    # BEST ACCURACIES
    syn_cols = ["Source"]
    syn_cols.extend(perc_values)
    syn_best_acc_df = pd.DataFrame(columns=syn_cols)

    kin_aux_df = pd.DataFrame(data=kin_best_acc.transpose(), columns=perc_values)
    kin_aux_df.insert(0, "Source", ["Kin"] * 5)

    emg_pca_aux_df = pd.DataFrame(data=emg_pca_best_acc.transpose(), columns=perc_values)
    emg_pca_aux_df.insert(0, "Source", ["EMG PCA"] * 5)

    tact_aux_df = pd.DataFrame(data=tact_best_acc.transpose(), columns=perc_values)
    tact_aux_df.insert(0, "Source", ["Tact"] * 5)

    multi_aux_df = pd.DataFrame(data=multi_best_acc.transpose(), columns=perc_values)
    multi_aux_df.insert(0, "Source", ["Multi"] * 5)

    hier_aux_df = pd.DataFrame(data=hier_best_acc.transpose(), columns=perc_values)
    hier_aux_df.insert(0, "Source", ["Hier"] * 5)

    syn_best_acc_df = pd.concat([syn_best_acc_df, kin_aux_df])
    syn_best_acc_df = pd.concat([syn_best_acc_df, emg_pca_aux_df])
    syn_best_acc_df = pd.concat([syn_best_acc_df, tact_aux_df])
    syn_best_acc_df = pd.concat([syn_best_acc_df, multi_aux_df])
    syn_best_acc_df = pd.concat([syn_best_acc_df, hier_aux_df])

    # BEST HYPERPARAMETERS
    # syn_best_param_df = pd.DataFrame(columns=syn_cols)
    #
    # kin_l1c_param = pd.DataFrame(data=[kin_best_params], columns=perc_values)
    # kin_l1c_param.insert(0, "Source", ["Kin"])
    #
    # emg_pca_l1c_param = pd.DataFrame(data=[emg_pca_best_params], columns=perc_values)
    # emg_pca_l1c_param.insert(0, "Source", ["EMG PCA"])
    #
    # tact_l1c_param = pd.DataFrame(data=[tact_best_params], columns=perc_values)
    # tact_l1c_param.insert(0, "Source", ["Tact"])
    #
    # multi_l1c_param = pd.DataFrame(data=[multi_best_params], columns=perc_values)
    # multi_l1c_param.insert(0, "Source", ["Multi"])
    #
    # hier_l1c_param = pd.DataFrame(data=[hier_best_params], columns=perc_values)
    # hier_l1c_param.insert(0, "Source", ["Hier"])
    #
    # syn_best_param_df = pd.concat([syn_best_param_df, kin_l1c_param])
    # syn_best_param_df = pd.concat([syn_best_param_df, emg_pca_l1c_param])
    # syn_best_param_df = pd.concat([syn_best_param_df, tact_l1c_param])
    # syn_best_param_df = pd.concat([syn_best_param_df, multi_l1c_param])
    # syn_best_param_df = pd.concat([syn_best_param_df, hier_l1c_param])


    ## LOAD RAW BEST RESULTS
    raw_cols = ['Kind', 'Family', 'bins', 'L1vsL2', 'C', 'Acc', 'Mean']
    raw_results_df = pd.read_csv('./results/Raw/accuracy/raw_results.csv', header=None)
    raw_results_df.columns = raw_cols

    kin_raw_df = raw_results_df.loc[raw_results_df["Kind"] == "Kin"]
    emg_raw_df = raw_results_df.loc[raw_results_df["Kind"] == "EMG"]
    tact_raw_df = raw_results_df.loc[raw_results_df["Kind"] == "Tactile"]
    multi_raw_df = raw_results_df.loc[raw_results_df["Kind"] == "Multimodal"]
    hier_raw_df = raw_results_df.loc[raw_results_df["Kind"] == "Hierarchical"]

    """BEST RAW PARAMETERS"""
    # FIXED PARAMETERS FROM RAW CLASSIFICATION
    best_kin_param = [40, 0.25, 0.1]
    best_emg_param = [10, 0, 1.5]
    best_tact_param = [5, 0.5, 0.25]
    best_multi_param = [5, 0.75, 0.25]
    best_hier_param = 0.5

    # KIN
    best_raw_kin_results = kin_raw_df.loc[
        (kin_raw_df["bins"] == best_kin_param[0]) & (kin_raw_df["L1vsL2"] == best_kin_param[1]) & (
                kin_raw_df["C"] == best_kin_param[2])]["Mean"]

    best_raw_kin_df = pd.DataFrame()
    for x in range(len(syn_best_acc_df.columns) - 1):
        best_raw_kin_df = pd.concat([best_raw_kin_df, best_raw_kin_results], axis=1)
    # best_raw_kin_df = best_raw_kin_df.transpose()
    best_raw_kin_df.columns = perc_values

    # EMG PCA
    best_raw_emg_results = emg_raw_df.loc[
        (emg_raw_df["bins"] == best_emg_param[0]) & (emg_raw_df["L1vsL2"] == best_emg_param[1]) & (
                emg_raw_df["C"] == best_emg_param[2])]["Mean"]

    best_raw_emg_df = pd.DataFrame()
    for x in range(len(syn_best_acc_df.columns) - 1):
        best_raw_emg_df = pd.concat([best_raw_emg_df, best_raw_emg_results], axis=1)
    # best_raw_emg_df = best_raw_emg_df.transpose()
    best_raw_emg_df.columns = perc_values

    # TACT
    best_raw_tact_results = tact_raw_df.loc[
        (tact_raw_df["bins"] == best_tact_param[0]) & (tact_raw_df["L1vsL2"] == best_tact_param[1]) & (
                tact_raw_df["C"] == best_tact_param[2])]["Mean"]

    best_raw_tact_df = pd.DataFrame()
    for x in range(len(syn_best_acc_df.columns) - 1):
        best_raw_tact_df = pd.concat([best_raw_tact_df, best_raw_tact_results], axis=1)
    # best_raw_tact_df = best_raw_tact_df.transpose()
    best_raw_tact_df.columns = perc_values

    # MULTI
    best_raw_multi_results = multi_raw_df.loc[
        (multi_raw_df["bins"] == best_multi_param[0]) & (multi_raw_df["L1vsL2"] == best_multi_param[1]) & (
                multi_raw_df["C"] == best_multi_param[2])]["Mean"]

    best_raw_multi_df = pd.DataFrame()
    for x in range(len(syn_best_acc_df.columns) - 1):
        best_raw_multi_df = pd.concat([best_raw_multi_df, best_raw_multi_results], axis=1)
    # best_raw_multi_df = best_raw_multi_df.transpose()
    best_raw_multi_df.columns = perc_values

    # HIER
    best_raw_hier_results = hier_raw_df.loc[hier_raw_df["C"] == best_hier_param]["Mean"]

    best_raw_hier_df = pd.DataFrame()
    for x in range(len(syn_best_acc_df.columns) - 1):
        best_raw_hier_df = pd.concat([best_raw_hier_df, best_raw_hier_results], axis=1)
    #best_raw_hier_df = best_raw_hier_df.transpose()
    best_raw_hier_df.columns = perc_values

    """ KIN lines plot """
    sns.pointplot(data=pd.DataFrame(kin_cum_var).transpose(), label='Variance Explained', color='0', scale=.5)
    i = sns.pointplot(data=syn_best_acc_df.loc[syn_best_acc_df["Source"] == "Kin"], errorbar='ci', errwidth='.75', capsize=.2,
                      color="r", label='Syn classifier')
    sns.pointplot(data=best_raw_kin_df, errorbar='ci', errwidth='.75', capsize=.2, color="b", label='Raw classifier')
    i.set(ylabel="Accuracy (95% ci)\nVariance Explained")
    i.set(xlabel="Percentage of Synergies")
    i.axhline(33, color='0.75', linestyle='--', label='Chance level')
    # i.axhline(55.52, color='r', linestyle='--', label='Raw Classifier')
    leg = plt.legend(labels=['Syn classifier', 'Raw classifier', 'Chance Level', 'Variance Explained'], labelcolor=['r', 'b', '0.75', '0'])
    handles = leg.legendHandles
    colors = ['r', 'b', '0.75', '0']
    for it, handle in enumerate(handles):
        handle.set_color(colors[it])
        handle.set_linewidth(1)
    i.set_ylim([0, 100])
    sns.move_legend(i, "best")
    # plt.show()

    if type == 'all':
        title = 'Kinematic accuracy comparison for all subjects'
        fig_file = './results/Syn/plots/kin_drop_syn_acc'
    elif type == 'clustering':
        title = 'Kinematic accuracy comparison for each subject with clustering'
        fig_file = './results/Syn/plots/subj_clust_kin_drop_syn_acc'
    elif type == 'alternative':
        title = 'Kinematic accuracy comparison for each subject with clustering'
        fig_file = './results/Syn/plots/alternative_kin_drop_syn_acc'
    else:  # early enclosure
        title = 'Kinematic accuracy comparison for early enclosure'
        fig_file = './results/Early Enclosure/plots/kin_drop_syn_acc'

    if discard == 'less':
        title += '\ndiscarding less relevant synergies'
        fig_file += '.png'
    else:  # discard the most
        title += '\ndiscarding most relevant synergies'
        fig_file += '_decr.png'

    i.set(title=title)
    # plt.savefig(fig_file, dpi=600)
    plt.close()
    plt.clf()

    """ KIN bar pval plot """
    kin_pval_df = pd.DataFrame(syn_best_acc_df.iloc[:,1:].loc[syn_best_acc_df["Source"] == "Kin"])
    kin_pval_df.insert(0, 'Raw', best_raw_kin_df.iloc[:,0].values)
    plt.figure()
    sns.pointplot(data=pd.DataFrame(extended_kin_cum_var).transpose(), label='Variance Explained', color='r', scale=.5)
    i = sns.barplot(data=kin_pval_df)
    # pairs_kin = [('Raw', '100'), ('Raw', '90'), ('Raw', '80'), ('Raw', '70'), ('Raw', '60'), ('Raw', '50'), ('Raw', '40'), ('Raw', '30'), ('Raw', '20'), ('Raw', '10')]
    pairs_kin = [('Raw', 100), ('Raw', 90), ('Raw', 80), ('Raw', 70), ('Raw', 60), ('Raw', 50),
                 ('Raw', 40), ('Raw', 30), ('Raw', 20), ('Raw', 10)]
    annotator_i = Annotator(i, pairs_kin[::-1], data=kin_pval_df)
    annotator_i.configure(test="Mann-Whitney", text_format="star", show_test_name=False, fontsize='xx-small', line_height=0.01, line_width=0.5, color='0')
    annotator_i.apply_and_annotate()

    i.axhline(33, color='b', linestyle='--', label='Chance level')
    i.set(ylabel="Accuracy (95% ci)")
    leg = plt.legend(labels=['Chance Level', 'Variance Explained'], labelcolor=['b', 'r'])
    handles = leg.legendHandles
    colors = ['b', 'r']
    for it, handle in enumerate(handles):
        handle.set_color(colors[it])
        handle.set_linewidth(1)
    sns.move_legend(i, "center right")
    # i.set(xlabel=None)
    # plt.xticks(rotation=45, size=4)
    # # i.axhline(20, color='r')

    if type == 'all':
        title = 'Kinematic accuracy comparison for all subjects'
        fig_file = './results/Syn/plots/kin_drop_syn_acc_pval'
    elif type == 'clustering':
        title = 'Kinematic accuracy comparison for each subject with clustering'
        fig_file = './results/Syn/plots/subj_clust_kin_drop_syn_acc_pval'
    elif type == 'alternative':
        title = 'Kinematic accuracy comparison for each subject with clustering'
        fig_file = './results/Syn/plots/alternative_kin_drop_syn_acc_pval'
    else:  # early enclosure
        title = 'Kinematic accuracy comparison for early enclosure'
        fig_file = './results/Early Enclosure/plots/kin_drop_syn_acc_pval'

    if discard == 'less':
        title += '\ndiscarding less relevant synergies'
        fig_file += '.png'
    else:  # discard the most
        title += '\ndiscarding most relevant synergies'
        fig_file += '_decr.png'

    i.set(title=title)
    i.set_ylim(0, 200)
    plt.savefig(fig_file, dpi=600)
    # plt.show()
    plt.close()
    plt.clf()

    """  EMG PCA plot """
    sns.pointplot(data=pd.DataFrame(emg_pca_cum_var).transpose(), label='Variance Explained', color='0', scale=.5)
    j = sns.pointplot(data=syn_best_acc_df.loc[syn_best_acc_df["Source"] == "EMG PCA"], errorbar='ci', errwidth='.75', capsize=.2,
                      color="r", label='Syn classifier')
    sns.pointplot(data=best_raw_emg_df, errorbar='ci', errwidth='.75', capsize=.2, color="b", label='Raw classifier')
    j.set(ylabel="Accuracy (95% ci)")
    j.set(xlabel="Percentage of Synergies")
    j.axhline(33, color='0.75', linestyle='--', label='Chance level')
    # j.axhline(55.52, color='r', linestyle='--', label='Raw Classifier')
    leg = plt.legend(labels=['Syn classifier', 'Raw classifier', 'Chance Level', 'Variance Explained'],
                     labelcolor=['r', 'b', '0.75', '0'])
    handles = leg.legendHandles
    colors = ['r', 'b', '0.75', '0']
    for it, handle in enumerate(handles):
        handle.set_color(colors[it])
        handle.set_linewidth(1)
    j.set_ylim([0, 100])
    sns.move_legend(j, "best")
    # plt.show()

    if type == 'all':
        title = 'EMG PCA accuracy comparison for all subjects'
        fig_file = './results/Syn/plots/emg_pca_drop_syn_acc'
    elif type == 'clustering':
        title = 'EMG PCA accuracy comparison for each subject with clustering'
        fig_file = './results/Syn/plots/subj_clust_emg_pca_drop_syn_acc'
    else:  # early enclosure
        title = 'EMG PCA accuracy comparison for early enclosure'
        fig_file = './results/Early Enclosure/plots/emg_pca_drop_syn_acc'

    if discard == 'less':
        title += '\ndiscarding less relevant synergies'
        fig_file += '.png'
    else:  # discard the most
        title += '\ndiscarding most relevant synergies'
        fig_file += '_decr.png'

    j.set(title=title)
    plt.savefig(fig_file, dpi=600)
    plt.close()
    plt.clf()

    """ EMG PCA bar pval plot """
    emg_pca_pval_df = pd.DataFrame(syn_best_acc_df.iloc[:, 1:].loc[syn_best_acc_df["Source"] == "EMG PCA"])
    emg_pca_pval_df.insert(0, 'Raw', best_raw_emg_df.iloc[:,0].values)
    plt.figure()
    sns.pointplot(data=pd.DataFrame(extended_emg_pca_cum_var).transpose(), label='Variance Explained', color='r', scale=.5)
    i = sns.barplot(data=emg_pca_pval_df)
    # pairs_emg_pca = [('Raw', '100'), ('Raw', '90'), ('Raw', '80'), ('Raw', '70'), ('Raw', '60'), ('Raw', '50'), ('Raw', '40'), ('Raw', '30'), ('Raw', '20'), ('Raw', '10')]
    pairs_emg_pca = [('Raw', 100), ('Raw', 90), ('Raw', 80), ('Raw', 70), ('Raw', 60), ('Raw', 50),
                     ('Raw', 40), ('Raw', 30), ('Raw', 20), ('Raw', 10)]
    annotator_i = Annotator(i, pairs_emg_pca, data=emg_pca_pval_df)
    annotator_i.configure(test="Mann-Whitney", text_format="star", show_test_name=False, fontsize='xx-small', line_height=0.01, line_width=0.5, color='0')
    annotator_i.apply_and_annotate()
    i.set(ylabel="Accuracy (95% ci)")
    i.axhline(33, color='b', linestyle='--', label='Chance level')
    leg = plt.legend(labels=['Chance Level', 'Variance Explained'], labelcolor=['b', 'r'])
    handles = leg.legendHandles
    colors = ['b', 'r']
    for it, handle in enumerate(handles):
        handle.set_color(colors[it])
        handle.set_linewidth(1)
    sns.move_legend(i, "center right")

    if type == 'all':
        title = 'EMG PCA accuracy comparison for all subjects'
        fig_file = './results/Syn/plots/emg_pca_drop_syn_acc_pval'
    elif type == 'clustering':
        title = 'EMG PCA accuracy comparison for each subject with clustering'
        fig_file = './results/Syn/plots/subj_clust_emg_pca_drop_syn_acc_pval'
    else:  # early enclosure
        title = 'EMG PCA accuracy comparison for early enclosure'
        fig_file = './results/Early Enclosure/plots/emg_pca_drop_syn_acc_pval'

    if discard == 'less':
        title += '\ndiscarding less relevant synergies'
        fig_file += '.png'
    else:  # discard the most
        title += '\ndiscarding most relevant synergies'
        fig_file += '_decr.png'

    i.set(title=title)
    i.set_ylim(0, 200)
    plt.savefig(fig_file, dpi=600)
    plt.close()
    plt.clf()

    """ TACT plot """
    sns.pointplot(data=pd.DataFrame(tact_cum_var).transpose(), label='Variance Explained', color='0', scale=.5)
    k = sns.pointplot(data=syn_best_acc_df.loc[syn_best_acc_df["Source"] == "Tact"], errorbar='ci', errwidth='.75', capsize=.2,
                      color="r", label='Syn classifier')
    sns.pointplot(data=best_raw_tact_df, errorbar='ci', errwidth='.75', capsize=.2, color="b", label='Raw classifier')
    k.set(ylabel="Accuracy (95% ci)")
    k.set(xlabel="Percentage of Synergies")
    k.axhline(33, color='0.75', linestyle='--', label='Chance level')
    # k.axhline(55.52, color='r', linestyle='--', label='Raw Classifier')
    leg = plt.legend(labels=['Syn classifier', 'Raw classifier', 'Chance Level', 'Variance Explained'],
                     labelcolor=['r', 'b', '0.75', '0'])
    handles = leg.legendHandles
    colors = ['r', 'b', '0.75', '0']
    for it, handle in enumerate(handles):
        handle.set_color(colors[it])
        handle.set_linewidth(1)
    k.set_ylim([0, 100])
    sns.move_legend(k, "best")
    # plt.show()

    if type == 'all':
        title = 'Tactile accuracy comparison for all subjects'
        fig_file = './results/Syn/plots/tact_drop_syn_acc'
    elif type == 'clustering':
        title = 'Tactile accuracy comparison for each subject with clustering'
        fig_file = './results/Syn/plots/subj_clust_tact_drop_syn_acc'
    else:  # early enclosure
        title = 'Tactile accuracy comparison for early enclosure'
        fig_file = './results/Early Enclosure/plots/tact_drop_syn_acc'

    if discard == 'less':
        title += '\ndiscarding less relevant synergies'
        fig_file += '.png'
    else:  # discard the most
        title += '\ndiscarding most relevant synergies'
        fig_file += '_decr.png'

    k.set(title=title)
    plt.savefig(fig_file, dpi=600)
    plt.close()
    plt.clf()

    """ TACT bar pval plot """
    tact_pval_df = pd.DataFrame(syn_best_acc_df.iloc[:, 1:].loc[syn_best_acc_df["Source"] == "Tact"])
    tact_pval_df.insert(0, 'Raw', best_raw_tact_df.iloc[:, 0].values)
    plt.figure()
    sns.pointplot(data=pd.DataFrame(extended_tact_cum_var).transpose(), label='Variance Explained', color='r', scale=.5)
    i = sns.barplot(data=tact_pval_df)
    # pairs_tact = [('Raw', '100'), ('Raw', '90'), ('Raw', '80'), ('Raw', '70'), ('Raw', '60'), ('Raw', '50'), ('Raw', '40'), ('Raw', '30'), ('Raw', '20'), ('Raw', '10')]
    pairs_tact = [('Raw', 100), ('Raw', 90), ('Raw', 80), ('Raw', 70), ('Raw', 60), ('Raw', 50),
                  ('Raw', 40), ('Raw', 30), ('Raw', 20), ('Raw', 10)]
    annotator_i = Annotator(i, pairs_tact, data=tact_pval_df)
    annotator_i.configure(test="Mann-Whitney", text_format="star", show_test_name=False, fontsize='xx-small', line_height=0.01, line_width=0.5, color='0')
    annotator_i.apply_and_annotate()
    i.set(ylabel="Accuracy (95% ci)")
    i.axhline(33, color='b', linestyle='--', label='Chance level')
    leg = plt.legend(labels=['Chance Level', 'Variance Explained'], labelcolor=['b', 'r'])
    handles = leg.legendHandles
    colors = ['b', 'r']
    for it, handle in enumerate(handles):
        handle.set_color(colors[it])
        handle.set_linewidth(1)
    sns.move_legend(i, "center right")
    # i.set(xlabel=None)
    # plt.xticks(rotation=45, size=4)
    # # i.axhline(20, color='r')

    if type == 'all':
        title = 'Tactile accuracy comparison for all subjects'
        fig_file = './results/Syn/plots/tact_drop_syn_acc_pval'
    elif type == 'clustering':
        title = 'Tactile accuracy comparison for each subject with clustering'
        fig_file = './results/Syn/plots/subj_clust_tact_drop_syn_acc_pval'
    else:  # early enclosure
        title = 'Tactile accuracy comparison for early enclosure'
        fig_file = './results/Early Enclosure/plots/tact_drop_syn_acc_pval'

    if discard == 'less':
        title += '\ndiscarding less relevant synergies'
        fig_file += '.png'
    else:  # discard the most
        title += '\ndiscarding most relevant synergies'
        fig_file += '_decr.png'

    i.set(title=title)
    i.set_ylim(0, 200)
    plt.savefig(fig_file, dpi=600)
    plt.close()
    plt.clf()

    """ MULTI plot """
    sns.pointplot(data=pd.DataFrame(combined_cum_var).transpose(), label='Variance Explained', color='0', scale=.5)
    k = sns.pointplot(data=syn_best_acc_df.loc[syn_best_acc_df["Source"] == "Multi"], errorbar='ci', errwidth='.75', capsize=.2,
                      color="r", label='Syn classifier')
    sns.pointplot(data=best_raw_multi_df, errorbar='ci', errwidth='.75', capsize=.2, color="b", label='Raw classifier')
    k.set(ylabel="Accuracy (95% ci)")
    k.set(xlabel="Percentage of Synergies")
    k.axhline(33, color='0.75', linestyle='--', label='Chance level')
    # k.axhline(55.52, color='r', linestyle='--', label='Raw Classifier')
    leg = plt.legend(labels=['Syn classifier', 'Raw classifier', 'Chance Level', 'Variance Explained'],
                     labelcolor=['r', 'b', '0.75', '0'])
    handles = leg.legendHandles
    colors = ['r', 'b', '0.75', '0']
    for it, handle in enumerate(handles):
        handle.set_color(colors[it])
        handle.set_linewidth(1)
    k.set_ylim([0, 100])
    sns.move_legend(k, "best")
    # plt.show()

    if type == 'all':
        title = 'Multimodal accuracy comparison for all subjects'
        fig_file = './results/Syn/plots/multi_drop_syn_acc'
    elif type == 'clustering':
        title = 'Multimodal accuracy comparison for each subject with clustering'
        fig_file = './results/Syn/plots/subj_clust_multi_drop_syn_acc'
    else:  # early enclosure
        title = 'Multimodal accuracy comparison for early enclosure'
        fig_file = './results/Early Enclosure/plots/multi_drop_syn_acc'

    if discard == 'less':
        title += '\ndiscarding less relevant synergies'
        fig_file += '.png'
    else:  # discard the most
        title += '\ndiscarding most relevant synergies'
        fig_file += '_decr.png'

    k.set(title=title)
    plt.savefig(fig_file, dpi=600)
    plt.close()
    plt.clf()

    """ MULTI bar pval plot """
    multi_pval_df = pd.DataFrame(syn_best_acc_df.iloc[:, 1:].loc[syn_best_acc_df["Source"] == "Multi"])
    multi_pval_df.insert(0, 'Raw', best_raw_multi_df.iloc[:, 0].values)
    plt.figure()
    sns.pointplot(data=pd.DataFrame(extended_combined_cum_var).transpose(), label='Variance Explained', color='r', scale=.5)
    i = sns.barplot(data=multi_pval_df)
    # pairs_multi = [('Raw', '100'), ('Raw', '90'), ('Raw', '80'), ('Raw', '70'), ('Raw', '60'), ('Raw', '50'), ('Raw', '40'), ('Raw', '30'), ('Raw', '20'), ('Raw', '10')]
    pairs_multi = [('Raw', 100), ('Raw', 90), ('Raw', 80), ('Raw', 70), ('Raw', 60), ('Raw', 50),
                   ('Raw', 40), ('Raw', 30), ('Raw', 20), ('Raw', 10)]
    annotator_i = Annotator(i, pairs_multi, data=multi_pval_df)
    annotator_i.configure(test="Mann-Whitney", text_format="star", show_test_name=False, fontsize='xx-small', line_height=0.01, line_width=0.5, color='0')
    annotator_i.apply_and_annotate()
    i.set(ylabel="Accuracy (95% ci)")
    i.axhline(33, color='b', linestyle='--', label='Chance level')
    leg = plt.legend(labels=['Chance Level', 'Variance Explained'], labelcolor=['b', 'r'])
    handles = leg.legendHandles
    colors = ['b', 'r']
    for it, handle in enumerate(handles):
        handle.set_color(colors[it])
        handle.set_linewidth(1)
    sns.move_legend(i, "center right")
    i.set(xlabel=None)
    plt.xticks(rotation=45, size=4)
    # i.axhline(20, color='r')

    if type == 'all':
        title = 'Multimodal accuracy comparison for all subjects'
        fig_file = './results/Syn/plots/multi_drop_syn_acc_pval'
    elif type == 'clustering':
        title = 'Multimodal accuracy comparison for each subject with clustering'
        fig_file = './results/Syn/plots/subj_clust_multi_drop_syn_acc_pval'
    else:  # early enclosure
        title = 'Multimodal accuracy comparison for early enclosure'
        fig_file = './results/Early Enclosure/plots/multi_drop_syn_acc_pval'

    if discard == 'less':
        title += '\ndiscarding less relevant synergies'
        fig_file += '.png'
    else:  # discard the most
        title += '\ndiscarding most relevant synergies'
        fig_file += '_decr.png'

    i.set(title=title)
    i.set_ylim(0, 200)
    plt.savefig(fig_file, dpi=600)
    plt.close()
    plt.clf()

    """ HIER plot """
    sns.pointplot(data=pd.DataFrame(combined_cum_var).transpose(), label='Variance Explained', color='0', scale=.5)
    k = sns.pointplot(data=syn_best_acc_df.loc[syn_best_acc_df["Source"] == "Hier"], errorbar='ci', errwidth='.75',
                      capsize=.2,
                      color="r", label='Syn classifier')
    sns.pointplot(data=best_raw_hier_df, errorbar='ci', errwidth='.75', capsize=.2, color="b",
                  label='Raw classifier')
    k.set(ylabel="Accuracy (95% ci)")
    k.set(xlabel="Percentage of Synergies")
    k.axhline(33, color='0.75', linestyle='--', label='Chance level')
    # k.axhline(55.52, color='r', linestyle='--', label='Raw Classifier')
    leg = plt.legend(labels=['Syn classifier', 'Raw classifier', 'Chance Level', 'Variance Explained'],
                     labelcolor=['r', 'b', '0.75', '0'])
    handles = leg.legendHandles
    colors = ['r', 'b', '0.75', '0']
    for it, handle in enumerate(handles):
        handle.set_color(colors[it])
        handle.set_linewidth(1)
    k.set_ylim([0, 100])
    sns.move_legend(k, "best")
    # plt.show()

    if type == 'all':
        title = 'Hierarchical accuracy comparison for all subjects'
        fig_file = './results/Syn/plots/hier_drop_syn_acc'
    elif type == 'clustering':
        title = 'Hierarchical accuracy comparison for each subject with clustering'
        fig_file = './results/Syn/plots/subj_clust_hier_drop_syn_acc'
    else:  # early enclosure
        title = 'Hierarchical accuracy comparison for early enclosure'
        fig_file = './results/Early Enclosure/plots/hier_drop_syn_acc'

    if discard == 'less':
        title += '\ndiscarding less relevant synergies'
        fig_file += '.png'
    else:  # discard the most
        title += '\ndiscarding most relevant synergies'
        fig_file += '_decr.png'

    k.set(title=title)
    plt.savefig(fig_file, dpi=600)
    plt.close()
    plt.clf()

    """ HIER bar pval plot """
    hier_pval_df = pd.DataFrame(syn_best_acc_df.iloc[:, 1:].loc[syn_best_acc_df["Source"] == "Hier"])
    hier_pval_df.insert(0, 'Raw', best_raw_hier_df.iloc[:, 0].values)
    plt.figure()
    sns.pointplot(data=pd.DataFrame(extended_combined_cum_var).transpose(), label='Variance Explained', color='r', scale=.5)
    i = sns.barplot(data=hier_pval_df)
    # pairs_hier = [('Raw', '100'), ('Raw', '90'), ('Raw', '80'), ('Raw', '70'), ('Raw', '60'), ('Raw', '50'), ('Raw', '40'), ('Raw', '30'), ('Raw', '20'), ('Raw', '10')]
    pairs_hier = [('Raw', 100), ('Raw', 90), ('Raw', 80), ('Raw', 70), ('Raw', 60), ('Raw', 50),
                  ('Raw', 40), ('Raw', 30), ('Raw', 20), ('Raw', 10)]
    annotator_i = Annotator(i, pairs_hier, data=hier_pval_df)
    annotator_i.configure(test="Mann-Whitney", text_format="star", show_test_name=False, fontsize='xx-small', line_height=0.01, line_width=0.5, color='0')
    annotator_i.apply_and_annotate()
    i.set(ylabel="Accuracy (95% ci)")
    i.axhline(33, color='b', linestyle='--', label='Chance level')
    i.set(xlabel=None)
    plt.xticks(rotation=45, size=4)
    leg = plt.legend(labels=['Chance Level', 'Variance Explained'], labelcolor=['b', 'r'])
    handles = leg.legendHandles
    colors = ['b', 'r']
    for it, handle in enumerate(handles):
        handle.set_color(colors[it])
        handle.set_linewidth(1)
    sns.move_legend(i, "center right")
    # i.set(xlabel=None)
    # plt.xticks(rotation=45, size=4)
    # # i.axhline(20, color='r')

    if type == 'all':
        title = 'Hierarchical accuracy comparison for all subjects'
        fig_file = './results/Syn/plots/hier_drop_syn_acc_pval'
    elif type == 'clustering':
        title = 'Hierarchical accuracy comparison for each subject with clustering'
        fig_file = './results/Syn/plots/subj_clust_hier_drop_syn_acc_pval'
    else:  # early enclosure
        title = 'Hierarchical accuracy comparison for early enclosure'
        fig_file = './results/Early Enclosure/plots/hier_drop_syn_acc_pval'

    if discard == 'less':
        title += '\ndiscarding less relevant synergies'
        fig_file += '.png'
    else:  # discard the most
        title += '\ndiscarding most relevant synergies'
        fig_file += '_decr.png'

    i.set(title=title)
    i.set_ylim(0, 200)
    plt.savefig(fig_file, dpi=600)
    plt.close()
    plt.clf()


def early_fine_vs_coarse():

    plt.close()
    plt.clf()

    # open scores
    kin_score_df = pd.read_csv('./results/Early Enclosure/scores/alternative_reordered_kin_scores.csv', index_col=0)

    selected_df = kin_score_df[['0', '1', '2', '3', '15', '16', '17', '18']].abs()

    i = sns.boxplot(data=selected_df)
    i.set(title='Kinematic score comparison Low vs. High order synergies\nfor Early Enclosure')
    i.set(ylabel="Z-score (absolute value)")
    i.set(xlabel="Synergies")
    # plt.show()
    plt.savefig('./results/Early Enclosure/plots/fine_vs_coarse.png', dpi=600)


def syn_fine_vs_coarse_fam(type):

    if type == 'cluster':
        kin_score_df = pd.read_csv('./results/Syn/scores/reordered_kin_scores.csv', index_col=0)
    else:
        kin_score_df = pd.read_csv('./results/Syn/scores/kin_scores.csv', index_col=0)

    extra_data = pd.read_csv('./results/Syn/extra_data.csv')

    fine_families = ['Cutlery', 'Geometric']
    coarse_families = ['Ball', 'Plates']
    fine_idx = extra_data.loc[extra_data['Family'].isin(fine_families)]
    coarse_idx = extra_data.loc[extra_data['Family'].isin(coarse_families)]

    fine_df = kin_score_df.iloc[fine_idx.index]
    selected_fine = fine_df[['0', '1', '2', '3', '15', '16', '17', '18']].abs()

    coarse_df = kin_score_df.iloc[coarse_idx.index]
    selected_coarse = coarse_df[['0', '1', '2', '3', '15', '16', '17', '18']].abs()

    selected_fine['Condition'] = 'Fine'
    selected_coarse['Condition'] = 'Coarse'
    combined_df = pd.concat([selected_fine, selected_coarse], ignore_index=True)

    # Melt the DataFrame for seaborn plotting
    melted_df = combined_df.melt(id_vars='Condition', var_name='Column', value_name='Value')

    # Step 2: Plotting
    plt.close()
    plt.clf()
    plt.figure(figsize=(12, 6))
    ax = sns.violinplot(x='Column', y='Value', hue='Condition', data=melted_df, split=True, inner='quartile')
    ax.legend(loc='center right', title='Condition')
    plt.xlabel("Synergies")
    plt.ylabel("Z-score (absolute value)")

    # Step 3: Statistical Test and Step 4: Annotation
    # Perform statistical tests, check direction, and annotate
    columns = ['0', '1', '2', '3', '15', '16', '17', '18']
    for i, column in enumerate(columns):
        groupA = selected_fine[column]  # Fine group
        groupB = selected_coarse[column]  # Coarse group
        stat, p_value = mannwhitneyu(groupA, groupB, alternative='two-sided')

        # Check for significance
        if p_value < 0.05:
            # Set color based on the direction of the difference
            if column in ['0', '1', '2', '3']:  # Expecting groupB > groupA
                color = 'black' if groupB.median() > groupA.median() else 'red'
            else:  # For columns 15-18, expecting groupA > groupB
                color = 'black' if groupA.median() > groupB.median() else 'red'

            # Annotate the plot with colored asterisks
            plt.text(i, melted_df['Value'].max() * 1.05, '*', ha='center', fontsize=14, color=color)

    tit = 'Kinematic score comparison Low vs. High order synergies\nfor Fine vs. Coarse families'
    if type == 'cluster':
        tit += '. Each subject + clustering'
    else:
        tit += '. All subjects'
    plt.title(tit)
    plt.tight_layout()
    # plt.show()

    if type == 'cluster':
        plt.savefig('./results/Syn/plots/clust_fine_vs_coarse_fam.png', dpi=600)
    else:
        plt.savefig('./results/Syn/plots/all_fine_vs_coarse_fam.png', dpi=600)


def syn_fine_vs_coarse_ep(type):

    if type == 'cluster':
        kin_score_df = pd.read_csv('./results/Syn/scores/reordered_kin_scores.csv', index_col=0)
    else:
        kin_score_df = pd.read_csv('./results/Syn/scores/kin_scores.csv', index_col=0)

    extra_data = pd.read_csv('./results/Syn/extra_data.csv')

    fine_eps = ['edge following', 'function test']
    coarse_eps = ['enclosure', 'weighting', 'pressure', 'enclosure part']
    fine_idx = extra_data.loc[extra_data['EP'].isin(fine_eps)]
    coarse_idx = extra_data.loc[extra_data['EP'].isin(coarse_eps)]

    fine_df = kin_score_df.iloc[fine_idx.index]
    selected_fine = fine_df[['0', '1', '2', '3', '15', '16', '17', '18']].abs()

    coarse_df = kin_score_df.iloc[coarse_idx.index]
    selected_coarse = coarse_df[['0', '1', '2', '3', '15', '16', '17', '18']].abs()

    selected_fine['Condition'] = 'Fine'
    selected_coarse['Condition'] = 'Coarse'
    combined_df = pd.concat([selected_fine, selected_coarse], ignore_index=True)

    # Melt the DataFrame for seaborn plotting
    melted_df = combined_df.melt(id_vars='Condition', var_name='Column', value_name='Value')

    # Step 2: Plotting
    plt.close()
    plt.clf()
    plt.figure(figsize=(12, 6))
    ax = sns.violinplot(x='Column', y='Value', hue='Condition', data=melted_df, split=True, inner='quartile')
    ax.legend(loc='center right', title='Condition')
    plt.xlabel("Synergies")
    plt.ylabel("Z-score (absolute value)")

    # Step 3: Statistical Test and Step 4: Annotation
    # Perform statistical tests, check direction, and annotate
    columns = ['0', '1', '2', '3', '15', '16', '17', '18']
    for i, column in enumerate(columns):
        groupA = selected_fine[column]  # Fine group
        groupB = selected_coarse[column]  # Coarse group
        stat, p_value = mannwhitneyu(groupA, groupB, alternative='two-sided')

        # Check for significance
        if p_value < 0.05:
            # Set color based on the direction of the difference
            if column in ['0', '1', '2', '3']:  # Expecting groupB > groupA
                color = 'black' if groupB.median() > groupA.median() else 'red'
            else:  # For columns 15-18, expecting groupA > groupB
                color = 'black' if groupA.median() > groupB.median() else 'red'

            # Annotate the plot with colored asterisks
            plt.text(i, melted_df['Value'].max() * 1.05, '*', ha='center', fontsize=14, color=color)

    tit = 'Kinematic score comparison Low vs. High order synergies\nfor Fine vs. Coarse EPs'
    if type == 'cluster':
        tit += '. Each subject + clustering'
    else:
        tit += '. All subjects'
    plt.title(tit)
    plt.tight_layout()
    # plt.show()

    if type == 'cluster':
        plt.savefig('./results/Syn/plots/clust_fine_vs_coarse_ep.png', dpi=600)
    else:
        plt.savefig('./results/Syn/plots/all_fine_vs_coarse_ep.png', dpi=600)


def calculate_centroid_distances(df, group_column):

    aux_df = df.iloc[:, 0:19]
    aux_df[group_column] = df[group_column]
    centroids = aux_df.groupby(group_column).mean()  # Calculate centroids

    distances = []
    # Iterate over each group to calculate distances from points to their group's centroid
    for group, group_df in df.groupby(group_column):
        group_centroid = centroids.loc[group]  # Get the centroid for the current group
        group_distances = np.sqrt(
            ((group_df.iloc[:, 0:19] - group_centroid) ** 2).sum(axis=1))  # Euclidean distance to centroid
        distances.extend(group_distances)

    # Add the distances back to the original DataFrame (or to a new one if preferred)
    df['Distance'] = distances
    return df[[group_column, 'Distance']]


def cohen_d(x, y):

    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2

    return (mean(x) - mean(y)) / sqrt(((nx-1)*std(x, ddof=1) ** 2 + (ny-1)*std(y, ddof=1) ** 2) / dof)


def distances(type):

    if type == 'cluster':
        score_file = './results/Syn/scores/reordered_kin_scores.csv'
    else:
        score_file = './results/Syn/scores/kin_scores.csv'

    kin_scores_df = pd.read_csv(score_file)
    kin_scores_df = kin_scores_df.drop(kin_scores_df.columns[0], axis=1)

    extra_data_df = pd.read_csv('./results/Syn/extra_data.csv')

    # Assuming 'extra_data_df' and 'kin_scores_df' are already loaded and have the same length
    combined_df = kin_scores_df.join(extra_data_df)

    intra_subject_distances = calculate_centroid_distances(combined_df.copy(), 'Subject')
    intra_EP_distances = calculate_centroid_distances(combined_df.copy(), 'EP')

    distances_df = pd.concat([
        intra_subject_distances.rename(columns={'Distance': 'Value'}).assign(Type='Intra-Subject'),
        intra_EP_distances.rename(columns={'Distance': 'Value'}).assign(Type='Intra-EP')
    ])

    # Statistical Analysis
    stat, p_value = mannwhitneyu(
        distances_df[distances_df['Type'] == 'Intra-Subject']['Value'],
        distances_df[distances_df['Type'] == 'Intra-EP']['Value']
    )

    # Calculate effect size
    d = cohen_d(distances_df[distances_df['Type'] == 'Intra-Subject']['Value'],
                distances_df[distances_df['Type'] == 'Intra-EP']['Value'])

    # print(f"Cohen's d: {d}")

    # Plotting
    plt.close()
    plt.clf()
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Type', y='Value', data=distances_df)

    tit = 'Intra-Subject vs. Intra-EP Distances'
    if type == 'cluster':
        tit += '. Each subject + clustering'
    else:
        tit += '. All subjects'

    plt.title(tit)
    plt.ylabel('Euclidean distance to Centroid (z-score)')

    # Calculate the maximum value for y-axis to place annotations correctly
    max_y_value = distances_df['Value'].max()
    # Annotate with the p-value
    plt.text(0.5, max_y_value * 0.97, f'p-value: {p_value:.2e}', ha='center', va='bottom', fontsize=10)
    # Annotate with Cohen's d
    plt.text(0.5, max_y_value * 0.94, f"Cohen's d: {d:.3f}", ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    # plt.show()

    save_file = './results/Syn/plots/'
    if type == 'cluster':
        save_file += 'clust_subj_vs_ep.png'
    else:
        save_file += 'all_subj_vs_ep.png'

    plt.savefig(save_file, dpi=600)

def print_syn_results_alternative(discard):

    plt.close()
    plt.clf()
    cols = ['Kind', 'Perc', 'Family', 'L1vsL2', 'C', 'Acc', 'Mean']

    kin_var = pd.read_csv('./results/Syn/variance/alternative_overall_var_kin.csv')
    kin_var.drop(kin_var.columns[0], axis=1, inplace=True)

    res_file_name = './results/Syn/accuracy/fam_alternative_syn_results'

    if discard == 'less':
        res_file_name += '.csv'
    else:
        res_file_name += '_decr.csv'

    results_df = pd.read_csv(res_file_name, header=None)
    results_df.columns = cols
    kin_results_df = results_df.loc[results_df['Kind'] == 'Kin']

    """VARIANCE CALCULATIONS"""
    perc = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]

    kin_cum_var = []

    for p in perc:
        kin_syns = int(np.ceil(len(kin_var) * p / 100))

        if discard == 'less':
            kin_cum_var.append(kin_var.iloc[:kin_syns].sum()[0] * 100)
        else:  # discard most relevant
            kin_cum_var.append(kin_var.iloc[-kin_syns:].sum()[0] * 100)

    kin_cum_var = pd.DataFrame(kin_cum_var)
    # not sure what have done here
    extended_kin_cum_var = kin_cum_var[:]
    extended_kin_cum_var = np.insert(extended_kin_cum_var, 0, extended_kin_cum_var[0][0])

    ## GET SYNERGIES BEST RESULTS

    perc_values = np.flip(np.unique(results_df['Perc']))
    l1vsl2_values = np.unique(results_df['L1vsL2'])
    c_values = np.unique(results_df['C'])

    kin_best_acc = np.zeros((len(perc_values), 5))
    kin_best_params = [[[], []]] * len(perc_values)

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

    # BEST ACCURACIES
    syn_cols = ["Source"]
    syn_cols.extend(perc_values)
    syn_best_acc_df = pd.DataFrame(columns=syn_cols)

    kin_aux_df = pd.DataFrame(data=kin_best_acc.transpose(), columns=perc_values)
    kin_aux_df.insert(0, "Source", ["Kin"] * 5)

    syn_best_acc_df = pd.concat([syn_best_acc_df, kin_aux_df])

    ## LOAD RAW BEST RESULTS
    raw_cols = ['Kind', 'Family', 'bins', 'L1vsL2', 'C', 'Acc', 'Mean']
    raw_results_df = pd.read_csv('./results/Raw/accuracy/raw_fam_results.csv', header=None)
    raw_results_df.columns = raw_cols

    kin_raw_df = raw_results_df.loc[raw_results_df["Kind"] == "Kin"]

    """BEST RAW PARAMETERS"""
    bt_prm = get_best_params_single('clustered', 'less')
    # FIXED PARAMETERS FROM RAW CLASSIFICATION
    # best_kin_param = [40, 0.25, 0.1]
    # best_emg_param = [10, 0, 1.5]
    # best_tact_param = [5, 0.5, 0.25]
    # best_multi_param = [5, 0.75, 0.25]
    # best_hier_param = 0.5
    #
    # # KIN
    # best_raw_kin_results = kin_raw_df.loc[
    #     (kin_raw_df["bins"] == best_kin_param[0]) & (kin_raw_df["L1vsL2"] == best_kin_param[1]) & (
    #             kin_raw_df["C"] == best_kin_param[2])]["Mean"]
    #
    # best_raw_kin_df = pd.DataFrame()
    # for x in range(len(syn_best_acc_df.columns) - 1):
    #     best_raw_kin_df = pd.concat([best_raw_kin_df, best_raw_kin_results], axis=1)
    # # best_raw_kin_df = best_raw_kin_df.transpose()
    # best_raw_kin_df.columns = perc_values

    """ KIN lines plot """
    sns.pointplot(data=pd.DataFrame(kin_cum_var).transpose(), label='Variance Explained', color='0', scale=.5)
    i = sns.pointplot(data=syn_best_acc_df.loc[syn_best_acc_df["Source"] == "Kin"], errorbar='ci', errwidth='.75',
                      capsize=.2,
                      color="r", label='Syn classifier')
    sns.pointplot(data=best_raw_kin_df, errorbar='ci', errwidth='.75', capsize=.2, color="b", label='Raw classifier')
    i.set(ylabel="Accuracy (95% ci)\nVariance Explained")
    i.set(xlabel="Percentage of Synergies")
    i.axhline(33, color='0.75', linestyle='--', label='Chance level')
    # i.axhline(55.52, color='r', linestyle='--', label='Raw Classifier')
    leg = plt.legend(labels=['Syn classifier', 'Raw classifier', 'Chance Level', 'Variance Explained'],
                     labelcolor=['r', 'b', '0.75', '0'])
    handles = leg.legendHandles
    colors = ['r', 'b', '0.75', '0']
    for it, handle in enumerate(handles):
        handle.set_color(colors[it])
        handle.set_linewidth(1)
    i.set_ylim([0, 100])
    sns.move_legend(i, "best")
    # plt.show()

    if type == 'all':
        title = 'Kinematic accuracy comparison for all subjects'
        fig_file = './results/Syn/plots/kin_drop_syn_acc'
    elif type == 'clustering':
        title = 'Kinematic accuracy comparison for each subject with clustering'
        fig_file = './results/Syn/plots/subj_clust_kin_drop_syn_acc'
    elif type == 'alternative':
        title = 'Kinematic accuracy comparison for each subject with clustering'
        fig_file = './results/Syn/plots/alternative_kin_drop_syn_acc'
    else:  # early enclosure
        title = 'Kinematic accuracy comparison for early enclosure'
        fig_file = './results/Early Enclosure/plots/kin_drop_syn_acc'

    if discard == 'less':
        title += '\ndiscarding less relevant synergies'
        fig_file += '.png'
    else:  # discard the most
        title += '\ndiscarding most relevant synergies'
        fig_file += '_decr.png'

    i.set(title=title)
    # plt.savefig(fig_file, dpi=600)
    plt.close()
    plt.clf()

    """ KIN bar pval plot """
    kin_pval_df = pd.DataFrame(syn_best_acc_df.iloc[:, 1:].loc[syn_best_acc_df["Source"] == "Kin"])
    kin_pval_df.insert(0, 'Raw', best_raw_kin_df.iloc[:, 0].values)
    plt.figure()
    sns.pointplot(data=pd.DataFrame(extended_kin_cum_var).transpose(), label='Variance Explained', color='r', scale=.5)
    i = sns.barplot(data=kin_pval_df)
    # pairs_kin = [('Raw', '100'), ('Raw', '90'), ('Raw', '80'), ('Raw', '70'), ('Raw', '60'), ('Raw', '50'), ('Raw', '40'), ('Raw', '30'), ('Raw', '20'), ('Raw', '10')]
    pairs_kin = [('Raw', 100), ('Raw', 90), ('Raw', 80), ('Raw', 70), ('Raw', 60), ('Raw', 50),
                 ('Raw', 40), ('Raw', 30), ('Raw', 20), ('Raw', 10)]
    annotator_i = Annotator(i, pairs_kin[::-1], data=kin_pval_df)
    annotator_i.configure(test="Mann-Whitney", text_format="star", show_test_name=False, fontsize='xx-small',
                          line_height=0.01, line_width=0.5, color='0')
    annotator_i.apply_and_annotate()

    i.axhline(33, color='b', linestyle='--', label='Chance level')
    i.set(ylabel="Accuracy (95% ci)")
    leg = plt.legend(labels=['Chance Level', 'Variance Explained'], labelcolor=['b', 'r'])
    handles = leg.legendHandles
    colors = ['b', 'r']
    for it, handle in enumerate(handles):
        handle.set_color(colors[it])
        handle.set_linewidth(1)
    sns.move_legend(i, "center right")
    # i.set(xlabel=None)
    # plt.xticks(rotation=45, size=4)
    # # i.axhline(20, color='r')

    if type == 'all':
        title = 'Kinematic accuracy comparison for all subjects'
        fig_file = './results/Syn/plots/kin_drop_syn_acc_pval'
    elif type == 'clustering':
        title = 'Kinematic accuracy comparison for each subject with clustering'
        fig_file = './results/Syn/plots/subj_clust_kin_drop_syn_acc_pval'
    elif type == 'alternative':
        title = 'Kinematic accuracy comparison for each subject with clustering'
        fig_file = './results/Syn/plots/alternative_kin_drop_syn_acc_pval'
    else:  # early enclosure
        title = 'Kinematic accuracy comparison for early enclosure'
        fig_file = './results/Early Enclosure/plots/kin_drop_syn_acc_pval'

    if discard == 'less':
        title += '\ndiscarding less relevant synergies'
        fig_file += '.png'
    else:  # discard the most
        title += '\ndiscarding most relevant synergies'
        fig_file += '_decr.png'

    i.set(title=title)
    i.set_ylim(0, 200)
    plt.savefig(fig_file, dpi=600)
    # plt.show()
    plt.close()
    plt.clf()

    """  EMG PCA plot """
    sns.pointplot(data=pd.DataFrame(emg_pca_cum_var).transpose(), label='Variance Explained', color='0', scale=.5)
    j = sns.pointplot(data=syn_best_acc_df.loc[syn_best_acc_df["Source"] == "EMG PCA"], errorbar='ci', errwidth='.75',
                      capsize=.2,
                      color="r", label='Syn classifier')
    sns.pointplot(data=best_raw_emg_df, errorbar='ci', errwidth='.75', capsize=.2, color="b", label='Raw classifier')
    j.set(ylabel="Accuracy (95% ci)")
    j.set(xlabel="Percentage of Synergies")
    j.axhline(33, color='0.75', linestyle='--', label='Chance level')
    # j.axhline(55.52, color='r', linestyle='--', label='Raw Classifier')
    leg = plt.legend(labels=['Syn classifier', 'Raw classifier', 'Chance Level', 'Variance Explained'],
                     labelcolor=['r', 'b', '0.75', '0'])
    handles = leg.legendHandles
    colors = ['r', 'b', '0.75', '0']
    for it, handle in enumerate(handles):
        handle.set_color(colors[it])
        handle.set_linewidth(1)
    j.set_ylim([0, 100])
    sns.move_legend(j, "best")
    # plt.show()

    if type == 'all':
        title = 'EMG PCA accuracy comparison for all subjects'
        fig_file = './results/Syn/plots/emg_pca_drop_syn_acc'
    elif type == 'clustering':
        title = 'EMG PCA accuracy comparison for each subject with clustering'
        fig_file = './results/Syn/plots/subj_clust_emg_pca_drop_syn_acc'
    else:  # early enclosure
        title = 'EMG PCA accuracy comparison for early enclosure'
        fig_file = './results/Early Enclosure/plots/emg_pca_drop_syn_acc'

    if discard == 'less':
        title += '\ndiscarding less relevant synergies'
        fig_file += '.png'
    else:  # discard the most
        title += '\ndiscarding most relevant synergies'
        fig_file += '_decr.png'

    j.set(title=title)
    plt.savefig(fig_file, dpi=600)
    plt.close()
    plt.clf()

    """ EMG PCA bar pval plot """
    emg_pca_pval_df = pd.DataFrame(syn_best_acc_df.iloc[:, 1:].loc[syn_best_acc_df["Source"] == "EMG PCA"])
    emg_pca_pval_df.insert(0, 'Raw', best_raw_emg_df.iloc[:, 0].values)
    plt.figure()
    sns.pointplot(data=pd.DataFrame(extended_emg_pca_cum_var).transpose(), label='Variance Explained', color='r',
                  scale=.5)
    i = sns.barplot(data=emg_pca_pval_df)
    # pairs_emg_pca = [('Raw', '100'), ('Raw', '90'), ('Raw', '80'), ('Raw', '70'), ('Raw', '60'), ('Raw', '50'), ('Raw', '40'), ('Raw', '30'), ('Raw', '20'), ('Raw', '10')]
    pairs_emg_pca = [('Raw', 100), ('Raw', 90), ('Raw', 80), ('Raw', 70), ('Raw', 60), ('Raw', 50),
                     ('Raw', 40), ('Raw', 30), ('Raw', 20), ('Raw', 10)]
    annotator_i = Annotator(i, pairs_emg_pca, data=emg_pca_pval_df)
    annotator_i.configure(test="Mann-Whitney", text_format="star", show_test_name=False, fontsize='xx-small',
                          line_height=0.01, line_width=0.5, color='0')
    annotator_i.apply_and_annotate()
    i.set(ylabel="Accuracy (95% ci)")
    i.axhline(33, color='b', linestyle='--', label='Chance level')
    leg = plt.legend(labels=['Chance Level', 'Variance Explained'], labelcolor=['b', 'r'])
    handles = leg.legendHandles
    colors = ['b', 'r']
    for it, handle in enumerate(handles):
        handle.set_color(colors[it])
        handle.set_linewidth(1)
    sns.move_legend(i, "center right")

    if type == 'all':
        title = 'EMG PCA accuracy comparison for all subjects'
        fig_file = './results/Syn/plots/emg_pca_drop_syn_acc_pval'
    elif type == 'clustering':
        title = 'EMG PCA accuracy comparison for each subject with clustering'
        fig_file = './results/Syn/plots/subj_clust_emg_pca_drop_syn_acc_pval'
    else:  # early enclosure
        title = 'EMG PCA accuracy comparison for early enclosure'
        fig_file = './results/Early Enclosure/plots/emg_pca_drop_syn_acc_pval'

    if discard == 'less':
        title += '\ndiscarding less relevant synergies'
        fig_file += '.png'
    else:  # discard the most
        title += '\ndiscarding most relevant synergies'
        fig_file += '_decr.png'

    i.set(title=title)
    i.set_ylim(0, 200)
    plt.savefig(fig_file, dpi=600)
    plt.close()
    plt.clf()

    """ TACT plot """
    sns.pointplot(data=pd.DataFrame(tact_cum_var).transpose(), label='Variance Explained', color='0', scale=.5)
    k = sns.pointplot(data=syn_best_acc_df.loc[syn_best_acc_df["Source"] == "Tact"], errorbar='ci', errwidth='.75',
                      capsize=.2,
                      color="r", label='Syn classifier')
    sns.pointplot(data=best_raw_tact_df, errorbar='ci', errwidth='.75', capsize=.2, color="b", label='Raw classifier')
    k.set(ylabel="Accuracy (95% ci)")
    k.set(xlabel="Percentage of Synergies")
    k.axhline(33, color='0.75', linestyle='--', label='Chance level')
    # k.axhline(55.52, color='r', linestyle='--', label='Raw Classifier')
    leg = plt.legend(labels=['Syn classifier', 'Raw classifier', 'Chance Level', 'Variance Explained'],
                     labelcolor=['r', 'b', '0.75', '0'])
    handles = leg.legendHandles
    colors = ['r', 'b', '0.75', '0']
    for it, handle in enumerate(handles):
        handle.set_color(colors[it])
        handle.set_linewidth(1)
    k.set_ylim([0, 100])
    sns.move_legend(k, "best")
    # plt.show()

    if type == 'all':
        title = 'Tactile accuracy comparison for all subjects'
        fig_file = './results/Syn/plots/tact_drop_syn_acc'
    elif type == 'clustering':
        title = 'Tactile accuracy comparison for each subject with clustering'
        fig_file = './results/Syn/plots/subj_clust_tact_drop_syn_acc'
    else:  # early enclosure
        title = 'Tactile accuracy comparison for early enclosure'
        fig_file = './results/Early Enclosure/plots/tact_drop_syn_acc'

    if discard == 'less':
        title += '\ndiscarding less relevant synergies'
        fig_file += '.png'
    else:  # discard the most
        title += '\ndiscarding most relevant synergies'
        fig_file += '_decr.png'

    k.set(title=title)
    plt.savefig(fig_file, dpi=600)
    plt.close()
    plt.clf()

    """ TACT bar pval plot """
    tact_pval_df = pd.DataFrame(syn_best_acc_df.iloc[:, 1:].loc[syn_best_acc_df["Source"] == "Tact"])
    tact_pval_df.insert(0, 'Raw', best_raw_tact_df.iloc[:, 0].values)
    plt.figure()
    sns.pointplot(data=pd.DataFrame(extended_tact_cum_var).transpose(), label='Variance Explained', color='r', scale=.5)
    i = sns.barplot(data=tact_pval_df)
    # pairs_tact = [('Raw', '100'), ('Raw', '90'), ('Raw', '80'), ('Raw', '70'), ('Raw', '60'), ('Raw', '50'), ('Raw', '40'), ('Raw', '30'), ('Raw', '20'), ('Raw', '10')]
    pairs_tact = [('Raw', 100), ('Raw', 90), ('Raw', 80), ('Raw', 70), ('Raw', 60), ('Raw', 50),
                  ('Raw', 40), ('Raw', 30), ('Raw', 20), ('Raw', 10)]
    annotator_i = Annotator(i, pairs_tact, data=tact_pval_df)
    annotator_i.configure(test="Mann-Whitney", text_format="star", show_test_name=False, fontsize='xx-small',
                          line_height=0.01, line_width=0.5, color='0')
    annotator_i.apply_and_annotate()
    i.set(ylabel="Accuracy (95% ci)")
    i.axhline(33, color='b', linestyle='--', label='Chance level')
    leg = plt.legend(labels=['Chance Level', 'Variance Explained'], labelcolor=['b', 'r'])
    handles = leg.legendHandles
    colors = ['b', 'r']
    for it, handle in enumerate(handles):
        handle.set_color(colors[it])
        handle.set_linewidth(1)
    sns.move_legend(i, "center right")
    # i.set(xlabel=None)
    # plt.xticks(rotation=45, size=4)
    # # i.axhline(20, color='r')

    if type == 'all':
        title = 'Tactile accuracy comparison for all subjects'
        fig_file = './results/Syn/plots/tact_drop_syn_acc_pval'
    elif type == 'clustering':
        title = 'Tactile accuracy comparison for each subject with clustering'
        fig_file = './results/Syn/plots/subj_clust_tact_drop_syn_acc_pval'
    else:  # early enclosure
        title = 'Tactile accuracy comparison for early enclosure'
        fig_file = './results/Early Enclosure/plots/tact_drop_syn_acc_pval'

    if discard == 'less':
        title += '\ndiscarding less relevant synergies'
        fig_file += '.png'
    else:  # discard the most
        title += '\ndiscarding most relevant synergies'
        fig_file += '_decr.png'

    i.set(title=title)
    i.set_ylim(0, 200)
    plt.savefig(fig_file, dpi=600)
    plt.close()
    plt.clf()

    """ MULTI plot """
    sns.pointplot(data=pd.DataFrame(combined_cum_var).transpose(), label='Variance Explained', color='0', scale=.5)
    k = sns.pointplot(data=syn_best_acc_df.loc[syn_best_acc_df["Source"] == "Multi"], errorbar='ci', errwidth='.75',
                      capsize=.2,
                      color="r", label='Syn classifier')
    sns.pointplot(data=best_raw_multi_df, errorbar='ci', errwidth='.75', capsize=.2, color="b", label='Raw classifier')
    k.set(ylabel="Accuracy (95% ci)")
    k.set(xlabel="Percentage of Synergies")
    k.axhline(33, color='0.75', linestyle='--', label='Chance level')
    # k.axhline(55.52, color='r', linestyle='--', label='Raw Classifier')
    leg = plt.legend(labels=['Syn classifier', 'Raw classifier', 'Chance Level', 'Variance Explained'],
                     labelcolor=['r', 'b', '0.75', '0'])
    handles = leg.legendHandles
    colors = ['r', 'b', '0.75', '0']
    for it, handle in enumerate(handles):
        handle.set_color(colors[it])
        handle.set_linewidth(1)
    k.set_ylim([0, 100])
    sns.move_legend(k, "best")
    # plt.show()

    if type == 'all':
        title = 'Multimodal accuracy comparison for all subjects'
        fig_file = './results/Syn/plots/multi_drop_syn_acc'
    elif type == 'clustering':
        title = 'Multimodal accuracy comparison for each subject with clustering'
        fig_file = './results/Syn/plots/subj_clust_multi_drop_syn_acc'
    else:  # early enclosure
        title = 'Multimodal accuracy comparison for early enclosure'
        fig_file = './results/Early Enclosure/plots/multi_drop_syn_acc'

    if discard == 'less':
        title += '\ndiscarding less relevant synergies'
        fig_file += '.png'
    else:  # discard the most
        title += '\ndiscarding most relevant synergies'
        fig_file += '_decr.png'

    k.set(title=title)
    plt.savefig(fig_file, dpi=600)
    plt.close()
    plt.clf()

    """ MULTI bar pval plot """
    multi_pval_df = pd.DataFrame(syn_best_acc_df.iloc[:, 1:].loc[syn_best_acc_df["Source"] == "Multi"])
    multi_pval_df.insert(0, 'Raw', best_raw_multi_df.iloc[:, 0].values)
    plt.figure()
    sns.pointplot(data=pd.DataFrame(extended_combined_cum_var).transpose(), label='Variance Explained', color='r',
                  scale=.5)
    i = sns.barplot(data=multi_pval_df)
    # pairs_multi = [('Raw', '100'), ('Raw', '90'), ('Raw', '80'), ('Raw', '70'), ('Raw', '60'), ('Raw', '50'), ('Raw', '40'), ('Raw', '30'), ('Raw', '20'), ('Raw', '10')]
    pairs_multi = [('Raw', 100), ('Raw', 90), ('Raw', 80), ('Raw', 70), ('Raw', 60), ('Raw', 50),
                   ('Raw', 40), ('Raw', 30), ('Raw', 20), ('Raw', 10)]
    annotator_i = Annotator(i, pairs_multi, data=multi_pval_df)
    annotator_i.configure(test="Mann-Whitney", text_format="star", show_test_name=False, fontsize='xx-small',
                          line_height=0.01, line_width=0.5, color='0')
    annotator_i.apply_and_annotate()
    i.set(ylabel="Accuracy (95% ci)")
    i.axhline(33, color='b', linestyle='--', label='Chance level')
    leg = plt.legend(labels=['Chance Level', 'Variance Explained'], labelcolor=['b', 'r'])
    handles = leg.legendHandles
    colors = ['b', 'r']
    for it, handle in enumerate(handles):
        handle.set_color(colors[it])
        handle.set_linewidth(1)
    sns.move_legend(i, "center right")
    i.set(xlabel=None)
    plt.xticks(rotation=45, size=4)
    # i.axhline(20, color='r')

    if type == 'all':
        title = 'Multimodal accuracy comparison for all subjects'
        fig_file = './results/Syn/plots/multi_drop_syn_acc_pval'
    elif type == 'clustering':
        title = 'Multimodal accuracy comparison for each subject with clustering'
        fig_file = './results/Syn/plots/subj_clust_multi_drop_syn_acc_pval'
    else:  # early enclosure
        title = 'Multimodal accuracy comparison for early enclosure'
        fig_file = './results/Early Enclosure/plots/multi_drop_syn_acc_pval'

    if discard == 'less':
        title += '\ndiscarding less relevant synergies'
        fig_file += '.png'
    else:  # discard the most
        title += '\ndiscarding most relevant synergies'
        fig_file += '_decr.png'

    i.set(title=title)
    i.set_ylim(0, 200)
    plt.savefig(fig_file, dpi=600)
    plt.close()
    plt.clf()

    """ HIER plot """
    sns.pointplot(data=pd.DataFrame(combined_cum_var).transpose(), label='Variance Explained', color='0', scale=.5)
    k = sns.pointplot(data=syn_best_acc_df.loc[syn_best_acc_df["Source"] == "Hier"], errorbar='ci', errwidth='.75',
                      capsize=.2,
                      color="r", label='Syn classifier')
    sns.pointplot(data=best_raw_hier_df, errorbar='ci', errwidth='.75', capsize=.2, color="b",
                  label='Raw classifier')
    k.set(ylabel="Accuracy (95% ci)")
    k.set(xlabel="Percentage of Synergies")
    k.axhline(33, color='0.75', linestyle='--', label='Chance level')
    # k.axhline(55.52, color='r', linestyle='--', label='Raw Classifier')
    leg = plt.legend(labels=['Syn classifier', 'Raw classifier', 'Chance Level', 'Variance Explained'],
                     labelcolor=['r', 'b', '0.75', '0'])
    handles = leg.legendHandles
    colors = ['r', 'b', '0.75', '0']
    for it, handle in enumerate(handles):
        handle.set_color(colors[it])
        handle.set_linewidth(1)
    k.set_ylim([0, 100])
    sns.move_legend(k, "best")
    # plt.show()

    if type == 'all':
        title = 'Hierarchical accuracy comparison for all subjects'
        fig_file = './results/Syn/plots/hier_drop_syn_acc'
    elif type == 'clustering':
        title = 'Hierarchical accuracy comparison for each subject with clustering'
        fig_file = './results/Syn/plots/subj_clust_hier_drop_syn_acc'
    else:  # early enclosure
        title = 'Hierarchical accuracy comparison for early enclosure'
        fig_file = './results/Early Enclosure/plots/hier_drop_syn_acc'

    if discard == 'less':
        title += '\ndiscarding less relevant synergies'
        fig_file += '.png'
    else:  # discard the most
        title += '\ndiscarding most relevant synergies'
        fig_file += '_decr.png'

    k.set(title=title)
    plt.savefig(fig_file, dpi=600)
    plt.close()
    plt.clf()

    """ HIER bar pval plot """
    hier_pval_df = pd.DataFrame(syn_best_acc_df.iloc[:, 1:].loc[syn_best_acc_df["Source"] == "Hier"])
    hier_pval_df.insert(0, 'Raw', best_raw_hier_df.iloc[:, 0].values)
    plt.figure()
    sns.pointplot(data=pd.DataFrame(extended_combined_cum_var).transpose(), label='Variance Explained', color='r',
                  scale=.5)
    i = sns.barplot(data=hier_pval_df)
    # pairs_hier = [('Raw', '100'), ('Raw', '90'), ('Raw', '80'), ('Raw', '70'), ('Raw', '60'), ('Raw', '50'), ('Raw', '40'), ('Raw', '30'), ('Raw', '20'), ('Raw', '10')]
    pairs_hier = [('Raw', 100), ('Raw', 90), ('Raw', 80), ('Raw', 70), ('Raw', 60), ('Raw', 50),
                  ('Raw', 40), ('Raw', 30), ('Raw', 20), ('Raw', 10)]
    annotator_i = Annotator(i, pairs_hier, data=hier_pval_df)
    annotator_i.configure(test="Mann-Whitney", text_format="star", show_test_name=False, fontsize='xx-small',
                          line_height=0.01, line_width=0.5, color='0')
    annotator_i.apply_and_annotate()
    i.set(ylabel="Accuracy (95% ci)")
    i.axhline(33, color='b', linestyle='--', label='Chance level')
    i.set(xlabel=None)
    plt.xticks(rotation=45, size=4)
    leg = plt.legend(labels=['Chance Level', 'Variance Explained'], labelcolor=['b', 'r'])
    handles = leg.legendHandles
    colors = ['b', 'r']
    for it, handle in enumerate(handles):
        handle.set_color(colors[it])
        handle.set_linewidth(1)
    sns.move_legend(i, "center right")
    # i.set(xlabel=None)
    # plt.xticks(rotation=45, size=4)
    # # i.axhline(20, color='r')

    if type == 'all':
        title = 'Hierarchical accuracy comparison for all subjects'
        fig_file = './results/Syn/plots/hier_drop_syn_acc_pval'
    elif type == 'clustering':
        title = 'Hierarchical accuracy comparison for each subject with clustering'
        fig_file = './results/Syn/plots/subj_clust_hier_drop_syn_acc_pval'
    else:  # early enclosure
        title = 'Hierarchical accuracy comparison for early enclosure'
        fig_file = './results/Early Enclosure/plots/hier_drop_syn_acc_pval'

    if discard == 'less':
        title += '\ndiscarding less relevant synergies'
        fig_file += '.png'
    else:  # discard the most
        title += '\ndiscarding most relevant synergies'
        fig_file += '_decr.png'

    i.set(title=title)
    i.set_ylim(0, 200)
    plt.savefig(fig_file, dpi=600)
    plt.close()
    plt.clf()
