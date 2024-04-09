import os
import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

from load_subject import load
from load_subject import load_eps
from split_data import split
from classification import emg_classification
from classification import kinematic_classification
from classification import tactile_classification
from classification import multiple_source_classification
from classification import hierarchical_classification
from classification import ask_ep_presabs_classification
from classification import ask_ep_dur_classification
from classification import ask_ep_count_classification
from classification import giv_ep_presabs_classification
from classification import giv_ep_dur_classification
from classification import giv_ep_count_classification
from classification import fam_ep_presabs_classification
from classification import fam_ep_dur_classification
from classification import fam_ep_count_classification
from classification import get_raw_best_params
from ep_class_acc_stats import ep_classification_plots
from stat_analysis import variance
from weight_analysis import raw_weights
from weight_analysis import ep_weights
from synergy_pipeline import syn_extraction
from synergy_pipeline import syn_extraction_subj
from synergy_pipeline import syn_single_source_classification
from synergy_pipeline import print_syn_results
from synergy_pipeline import hierarchical_syn_classification
from synergy_pipeline import multisource_syn_classification
from synergy_pipeline import syn_clustering
from synergy_pipeline import score_reordering
from synergy_pipeline import syn_extraction_early_enclosure
from synergy_pipeline import syn_clustering_early_enclosure
from synergy_pipeline import score_reordering_early_enclosure
from synergy_pipeline import extract_early_enclosure_alt
from synergy_pipeline import get_best_params_single
from synergy_pipeline import get_best_params_multi
from synergy_pipeline import get_best_params_hier
from synergy_pipeline import early_fine_vs_coarse
from synergy_pipeline import syn_fine_vs_coarse_fam
from synergy_pipeline import syn_fine_vs_coarse_ep
from family_pipeline import fam_syn_single_source_classification
from synergy_pipeline import all_subjects_comp
from synergy_pipeline import clustered_comp
from synergy_pipeline import distances
from ep_modelling import ep_from_scores_classif
from ep_modelling import ep_from_raw_classif
from synergy_pipeline import syn_clustering_alternative
from ep_modelling import ep_classification_plots

def main():

    # data_folder = '/BIDSData'
    # subject_folders = sorted([f.name for f in os.scandir(os.getcwd() + data_folder) if f.is_dir()])
    #
    # # """WE ARE REMOVING SUBJECTS 7 TO 9"""
    # # ##########################################################################################################################
    # # [subject_folders.remove(x) for x in ['sub-07', 'sub-08', 'sub-09']]
    # # ##########################################################################################################################
    #
    # data_df = pd.DataFrame()
    #
    # # ep_presabs_df = pd.DataFrame()
    # # ep_dur_df = pd.DataFrame()
    # # ep_count_df = pd.DataFrame()
    #
    # for subject in subject_folders:  # load data for each subject
    #     """LOAD RAW DATA"""
    #     subject_df = load(subject)
    #     data_df = pd.concat([data_df, subject_df], ignore_index=True)
    #
    #     # """LOAD EP TRIALS"""
    #     # [subject_ep_presabs, subject_ep_dur, subject_ep_count] = load_eps(subject)
    #     # ep_presabs_df = pd.concat([ep_presabs_df, subject_ep_presabs], ignore_index=True)
    #     # ep_dur_df = pd.concat([ep_dur_df, subject_ep_dur], ignore_index=True)
    #     # ep_count_df = pd.concat([ep_count_df, subject_ep_count], ignore_index=True)
    #
    # print("\nDATA LOADED")
    #
    # """RAW DATA PREPROCESSING"""
    # split_df = split(data_df)  # split data into trials and EPs and add fields
    # split_df['Trial num'] = split_df['Trial num'].astype('str')
    # split_df['EP num'] = split_df['EP num'].astype('str')
    # print("\nDATA PREPROCESSED")
    #
    # # """CHECK MIDDLE FINGER VALUES"""
    # # plt.figure()
    # # sns.boxplot(data=split_df, x="Subject", y="MiddleMPJ")
    # # plt.xticks(rotation=45, size=5)
    # # plt.show()
    # # plt.savefig('./MiddleMPJ.png')
    # # plt.close()
    #
    # """REMOVE DOUBLE EP TRIALS"""
    # to_remove = [x for x in split_df['EP'].unique() if '+' in x]
    # split_df = split_df[~split_df['EP'].isin(to_remove)]
    # # ep_presabs_df = ep_presabs_df.drop(to_remove, axis=1)
    # # ep_dur_df = ep_dur_df.drop(to_remove, axis=1)
    # # ep_count_df = ep_count_df.drop(to_remove, axis=1)
    # print("\nREMOVED DOUBLE EPs")
    #
    # """REPLACE CONTOUR FOLLOWING BY EDGE FOLLOWING"""
    # split_df.loc[split_df['EP'] == 'contour following', 'EP'] = 'edge following'
    # print("\nREPLACED CONTOUR FOLLOWING BY EDGE FOLLOWING")

    # """SELECT & SAVE EARLY ENCLOSURE DATA"""
    # early_enclosure = split_df[(split_df['EP num'].isin(['0', '1'])) & (split_df['EP'].isin(['enclosure', 'enclosure part']))]
    # early_enclosure.to_csv('./results/Early Enclosure/early_enclosure_data.csv')

    # check tactile distribution (only for checking)
    # tactile_cols = ['rmo', 'mdo', 'rmi', 'mmo', 'pcim', 'ldd', 'rmm', 'rp', 'rdd', 'lmi', 'rdo', 'lmm', 'lp', 'rdm', 'ldm', 'ptip', 'idi', 'mdi', 'ido', 'mmm', 'ipi', 'mdm', 'idd', 'idm', 'imo', 'pdi', 'mmi', 'pdm', 'imm', 'mdd', 'pdii', 'mp', 'ptod', 'ptmd', 'tdo', 'pcid', 'imi', 'tmm', 'tdi', 'tmi', 'ptop', 'ptid', 'ptmp', 'tdm', 'tdd', 'tmo', 'pcip', 'ip', 'pcmp', 'rdi', 'ldi', 'lmo', 'pcmd', 'ldo', 'pdl', 'pdr', 'pdlo', 'lpo']
    # g = sns.violinplot(data=split_df[tactile_cols])
    # g.set_xticklabels(labels=tactile_cols, rotation=45, size=4)
    # plt.ylabel('Raw Tactile Data Value')
    # # plt.show()
    # plt.savefig('./results/raw_boxplot_tactile.png', dpi=600)

    # VARIANCE STUDY (only for checking)
    # variance(split_df)

    ###################################
    ## EP CLASSIFICATION
    ###################################
    # """ASKED OBJECT CLASSIFICATION BY EP PRESENCE/ABSENCE"""
    # ask_ep_presabs_classification(ep_presabs_df)
    # """ASKED OBJECT CLASSIFICATION BY EP DURATION"""
    # ask_ep_dur_classification(ep_dur_df)
    # """ASKED OBJECT CLASSIFICATION BY EP COUNT"""
    # ask_ep_count_classification(ep_count_df)

    # """GIVEN OBJECT CLASSIFICATION BY EP PRESENCE/ABSENCE"""
    # giv_ep_presabs_classification(ep_presabs_df)
    # """GIVEN OBJECT CLASSIFICATION BY EP DURATION"""
    # giv_ep_dur_classification(ep_dur_df)
    # """GIVEN OBJECT CLASSIFICATION BY EP COUNT"""
    # giv_ep_count_classification(ep_count_df)

    # """FAMILY CLASSIFICATION BY EP PRESENCE/ABSENCE"""
    # fam_ep_presabs_classification(ep_presabs_df)
    # """FAMILY CLASSIFICATION BY EP DURATION"""
    # fam_ep_dur_classification(ep_dur_df)
    # """FAMILY CLASSIFICATION BY EP COUNT"""
    # fam_ep_count_classification(ep_count_df)

    # print("\nEP classification done!")

    # """EP ACCURACY PLOTS"""
    # # ep_classification_plots()
    # """EP WEIGHT PLOTS"""
    # ep_weights()

    # print("\nEP PLOTS DONE")

    ###################################
    ## RAW DATA CLASSIFICATION
    ###################################

    # """SINGLE SOURCE CLASSIFICATION"""
    # emg_classification(split_df)
    # print("\nEMG classification done!")
    #
    # kinematic_classification(split_df)
    # print("\nKinematic classification done!")
    #
    # tactile_classification(split_df)
    # print("\nTactile classification done!")
    #
    # """MULTIMODAL SOURCE CLASSIFICATION"""
    # multiple_source_classification(split_df)
    # print("\nMultimodal classification done!")
    #
    # """HIERARCHICAL CLASSIFICATION"""
    # hierarchical_classification(split_df)
    # print("\nHierarchical classification done!")

    ###################################
    ## SYNERGY EXTRACTION
    ###################################

    # syn_extraction(split_df)
    # print("\nSynergy extraction for all subjects done!")

    # syn_extraction_subj(split_df)
    # print("\nSynergy extraction for each subject done!")

    # syn_clustering()
    # print("\nSynergy clustering done!")

    # syn_clustering_alternative()
    # print("\nSynergy alternative clustering done!")
    #
    # # score_reordering('agglomerative')
    # score_reordering('alternative')
    # print("\nSynergy reordering done!")


    ###################################
    ## SYNERGY COMPARISON
    ###################################

    # all_subjects_comp()
    # clustered_comp()

    ###########################################################
    ## SYNERGY CLASSIFICATION ALL SUBJECTS
    ###########################################################

    # """single source"""
    # syn_single_source_classification('all', 'less')
    # get_best_params_single('all', 'less')
    # print("\nSingle source classification for all subjects discarding the less relevant DONE!")
    #
    # syn_single_source_classification('all', 'most')
    # get_best_params_single('all', 'most')
    # print("\nSingle source classification for all subjects discarding the most relevant DONE!")
    #
    # """multisource"""
    # multisource_syn_classification('all', 'less')
    # get_best_params_multi('all', 'less')
    # print("\nMultisource classification for all subjects discarding the less relevant DONE!")
    #
    # multisource_syn_classification('all', 'most')
    # get_best_params_multi('all', 'most')
    # print("\nMultisource classification for all subjects discarding the most relevant DONE!")
    #
    # """hierarchical"""
    # hierarchical_syn_classification('all', 'less')
    # get_best_params_hier('all', 'less')
    # print("\nHierarchical classification for all subjects discarding the less relevant DONE!")
    #
    # hierarchical_syn_classification('all', 'most')
    # get_best_params_hier('all', 'most')
    # print("\nHierarchical classification for all subjects discarding the most relevant DONE!")
    #
    # print_syn_results('all', 'less')
    # print_syn_results('all', 'most')

    ###########################################################
    ## SYNERGY CLASSIFICATION SINGLE SUBJECT + CLUSTERING
    ###########################################################

    # """single source"""
    # syn_single_source_classification('clustering', 'less')
    # get_best_params_single('clustering', 'less')
    # print("\nSingle source classification for each subject with clustering discarding the less relevant DONE!")
    #
    # syn_single_source_classification('clustering', 'most')
    # get_best_params_single('clustering', 'most')
    # print("\nSingle source classification for each subject with clustering discarding the most relevant DONE!")
    #
    # """multisource"""
    # multisource_syn_classification('clustering', 'less')
    # get_best_params_multi('clustering', 'less')
    # print("\nMultisource classification for each subject with clustering discarding the less relevant DONE!")
    #
    # multisource_syn_classification('clustering', 'most')
    # get_best_params_multi('clustering','most')
    # print("\nMultisource classification for each subject with clustering discarding the most relevant DONE!")
    #
    # """hierarchical"""
    # hierarchical_syn_classification('clustering', 'less')
    # get_best_params_hier('clustering', 'less')
    # print("\nHierarchical classification for each subject with clustering discarding the less relevant DONE!")
    #
    # hierarchical_syn_classification('clustering', 'most')
    # get_best_params_hier('clustering', 'most')
    # print("\nHierarchical classification for each subject with clustering discarding the most relevant DONE!")

    # print_syn_results('clustering', 'less')
    # print_syn_results('clustering', 'most')

    ###########################################################
    ## EARLY ENCLOSURE ALTERNATIVE (pca + ee)
    ###########################################################

    # extract_early_enclosure_alt()
    #
    # """single source"""
    # syn_single_source_classification('early', 'less')
    # get_best_params_single('early', 'less')
    # print("\nSingle source classification for early enclosure discarding the less relevant DONE!")
    #
    # syn_single_source_classification('early', 'most')
    # get_best_params_single('early', 'most')
    # print("\nSingle source classification for early enclosure discarding the most relevant DONE!")
    #
    # """multisource"""
    # multisource_syn_classification('early', 'less')
    # get_best_params_multi('early', 'less')
    # print("\nMultisource classification for early enclosure discarding the less relevant DONE!")
    #
    # multisource_syn_classification('early', 'most')
    # get_best_params_multi('early', 'most')
    # print("\nMultisource classification for early enclosure discarding the most relevant DONE!")
    #
    # """hierarchical"""
    # hierarchical_syn_classification('early', 'less')
    # get_best_params_hier('early', 'less')
    # print("\nHierarchical classification for early enclosure discarding the less relevant DONE!")
    #
    # hierarchical_syn_classification('early', 'most')
    # get_best_params_hier('early', 'most')
    # print("\nHierarchical classification for early enclosure discarding the most relevant DONE!")
    #
    # print_syn_results('early', 'less')
    # print_syn_results('early', 'most')

    ###########################################################
    ## FINE vs COARSE CHECKS
    ###########################################################
    # early_fine_vs_coarse()

    # syn_fine_vs_coarse_fam('cluster')
    # syn_fine_vs_coarse_fam('all')
    #
    # syn_fine_vs_coarse_ep('cluster')
    # syn_fine_vs_coarse_ep('all')

    ###########################################################
    ## DISTANCE CHECK
    ###########################################################

    # distances('cluster')
    # distances('all')

    ###########################################################
    ## TARGETING FAMILY
    ###########################################################
    # ['all', 'clustering', 'early'] ['less', 'most']
    fam_syn_single_source_classification('alternative', 'less')
    print("\nFamily classification from syn scores discarding less relevant components DONE!")
    fam_syn_single_source_classification('alternative', 'most')
    print("\nFamily classification from syn scores discarding most relevant components DONE!")

    ###########################################################
    ## TARGETING EP
    ###########################################################

    # ep_from_raw_classif(split_df, False)
    # print("\nEP classification from raw scores without subjects DONE!")
    #
    # ep_from_raw_classif(split_df, True)
    # print("\nEP classification from raw scores with subjects DONE!")
    #
    # ep_from_scores_classif(False)
    # print("\nEP classification from syn scores without subjects DONE!")
    #
    # ep_from_scores_classif(True)
    # print("\nEP classification from syn scores with subjects DONE!")
    #
    # # # ['syn', 'raw', 'syn_raw_suj', 'syn_raw_no_suj]
    # ep_classification_plots('syn')
    # ep_classification_plots('raw')
    # ep_classification_plots('syn_raw_suj')
    # ep_classification_plots('syn_raw_no_suj')


if __name__ == "__main__":
    main()

