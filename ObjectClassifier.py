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
from ep_class_acc_stats import ep_stats_classification_plots
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
from ep_modelling import ep_from_scores_classif
from ep_modelling import ep_from_raw_classif
from synergy_pipeline import syn_clustering_alternative
from ep_modelling import ep_classification_plots
from synergy_pipeline import print_syn_results_alternative
from classification import kinematic_family_classification
from ep_modelling import ep_all_suj_syn_one_subject_out
from ep_modelling import ep_clust_suj_syn_one_subject_out
from ep_modelling import ep_all_suj_plots
from ep_modelling import ep_clust_suj_plots
from ep_modelling import build_subject_clusters
from ep_modelling import build_ep_clusters
from aux_get_best_param import best_parameter_combination_across_families
from feature_stats import feature_plots
from aux_distinguis import get_dist_heatmap

def main():

    """
    This is the main function for executing the entire pipeline

    """

    data_folder = '/BIDSData'
    subject_folders = sorted([f.name for f in os.scandir(os.getcwd() + data_folder) if f.is_dir()])

    data_df = pd.DataFrame()

    ep_presabs_df = pd.DataFrame()
    ep_dur_df = pd.DataFrame()
    ep_count_df = pd.DataFrame()

    for subject in subject_folders:  # load data for each subject
        """LOAD RAW DATA"""
        subject_df = load(subject)
        data_df = pd.concat([data_df, subject_df], ignore_index=True)

        # """LOAD EP TRIALS"""
        [subject_ep_presabs, subject_ep_dur, subject_ep_count] = load_eps(subject)
        ep_presabs_df = pd.concat([ep_presabs_df, subject_ep_presabs], ignore_index=True)
        ep_dur_df = pd.concat([ep_dur_df, subject_ep_dur], ignore_index=True)
        ep_count_df = pd.concat([ep_count_df, subject_ep_count], ignore_index=True)

    print("\nDATA LOADED")

    """RAW DATA PREPROCESSING"""
    split_df = split(data_df)  # split data into trials and EPs and add fields
    split_df['Trial num'] = split_df['Trial num'].astype('str')
    split_df['EP num'] = split_df['EP num'].astype('str')
    print("\nDATA PREPROCESSED")

    """REMOVE DOUBLE EP TRIALS"""
    to_remove = [x for x in split_df['EP'].unique() if '+' in x]
    split_df = split_df[~split_df['EP'].isin(to_remove)]
    ep_presabs_df = ep_presabs_df.drop(to_remove, axis=1)
    ep_dur_df = ep_dur_df.drop(to_remove, axis=1)
    ep_count_df = ep_count_df.drop(to_remove, axis=1)
    print("\nREMOVED DOUBLE EPs")

    """REPLACE CONTOUR FOLLOWING BY EDGE FOLLOWING"""
    split_df.loc[split_df['EP'] == 'edge following', 'EP'] = 'contour following'
    ep_presabs_df['contour following'] = ep_presabs_df['contour following'] + ep_presabs_df['edge following']
    ep_presabs_df = ep_presabs_df.drop(columns=['edge following'])
    ep_dur_df['contour following'] = ep_dur_df['contour following'] + ep_dur_df['edge following']
    ep_dur_df = ep_dur_df.drop(columns=['edge following'])
    ep_count_df['contour following'] = ep_count_df['contour following'] + ep_count_df['edge following']
    ep_count_df = ep_count_df.drop(columns=['edge following'])
    print("\nREPLACED EDGE FOLLOWING BY CONTOUR FOLLOWING")

    ###################################
    ## EP CLASSIFICATION
    ###################################
    """ASKED OBJECT CLASSIFICATION BY EP PRESENCE/ABSENCE"""
    ask_ep_presabs_classification(ep_presabs_df)
    """ASKED OBJECT CLASSIFICATION BY EP DURATION"""
    ask_ep_dur_classification(ep_dur_df)
    """ASKED OBJECT CLASSIFICATION BY EP COUNT"""
    ask_ep_count_classification(ep_count_df)

    """GIVEN OBJECT CLASSIFICATION BY EP PRESENCE/ABSENCE"""
    giv_ep_presabs_classification(ep_presabs_df)
    """GIVEN OBJECT CLASSIFICATION BY EP DURATION"""
    giv_ep_dur_classification(ep_dur_df)
    """GIVEN OBJECT CLASSIFICATION BY EP COUNT"""
    giv_ep_count_classification(ep_count_df)

    """FAMILY CLASSIFICATION BY EP PRESENCE/ABSENCE"""
    fam_ep_presabs_classification(ep_presabs_df)
    """FAMILY CLASSIFICATION BY EP DURATION"""
    fam_ep_dur_classification(ep_dur_df)
    """FAMILY CLASSIFICATION BY EP COUNT"""
    fam_ep_count_classification(ep_count_df)

    print("\nEP classification done!")

    """EP ACCURACY PLOTS"""
    ep_stats_classification_plots()
    """EP WEIGHT PLOTS"""
    ep_weights()

    feature_plots(ep_presabs_df, ep_dur_df, ep_count_df)
    print("\nEP PLOTS DONE")

    ###################################
    ## RAW DATA CLASSIFICATION
    ###################################

    """THESE FUNCTIONS ARE DEPRECATED"""
    """CHECK kinematic_family_classification()"""

    # """SINGLE SOURCE CLASSIFICATION"""
    # emg_classification(split_df)
    # print("\nEMG classification done!")

    # kinematic_classification(split_df)
    # print("\nKinematic classification done!")

    # tactile_classification(split_df)
    # print("\nTactile classification done!")

    # """MULTIMODAL SOURCE CLASSIFICATION"""
    # multiple_source_classification(split_df)
    # print("\nMultimodal classification done!")

    # """HIERARCHICAL CLASSIFICATION"""
    # hierarchical_classification(split_df)
    # print("\nHierarchical classification done!")

    ###################################
    ## SYNERGY EXTRACTION
    ###################################

    syn_extraction(split_df)
    print("\nSynergy extraction for all subjects done!")

    # syn_extraction_subj(split_df)
    # print("\nSynergy extraction for each subject done!")

    """
    CLUSTERING FUNCTIONS ARE NOT USED ANYMORE
    They were used to cluster synergies extracted from each subject into a general synergy
    """
    # syn_clustering()
    # print("\nSynergy clustering done!")

    # syn_clustering_alternative()
    # print("\nSynergy alternative clustering done!")

    # score_reordering('agglomerative')
    # score_reordering('alternative')
    # print("\nSynergy reordering done!")

    ###########################################################
    ## SYNERGY CLASSIFICATION ALL SUBJECTS (targeting given)
    ###########################################################
    """
    THIS FUNCTIONS ARE DEPRECATED
    See TARGETING FAMILY group of functions
    """

    # """single source"""
    # syn_single_source_classification('all', 'least')
    # get_best_params_single('all', 'least')
    # print("\nSingle source classification for all subjects discarding the least relevant DONE!")
    #
    # syn_single_source_classification('all', 'most')
    # get_best_params_single('all', 'most')
    # print("\nSingle source classification for all subjects discarding the most relevant DONE!")
    #
    # """multisource"""
    # multisource_syn_classification('all', 'least')
    # get_best_params_multi('all', 'least')
    # print("\nMultisource classification for all subjects discarding the least relevant DONE!")
    #
    # multisource_syn_classification('all', 'most')
    # get_best_params_multi('all', 'most')
    # print("\nMultisource classification for all subjects discarding the most relevant DONE!")
    #
    # """hierarchical"""
    # hierarchical_syn_classification('all', 'least')
    # get_best_params_hier('all', 'least')
    # print("\nHierarchical classification for all subjects discarding the least relevant DONE!")
    #
    # hierarchical_syn_classification('all', 'most')
    # get_best_params_hier('all', 'most')
    # print("\nHierarchical classification for all subjects discarding the most relevant DONE!")
    #
    # print_syn_results('all', 'least')
    # print_syn_results('all', 'most')

    ###########################################################
    ## SYNERGY CLASSIFICATION SINGLE SUBJECT + CLUSTERING
    ###########################################################
    """
    THESE FUNCTIONS ARE DEPRECATED
    Were used to perform classification over synergies extracted from each subject and clustered 
    """

    # """single source"""
    # syn_single_source_classification('clustering', 'least')
    # get_best_params_single('clustering', 'least')
    # print("\nSingle source classification for each subject with clustering discarding the least relevant DONE!")
    #
    # syn_single_source_classification('clustering', 'most')
    # get_best_params_single('clustering', 'most')
    # print("\nSingle source classification for each subject with clustering discarding the most relevant DONE!")
    #
    # """multisource"""
    # multisource_syn_classification('clustering', 'least')
    # get_best_params_multi('clustering', 'least')
    # print("\nMultisource classification for each subject with clustering discarding the least relevant DONE!")
    #
    # multisource_syn_classification('clustering', 'most')
    # get_best_params_multi('clustering','most')
    # print("\nMultisource classification for each subject with clustering discarding the most relevant DONE!")
    #
    # """hierarchical"""
    # hierarchical_syn_classification('clustering', 'least')
    # get_best_params_hier('clustering', 'least')
    # print("\nHierarchical classification for each subject with clustering discarding the least relevant DONE!")
    #
    # hierarchical_syn_classification('clustering', 'most')
    # get_best_params_hier('clustering', 'most')
    # print("\nHierarchical classification for each subject with clustering discarding the most relevant DONE!")

    # print_syn_results('clustering', 'least')
    # print_syn_results('clustering', 'most')

    ###########################################################
    ## FINE vs COARSE CHECKS
    ###########################################################
    """
    These functions compare the synergy scores for low and high order synergies 
    between fine and coarse (EPs or families).
    'Cluster' is deprecated
    """

    # syn_fine_vs_coarse_fam('cluster')
    syn_fine_vs_coarse_fam('all')

    # syn_fine_vs_coarse_ep('cluster')
    syn_fine_vs_coarse_ep('all')

    ###########################################################
    ## TARGETING FAMILY
    ###########################################################
    """
    This functions classify the trials targeting its family
    """

    """RAW CLASSIFICATION"""
    kinematic_family_classification(split_df)

    """DISCARDING SYNERGIES"""
    # # ['all', 'clustering', 'early'] ['least', 'most']
    fam_syn_single_source_classification('all', 'least')
    print("\nFamily classification from syn scores discarding the least relevant components DONE!")
    fam_syn_single_source_classification('all', 'most')
    print("\nFamily classification from syn scores discarding most relevant components DONE!")

    print_syn_results_alternative('most')
    print_syn_results_alternative('least')

    ###########################################################
    ## TARGETING EP
    ###########################################################
    """
    - This classifiers split the trials in EPs and bin them
    - They target the EP
    - The code is prepared to be able to include the subject as one-hot encoding
        but we are not using it
    """

    ep_from_raw_classif(split_df, False)
    print("\nEP classification from raw scores without subjects DONE!")

    # ep_from_raw_classif(split_df, True)
    # print("\nEP classification from raw scores with subjects DONE!")

    # # include_subjects = [true/false]
    ep_from_scores_classif(False)
    print("\nEP classification from syn scores without subjects DONE!")

    # ep_from_scores_classif(True)
    # print("\nEP classification from syn scores with subjects DONE!")

    ep_all_suj_syn_one_subject_out()
    # ep_clust_suj_syn_one_subject_out()

    # # ['syn', 'raw', 'syn_raw_suj', 'syn_raw_no_suj]
    """THIS FUNCTIONS ARE DEPRECATED"""
    # ep_classification_plots('syn')
    # ep_classification_plots('raw')
    # ep_classification_plots('syn_raw_suj')
    # ep_classification_plots('syn_raw_no_suj')

    ep_all_suj_plots()
    ep_clust_suj_plots()

    ################################
    ## EXTRA
    ################################
    """
    These are some auxiliar functions to perform complementary analyses
    """

    """GET BEST CLASSIFICATION ACCURACY DEPENDING ON THE METHOD"""
    # log_reg_file = './results/Raw/accuracy/raw_results.csv'
    # mini_batch_file = './results/Raw/accuracy/raw_fam_results.csv'
    #
    # log_reg_df = pd.read_csv(log_reg_file)
    # mini_batch_df = pd.read_csv(mini_batch_file)
    #
    # print("\nBest results with LogReg classifier:")
    # best_parameter_combination_across_families(log_reg_df)
    # print("\nBest results with MiniBatch classifier:")
    # best_parameter_combination_across_families(mini_batch_df)

    """GET BINARY DISTINGUISHABILITY TABLE"""
    # get_dist_heatmap()


if __name__ == "__main__":
    main()

