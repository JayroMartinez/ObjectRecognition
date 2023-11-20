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
from synergy_pipeline import  syn_clustering
from synergy_pipeline import score_reordering


def main():

    # data_folder = '/BIDSData'
    # subject_folders = sorted([f.name for f in os.scandir(os.getcwd() + data_folder) if f.is_dir()])
    #
    # data_df = pd.DataFrame()
    #
    # ep_presabs_df = pd.DataFrame()
    # ep_dur_df = pd.DataFrame()
    # ep_count_df = pd.DataFrame()
    #
    # for subject in subject_folders:  # load data for each subject
    #     # LOAD RAW DATA
    #     subject_df = load(subject)
    #     data_df = pd.concat([data_df, subject_df], ignore_index=True)
    #
    #     # LOAD EP TRIALS
    #     [subject_ep_presabs, subject_ep_dur, subject_ep_count] = load_eps(subject)
    #     ep_presabs_df = pd.concat([ep_presabs_df, subject_ep_presabs], ignore_index=True)
    #     ep_dur_df = pd.concat([ep_dur_df, subject_ep_dur], ignore_index=True)
    #     ep_count_df = pd.concat([ep_count_df, subject_ep_count], ignore_index=True)



    # # RAW DATA PREPROCESSING
    # split_df = split(data_df)  # split data into trials and EPs and add fields
    # split_df['Trial num'] = split_df['Trial num'].astype('str')
    # split_df['EP num'] = split_df['EP num'].astype('str')

    # check tactile distribution (only for checking)
    # tactile_cols = ['rmo', 'mdo', 'rmi', 'mmo', 'pcim', 'ldd', 'rmm', 'rp', 'rdd', 'lmi', 'rdo', 'lmm', 'lp', 'rdm', 'ldm', 'ptip', 'idi', 'mdi', 'ido', 'mmm', 'ipi', 'mdm', 'idd', 'idm', 'imo', 'pdi', 'mmi', 'pdm', 'imm', 'mdd', 'pdii', 'mp', 'ptod', 'ptmd', 'tdo', 'pcid', 'imi', 'tmm', 'tdi', 'tmi', 'ptop', 'ptid', 'ptmp', 'tdm', 'tdd', 'tmo', 'pcip', 'ip', 'pcmp', 'rdi', 'ldi', 'lmo', 'pcmd', 'ldo', 'pdl', 'pdr', 'pdlo', 'lpo']
    # g = sns.violinplot(data=split_df[tactile_cols])
    # g.set_xticklabels(labels=tactile_cols, rotation=45, size=4)
    # plt.ylabel('Raw Tactile Data Value')
    # # plt.show()
    # plt.savefig('./results/raw_boxplot_tactile.png', dpi=600)

    # VARIANCE STUDY (only for checking)
    # variance(split_df)

    # ################################
    # ## EP CLASSIFICATION
    # ################################
    # # ASKED OBJECT CLASSIFICATION BY EP PRESENCE/ABSENCE
    # ask_ep_presabs_classification(ep_presabs_df)
    # # ASKED OBJECT CLASSIFICATION BY EP DURATION
    # ask_ep_dur_classification(ep_dur_df)
    # # ASKED OBJECT CLASSIFICATION BY EP COUNT
    # ask_ep_count_classification(ep_count_df)
    #
    # # GIVEN OBJECT CLASSIFICATION BY EP PRESENCE/ABSENCE
    # giv_ep_presabs_classification(ep_presabs_df)
    # # GIVEN OBJECT CLASSIFICATION BY EP DURATION
    # giv_ep_dur_classification(ep_dur_df)
    # # GIVEN OBJECT CLASSIFICATION BY EP COUNT
    # giv_ep_count_classification(ep_count_df)
    #
    # # FAMILY CLASSIFICATION BY EP PRESENCE/ABSENCE
    # fam_ep_presabs_classification(ep_presabs_df)
    # # FAMILY CLASSIFICATION BY EP DURATION
    # fam_ep_dur_classification(ep_dur_df)
    # # FAMILY CLASSIFICATION BY EP COUNT
    # fam_ep_count_classification(ep_count_df)

    # EP ACCURACY PLOTS
    # ep_classification_plots()
    # EP WEIGHT PLOTS
    # ep_weights()

    ################################
    ## RAW DATA CLASSIFICATION
    ################################

    # SINGLE SOURCE CLASSIFICATION
    # init_time = time.time()
    # emg_classification(split_df)
    # emg_time = time.time()
    # print("EMG elapsed time: ", round(emg_time - init_time))
    # kinematic_classification(split_df)
    # kinematic_time = time.time()
    # print("Kinematic elapsed time: ", round(kinematic_time - emg_time))
    # tactile_classification(split_df)
    # tactile_time = time.time()
    # print("Tactile elapsed time: ", round(tactile_time - kinematic_time))
    # # MULTIMODAL SOURCE CLASSIFICATION
    # multiple_source_classification(split_df)
    # multimodal_time = time.time()
    # print("Multimodal elapsed time: ", round(multimodal_time - tactile_time))
    # print("###########################################")
    # print("TOTAL elapsed time: ", round(multimodal_time - init_time))

    # HIERARCHICAL CLASSIFICATION
    # hierarchical_classification(split_df)

    ################################
    ## SYNERGY EXTRACTION
    ################################
    # split_df = []
    # syn_extraction(split_df)
    # syn_extraction_subj(split_df)
    """NOT PERFORMING CLUSTERING NOW"""
    # [syn_clustering() for x in range(0, 5)]
    # syn_clustering()
    # score_reordering()
    # print_syn_results()

    ################################
    ## SYNERGY CLASSIFICATION
    ################################
    syn_single_source_classification()
    multisource_syn_classification()
    # hierarchical_syn_classification()
    # print_syn_results()



if __name__ == "__main__":
    main()

