import os
import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

from load_subject import load
from split_data import split
from classification import emg_classification
from classification import kinematic_classification
from classification import tactile_classification
from classification import multiple_source_classification
from classification import hierarchical_classification
from classification import eq_seq_classification
from stat_analysis import variance


def main():

    data_folder = '/BIDSData'
    subject_folders = sorted([f.name for f in os.scandir(os.getcwd() + data_folder) if f.is_dir()])

    data_df = pd.DataFrame()

    for subject in subject_folders:  # load data for each subject
        subject_df = load(subject)
        data_df = pd.concat([data_df, subject_df], ignore_index=True)

    split_df = split(data_df)  # split data into trials and EPs and add fields
    split_df['Trial num'] = split_df['Trial num'].astype('str')
    split_df['EP num'] = split_df['EP num'].astype('str')

    # check tactile distribution
    # tactile_cols = ['rmo', 'mdo', 'rmi', 'mmo', 'pcim', 'ldd', 'rmm', 'rp', 'rdd', 'lmi', 'rdo', 'lmm', 'lp', 'rdm', 'ldm', 'ptip', 'idi', 'mdi', 'ido', 'mmm', 'ipi', 'mdm', 'idd', 'idm', 'imo', 'pdi', 'mmi', 'pdm', 'imm', 'mdd', 'pdii', 'mp', 'ptod', 'ptmd', 'tdo', 'pcid', 'imi', 'tmm', 'tdi', 'tmi', 'ptop', 'ptid', 'ptmp', 'tdm', 'tdd', 'tmo', 'pcip', 'ip', 'pcmp', 'rdi', 'ldi', 'lmo', 'pcmd', 'ldo', 'pdl', 'pdr', 'pdlo', 'lpo']
    # g = sns.violinplot(data=split_df[tactile_cols])
    # g.set_xticklabels(labels=tactile_cols, rotation=45, size=4)
    # plt.ylabel('Raw Tactile Data Value')
    # # plt.show()
    # plt.savefig('./results/raw_boxplot_tactile.png', dpi=600)

    a = 1

    # VARIANCE STUDY
    # variance(split_df)

    # CLASSIFICATION BY EP 'SEQUENCE'
    # eq_seq_classification(split_df)

    # init_time = time.time()
    # emg_classification(split_df)
    # emg_time = time.time()
    # print("EMG elapsed time: ", round(emg_time - init_time))
    # kinematic_classification(split_df)
    kinematic_time = time.time()
    # print("Kinematic elapsed time: ", round(kinematic_time - emg_time))
    tactile_classification(split_df)
    tactile_time = time.time()
    print("Tactile elapsed time: ", round(tactile_time - kinematic_time))
    # multiple_source_classification(split_df)
    # multimodal_time = time.time()
    # print("Kinematic elapsed time: ", round(multimodal_time - tactile_time))
    # print("###########################################")
    # print("TOTAL elapsed time: ", round(multimodal_time - init_time))

    # HIERARCHICAL CLASSIFIER
    # hierarchical_classification(split_df)


if __name__ == "__main__":
    main()

