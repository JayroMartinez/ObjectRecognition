import os
import pandas as pd
import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

from load_subject import load
from split_data import split
from classification import emg_classification
from classification import kinematic_classification
from classification import multiple_source_classification

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

    init_time = time.time()
    emg_classification(split_df)
    emg_time = time.time()
    print("EMG elapsed time: ", round(emg_time - init_time))
    kinematic_classification(split_df)
    kinematic_time = time.time()
    print("Kinematic elapsed time: ", round(kinematic_time - emg_time))
    multiple_source_classification(split_df)
    multimodal_time = time.time()
    print("Kinematic elapsed time: ", round(multimodal_time - kinematic_time))
    print("###########################################")
    print("TOTAL elapsed time: ", round(multimodal_time - init_time))


if __name__ == "__main__":
    main()

