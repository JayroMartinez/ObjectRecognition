import os
import pandas as pd
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

    # emg_classification(split_df)
    # kinematic_classification(split_df)
    multiple_source_classification(split_df)


if __name__ == "__main__":
    main()

