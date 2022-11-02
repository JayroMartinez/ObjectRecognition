import os
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from load_subject import load
from split_data import split
from classification import emg_classification

def main():

    data_folder = '/BIDSData'
    subject_folders = sorted([f.name for f in os.scandir(os.getcwd() + data_folder) if f.is_dir()])

    data_df = pd.DataFrame()

    for subject in subject_folders:
        subject_df = load(subject)
        data_df = pd.concat([data_df, subject_df], ignore_index=True)

    split_df = split(data_df)
    split_df['Trial num'] = split_df['Trial num'].astype('str')
    split_df['EP num'] = split_df['EP num'].astype('str')

    emg_classification(split_df)
    # kinematic_classification(split_df)
    # multiple_source_classification(split_df)


if __name__ == "__main__":
        main()