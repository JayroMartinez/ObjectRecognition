import os
import pandas as pd

from load_subject import load

def main():

    data_folder = '/BIDSData'
    subject_folders = sorted([f.name for f in os.scandir(os.getcwd() + data_folder) if f.is_dir()])

    data_df = pd.DataFrame()
    for subject in subject_folders:
        subject_df = load(subject)
        data_df = pd.concat([data_df, subject_df], ignore_index=True)

    print(data_df.head)
    print(data_df.shape)





if __name__ == "__main__":
        main()