import os

from load_subject import load

def main():

    data_folder = '/BIDSData'
    subject_folders = sorted([f.name for f in os.scandir(os.getcwd() + data_folder) if f.is_dir()])

    for subject in subject_folders:
        load(subject)


if __name__ == "__main__":
        main()