import os
import csv
import pandas as pd

def load(subject):

    cyberglove_folder = os.path.join(os.getcwd(), 'BIDSData', subject, 'cyberglove')
    cyberglove_files = [f.name for f in os.scandir(cyberglove_folder) if (f.is_file() and f.path.find('_dt') == -1)]

    vicon_folder = os.path.join(os.getcwd(), 'BIDSData', subject, 'vicon')
    vicon_files = [f.name for f in os.scandir(vicon_folder) if (f.is_file() and f.path.find('_dt') == -1)]

    emg_folder = os.path.join(os.getcwd(), 'BIDSData', subject, 'sessantaquattro')
    emg_files = [f.name for f in os.scandir(emg_folder) if f.is_file()]

    clean_name_cyberglove = sorted([cy.replace('_cyberglove', '') for cy in cyberglove_files])
    clean_name_vicon = sorted([cy.replace('_vicon', '') for cy in vicon_files])
    clean_name_emg = sorted([cy.replace('_sessantaquattro', '') for cy in emg_files])

    # check if we have same tasks for all sources
    if clean_name_cyberglove != clean_name_vicon or clean_name_cyberglove != clean_name_emg or clean_name_vicon != clean_name_emg:
        print("ERROR!!!!: Error while loading %s. Missing task for some source." %subject)

    for cyberglove_task in cyberglove_files:

        task = cyberglove_task.replace(subject+'_', '').replace('_cyberglove.csv', '')
        [given, ask] = task.split('_')

        cyberglove_task_file = os.path.join(cyberglove_folder, cyberglove_task)

        with open(cyberglove_task_file) as cyberglove_t_f:
            op = csv.reader(cyberglove_t_f)
            head = next(op)
            print(head)
            rows = []
            for row in op:
                rows.append(row)

        task_df = pd.DataFrame(rows, columns=head)


