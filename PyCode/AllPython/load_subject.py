import os
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def load(subject):

    cyberglove_folder = os.path.join(os.getcwd(), 'BIDSData', subject, 'cyberglove')
    cyberglove_files = [f.name for f in os.scandir(cyberglove_folder) if (f.is_file() and f.path.find('_dt') == -1)]

    vicon_folder = os.path.join(os.getcwd(), 'BIDSData', subject, 'vicon')
    vicon_files = [f.name for f in os.scandir(vicon_folder) if (f.is_file() and f.path.find('_dt') == -1)]

    emg_folder = os.path.join(os.getcwd(), 'BIDSData', subject, 'sessantaquattro')
    emg_files = [f.name for f in os.scandir(emg_folder) if f.is_file()]

    tactile_folder = os.path.join(os.getcwd(), 'BIDSData', subject, 'tactileglove')
    tactile_files = [f.name for f in os.scandir(tactile_folder) if f.is_file()]

    # we are skipping the following checks for the tactile data

    clean_name_cyberglove = sorted([cy.replace('_cyberglove', '') for cy in cyberglove_files])
    clean_name_vicon = sorted([cy.replace('_vicon', '') for cy in vicon_files])
    clean_name_emg = sorted([cy.replace('_sessantaquattro', '') for cy in emg_files])

    # check if we have same tasks for all sources
    if clean_name_cyberglove != clean_name_vicon or clean_name_cyberglove != clean_name_emg or clean_name_vicon != clean_name_emg:
        print("ERROR!!!!: Error while loading %s. Missing task for some source." %subject)

    tasks = [name.replace(subject + '_', '').replace('.csv', '') for name in clean_name_cyberglove]

    # load each source and task
    cyberglove_list = list()
    vicon_list = list()
    emg_list = list()
    tactile_list = list()
    task_label = list()
    ep_label = list()

    for task in tasks:

        cyberglove_open_file =  os.path.join(os.getcwd(), 'BIDSData', subject, 'cyberglove', subject + '_' + task + '_cyberglove.csv')
        vicon_open_file = os.path.join(os.getcwd(), 'BIDSData', subject, 'vicon', subject + '_' + task + '_vicon.csv')
        emg_open_file = os.path.join(os.getcwd(), 'BIDSData', subject, 'sessantaquattro', subject + '_' + task + '_sessantaquattro.csv')
        tactile_open_file = os.path.join(os.getcwd(), 'BIDSData', subject, 'tactileglove', subject + '_' + task + '_tactileglove.csv')
        ep_open_file = os.path.join(os.getcwd(), 'BIDSData', subject, 'labels', subject + '_' + task + '_labels.csv')

        with open(cyberglove_open_file) as cyberglove_o_f:
            op_cg = csv.reader(cyberglove_o_f)
            head_cg = next(op_cg)
            rows_cg = []
            for row_cg in op_cg:
                rows_cg.append(row_cg)
            # if subject == 'sub-02' and task == 'MetalMug_CeramicMug':  # this is terrible
            #     # rows_cg.pop()
            #     a=1
        cyberglove_list.extend(rows_cg)

        with open(vicon_open_file) as vicon_o_f:
            op_vc = csv.reader(vicon_o_f)
            head_vc = next(op_vc)
            rows_vc = []
            for row_vc in op_vc:
                rows_vc.append(row_vc)
        if len(rows_vc) > len(rows_cg):  # vicon sometimes has more datapoints than cyberglove, sometimes has the same
            rows_vc = rows_vc[0:len(rows_cg)]  # discard last datapoint
        vicon_list.extend(rows_vc)

        with open(emg_open_file) as emg_o_f:
            op_emg = csv.reader(emg_o_f)
            head_emg = next(op_emg)
            rows_emg = []
            for row_emg in op_emg:
                rows_emg.append(row_emg)
        emg_idx = np.linspace(0, len(rows_emg) - 1, len(rows_vc)).astype(int)  # downsample emg
        sel_emg_datapoints = [rows_emg[idx] for idx in emg_idx]
        emg_list.extend(sel_emg_datapoints)

        with open(tactile_open_file) as tactile_o_f:
            op_tactile = csv.reader(tactile_o_f)
            head_tactile = next(op_tactile)
            rows_tactile = []
            for row_tactile in op_tactile:
                rows_tactile.append(row_tactile)
        tactile_list.extend(rows_tactile)

        with open(ep_open_file) as ep_o_f:
            op_ep = csv.reader(ep_o_f)
            rows_ep = []
            for row_ep in op_ep:
                rows_ep.append(row_ep[0])
        if len(rows_ep) > len(rows_cg):  # EP labels sometimes has more datapoints than cyberglove
            rows_ep = rows_ep[0:len(rows_cg)]  # discard last datapoints
        elif len(rows_ep) < len(rows_cg):
            rows_ep.extend(rows_ep[-1] * (len(rows_cg) - len(rows_ep)))
        ep_label.extend(rows_ep)

        task_label.extend([task] * len(rows_cg))

    all_sources = np.hstack((cyberglove_list, vicon_list, emg_list, tactile_list))
    source_labels = np.hstack((head_cg, head_vc, head_emg, head_tactile))
    all_sources_df = pd.DataFrame(all_sources, columns=source_labels, dtype='float')

    # Drop columns we don't want
    # We keep 14 cyberglove variables, 5 vicon variables and 64 emg variables
    columns_to_drop = ['UNIX_time', 'ThumbAb', 'MiddleIndexAb', 'RingMiddleAb', 'PinkieRingAb', 'L_Thorax_X', 'L_Thorax_Y', 'L_Thorax_Z', 'Elbow_X', 'Elbow_Y', 'Elbow_Z', 'Shoulder_X', 'Shoulder_Y', 'Shoulder_Z', 'R_Thorax_X', 'R_Thorax_Y', 'R_Thorax_Z', 'Wrist_X', 'Wrist_Y', 'Wrist_Z', 'Index_Abs_J1_X', 'Index_Proj_J1_Y', 'Pinkie_Abs_J1_X', 'Pinkie_Proj_J1_Y', 'Ring_Abs_J1_X', 'Ring_Proj_J1_Y', 'Middle_Abs_J1_X', 'Middle_Proj_J1_Y', 'Thumb_Abs_J1_X', 'Thumb_Proj_J1_Y', 'Thumb_Abs_J2_X']
    all_sources_df.drop(columns_to_drop, axis=1, inplace=True)

    # Add task labels
    all_sources_df['Task'] = task_label

    # Add EP labels
    all_sources_df['EP'] = ep_label

    # Add Subject
    all_sources_df['Subject'] = [subject] * all_sources_df.shape[0]

    # Rectify EMG
    emg_cols = [col for col in all_sources_df.columns if ('flexion' in col or 'extension' in col)]
    all_sources_df[emg_cols] = all_sources_df[emg_cols].abs()

    return all_sources_df


