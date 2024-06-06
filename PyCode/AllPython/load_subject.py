import os
import csv
import py_compile

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def load(subject):

    cyberglove_folder = os.path.join(os.getcwd(), 'BIDSData', subject, 'cyberglove')
    cyberglove_files = [f.name for f in os.scandir(cyberglove_folder) if (f.is_file() and f.path.find('_dt') == -1)]

    # vicon_folder = os.path.join(os.getcwd(), 'BIDSData', subject, 'vicon')
    # vicon_files = [f.name for f in os.scandir(vicon_folder) if (f.is_file() and f.path.find('_dt') == -1)]
    #
    # emg_folder = os.path.join(os.getcwd(), 'BIDSData', subject, 'sessantaquattro')
    # emg_files = [f.name for f in os.scandir(emg_folder) if f.is_file()]

    # tactile_folder = os.path.join(os.getcwd(), 'BIDSData', subject, 'tactileglove')
    # tactile_files = [f.name for f in os.scandir(tactile_folder) if f.is_file()]

    # we are skipping the following checks for the tactile data

    clean_name_cyberglove = sorted([cy.replace('_cyberglove', '') for cy in cyberglove_files])
    # clean_name_vicon = sorted([cy.replace('_vicon', '') for cy in vicon_files])
    # clean_name_emg = sorted([cy.replace('_sessantaquattro', '') for cy in emg_files])

    # check if we have same tasks for all sources
    # if clean_name_cyberglove != clean_name_vicon or clean_name_cyberglove != clean_name_emg or clean_name_vicon != clean_name_emg:
    #     print("ERROR!!!!: Error while loading %s. Missing task for some source." %subject)

    tasks = [name.replace(subject + '_', '').replace('.csv', '') for name in clean_name_cyberglove]

    # load each source and task
    cyberglove_list = list()
    # vicon_list = list()
    # emg_list = list()
    # tactile_list = list()
    task_label = list()
    ep_label = list()

    for task in tasks:

        cyberglove_open_file = os.path.join(os.getcwd(), 'BIDSData', subject, 'cyberglove', subject + '_' + task + '_cyberglove.csv')
        # vicon_open_file = os.path.join(os.getcwd(), 'BIDSData', subject, 'vicon', subject + '_' + task + '_vicon.csv')
        # emg_open_file = os.path.join(os.getcwd(), 'BIDSData', subject, 'sessantaquattro', subject + '_' + task + '_sessantaquattro.csv')
        # tactile_open_file = os.path.join(os.getcwd(), 'BIDSData', subject, 'tactileglove', subject + '_' + task + '_tactileglove.csv')
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

        # with open(vicon_open_file) as vicon_o_f:
        #     op_vc = csv.reader(vicon_o_f)
        #     head_vc = next(op_vc)
        #     rows_vc = []
        #     for row_vc in op_vc:
        #         rows_vc.append(row_vc)
        # if len(rows_vc) > len(rows_cg):  # vicon sometimes has more datapoints than cyberglove, sometimes has the same
        #     rows_vc = rows_vc[0:len(rows_cg)]  # discard last datapoint
        # vicon_list.extend(rows_vc)

        # with open(emg_open_file) as emg_o_f:
        #     op_emg = csv.reader(emg_o_f)
        #     head_emg = next(op_emg)
        #     rows_emg = []
        #     for row_emg in op_emg:
        #         rows_emg.append(row_emg)
        # emg_idx = np.linspace(0, len(rows_emg) - 1, len(rows_vc)).astype(int)  # downsample emg
        # sel_emg_datapoints = [rows_emg[idx] for idx in emg_idx]
        # emg_list.extend(sel_emg_datapoints)

        # with open(tactile_open_file) as tactile_o_f:
        #     op_tactile = csv.reader(tactile_o_f)
        #     head_tactile = next(op_tactile)
        #     rows_tactile = []
        #     for row_tactile in op_tactile:
        #         rows_tactile.append(row_tactile)
        # tactile_list.extend(rows_tactile)

        with open(ep_open_file) as ep_o_f:
            op_ep = csv.reader(ep_o_f)
            next(op_ep, None)
            rows_ep = []
            for row_ep in op_ep:
                # rows_ep.append(row_ep[0])
                rows_ep.append(row_ep[1])
        if len(rows_ep) > len(rows_cg):  # EP labels sometimes has more datapoints than cyberglove
            rows_ep = rows_ep[0:len(rows_cg)]  # discard last datapoints
        elif len(rows_ep) < len(rows_cg):
            rows_ep.extend(rows_ep[-1] * (len(rows_cg) - len(rows_ep)))
        ep_label.extend(rows_ep)

        task_label.extend([task] * len(rows_cg))

    # all_sources = np.hstack((cyberglove_list, vicon_list, emg_list, tactile_list))
    # source_labels = np.hstack((head_cg, head_vc, head_emg, head_tactile))
    all_sources = cyberglove_list
    source_labels = head_cg
    all_sources_df = pd.DataFrame(all_sources, columns=source_labels, dtype='float')

    # Drop columns we don't want
    # We keep 14 cyberglove variables, 5 vicon variables and 64 emg variables
    # columns_to_drop = ['UNIX_time', 'ThumbAb', 'MiddleIndexAb', 'RingMiddleAb', 'PinkieRingAb', 'L_Thorax_X', 'L_Thorax_Y', 'L_Thorax_Z', 'Elbow_X', 'Elbow_Y', 'Elbow_Z', 'Shoulder_X', 'Shoulder_Y', 'Shoulder_Z', 'R_Thorax_X', 'R_Thorax_Y', 'R_Thorax_Z', 'Wrist_X', 'Wrist_Y', 'Wrist_Z', 'Index_Abs_J1_X', 'Index_Proj_J1_Y', 'Pinkie_Abs_J1_X', 'Pinkie_Proj_J1_Y', 'Ring_Abs_J1_X', 'Ring_Proj_J1_Y', 'Middle_Abs_J1_X', 'Middle_Proj_J1_Y', 'Thumb_Abs_J1_X', 'Thumb_Proj_J1_Y', 'Thumb_Abs_J2_X']
    columns_to_drop = 'runtime'
    all_sources_df.drop(columns_to_drop, axis=1, inplace=True)

    # Add task labels
    all_sources_df['Task'] = task_label

    # Add EP labels
    all_sources_df['EP'] = ep_label

    # Add Subject
    all_sources_df['Subject'] = [subject] * all_sources_df.shape[0]

    # Rectify EMG
    # emg_cols = [col for col in all_sources_df.columns if ('flexion' in col or 'extension' in col)]
    # all_sources_df[emg_cols] = all_sources_df[emg_cols].abs()

    return all_sources_df


def load_eps(subject):

    ep_labs_cols = ['contour following', 'contour following + enclosure part',
                    'edge following', 'enclosure', 'enclosure part',
                    'enclosure part + function test', 'function test', 'pressure',
                    'rotation', 'translation', 'weighting',
                    'weighting + contour following']

    obj_fam = dict(CeramicMug='Mugs',
                   Glass='Mugs',
                   MetalMug='Mugs',
                   CeramicPlate='Plates',
                   MetalPlate='Plates',
                   PlasticPlate='Plates',
                   Cube='Geometric',
                   Cylinder='Geometric',
                   Triangle='Geometric',
                   Fork='Cutlery',
                   Knife='Cutlery',
                   Spoon='Cutlery',
                   PingPongBall='Ball',
                   SquashBall='Ball',
                   TennisBall='Ball',
                   )

    dataframe_columns = np.hstack((ep_labs_cols, 'Given', 'Asked', 'Family'))
    ep_presabs = pd.DataFrame(columns=dataframe_columns)
    ep_presabs = ep_presabs.fillna(0)  # Initialize with zeros

    ep_dur = pd.DataFrame(columns=dataframe_columns)
    ep_dur = ep_dur.fillna(0)  # Initialize with zeros

    ep_count = pd.DataFrame(columns=dataframe_columns)
    ep_count = ep_count.fillna(0)  # Initialize with zeros

    label_folder = os.path.join(os.getcwd(), 'BIDSData', subject, 'labels')
    label_files = [f.name for f in os.scandir(label_folder) if f.is_file()]

    for file in label_files:
        ep_open_file = os.path.join(label_folder, file)

        with open(ep_open_file) as ep_open_read:
            op_ep = csv.reader(ep_open_read)
            rows_ep = [row_ep for row_ep in op_ep]

        del rows_ep[0]  # delete file header

        # Convert rows to a DataFrame for easier manipulation
        ep_data = pd.DataFrame(rows_ep, columns=['runtime', 'exploratory procedure'])
        ep_data['runtime'] = ep_data['runtime'].astype(float)

        # Calculate the elapsed time for each movement
        ep_data['time_diff'] = ep_data['runtime'].diff().fillna(0)
        time_spent = ep_data.groupby('exploratory procedure')['time_diff'].sum().reset_index()
        time_spent.columns = ['ep', 'time']

        aux_df = pd.DataFrame(0, index=[0], columns=ep_labs_cols)

        for index, row in time_spent.iterrows():
            label = row['ep']
            time = row['time']
            aux_df.at[0, label] = time

        parts = file.split('_')
        given = parts[1]
        asked = parts[2]
        family = obj_fam[given]

        aux_df['Given'] = given
        aux_df['Asked'] = asked
        aux_df['Family'] = family

        ep_dur = pd.concat([ep_dur, aux_df], ignore_index=True)

        # Count label changes
        ep_data['prev_label'] = ep_data['exploratory procedure'].shift(1)
        ep_data['change'] = ep_data['exploratory procedure'] != ep_data['prev_label']
        change_counts = ep_data[ep_data['change']].groupby('exploratory procedure').size().reset_index(name='count')
        change_counts_df = pd.DataFrame(0, index=[0], columns=ep_labs_cols)

        for index, row in change_counts.iterrows():
            label = row['exploratory procedure']
            count = row['count']
            change_counts_df.at[0, label] = count

        change_counts_df['Given'] = given
        change_counts_df['Asked'] = asked
        change_counts_df['Family'] = family

        ep_count = pd.concat([ep_count, change_counts_df])

        # Create presence/absence DataFrame
        presence_absence_df = pd.DataFrame(0, index=[0], columns=ep_labs_cols)
        presence_absence_df.loc[0, ep_data['exploratory procedure'].unique()] = 1

        presence_absence_df['Given'] = given
        presence_absence_df['Asked'] = asked
        presence_absence_df['Family'] = family

        ep_presabs = pd.concat([ep_presabs, presence_absence_df])

    return [ep_presabs, ep_dur, ep_count]

