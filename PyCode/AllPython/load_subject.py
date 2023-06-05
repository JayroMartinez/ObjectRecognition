import os
import csv
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

        cyberglove_open_file = os.path.join(os.getcwd(), 'BIDSData', subject, 'cyberglove', subject + '_' + task + '_cyberglove.csv')
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
    ep_dur = pd.DataFrame(columns=dataframe_columns)
    ep_count = pd.DataFrame(columns=dataframe_columns)

    label_folder = os.path.join(os.getcwd(), 'BIDSData', subject, 'sep_labels')
    label_files = [f.name for f in os.scandir(label_folder) if f.is_file()]

    trial_result = []

    for file in label_files:

        ep_open_file = os.path.join(os.getcwd(), 'BIDSData', subject, 'sep_labels', file)
        cyb_file = file.replace('labels', 'cyberglove')
        cyb_open_file = os.path.join(os.getcwd(), 'BIDSData', subject, 'cyberglove', cyb_file)

        with open(ep_open_file) as ep_open_read:
            op_ep = csv.reader(ep_open_read)
            rows_ep = []
            # print("File:", ep_open_file)
            for row_ep in op_ep:
                rows_ep.append(row_ep)

        del rows_ep[0]  # delete file header
        if 'prestart' in rows_ep[0]:
            del rows_ep[0]
        if 'postend' in rows_ep[-1]:
            del rows_ep[-1]

        # with this we get the last timestamp from the trial (postend is already removed)
        with open(cyb_open_file) as cyb_open_read:
            op_cyb = csv.reader(cyb_open_read)
            rows_cyb = []
            for row_cyb in op_cyb:
                rows_cyb.append(row_cyb)

        elapsed_time = float(rows_cyb[-1][0]) - float(rows_cyb[1][0])
        date_time = datetime.datetime.fromtimestamp(elapsed_time/1000000000)
        aux_elapsed = date_time.second + date_time.microsecond / 1000000
        last_timestamp = float(rows_ep[0][0]) + float(aux_elapsed)
        rows_ep[-1][1] = last_timestamp

        ep_duration = []
        ep_label = []
        given_object = []
        for iter in rows_ep:
            ep_duration.append(round(float(iter[1]) - float(iter[0]), 2))
            ep_label.append(iter[2].strip())
            # given_object.append(file.split('_')[1])

        ep_aux_df = pd.DataFrame({"Label": ep_label, "Duration": ep_duration})
        ep_dur_df = ep_aux_df.groupby('Label').sum('Duration')
        ep_dur_df['Label'] = ep_aux_df['Label'].unique()

        ep_count_df = ep_aux_df.groupby('Label').count()
        ep_count_df.rename(columns={"Duration":"Count"}, inplace=True)

        dur_vec = np.zeros((1, len(ep_labs_cols)))
        count_vec = np.zeros((1, len(ep_labs_cols)))

        for ep in list(ep_dur_df['Label'].values):
            dur_vec[0, ep_labs_cols.index(ep)] = float(ep_dur_df.loc[ep_dur_df['Label'] == ep]['Duration'])
            if ep in ep_count_df.index:
                count_vec[0, ep_labs_cols.index(ep)] = ep_count_df.loc[ep].values

            # print(ep_dur_df.loc[ep_aux_df['Label'] == ep]['Duration'])

        given_obj = file.split('_')[1]
        asked_obj = file.split('_')[2]
        family = obj_fam[given_obj]

        aux_dat_dur = np.append(dur_vec, given_obj)
        aux_dat2_dur = np.append(aux_dat_dur, asked_obj)
        aux_dat3_dur = np.append(aux_dat2_dur, family)

        new_row_dur = pd.DataFrame([aux_dat3_dur], columns=dataframe_columns)
        ep_dur = pd.concat([ep_dur, new_row_dur], ignore_index=True)

        aux_dat_count = np.append(count_vec.astype(int), given_obj)
        aux_dat2_count = np.append(aux_dat_count, asked_obj)
        aux_dat3_count = np.append(aux_dat2_count, family)

        new_row_count = pd.DataFrame([aux_dat3_count], columns=dataframe_columns)
        ep_count = pd.concat([ep_count, new_row_count], ignore_index=True)

    dtype_list_dur = list(['float64'] * len(ep_labs_cols))
    dtype_list_dur.append('object')  # for given object
    dtype_list_dur.append('object')  # for asked object
    dtype_list_dur.append('object')  # for family

    dtype_list_count = list(['int64'] * len(ep_labs_cols))
    dtype_list_count.append('object')  # for given object
    dtype_list_count.append('object')  # for asked object
    dtype_list_count.append('object')  # for family

    trial_dur = ep_dur.astype(dict(zip(ep_dur.columns, dtype_list_dur)))
    trial_count = ep_count.astype(dict(zip(ep_count.columns, dtype_list_count)))
    trial_presabs = trial_count.copy()
    [trial_presabs[x].mask(trial_presabs[x] > 0, 1, inplace=True) for x in ep_labs_cols]

    return [trial_presabs, trial_dur, trial_count]

