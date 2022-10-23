import os
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

    tasks = [name.replace(subject + '_', '').replace('.csv', '') for name in clean_name_cyberglove]

    # load each source and task
    cyberglove_list = list()
    vicon_list = list()
    emg_list = list()
    for task in tasks:

        cyberglove_open_file =  os.path.join(os.getcwd(), 'BIDSData', subject, 'cyberglove', subject + '_' + task + '_cyberglove.csv')
        vicon_open_file = os.path.join(os.getcwd(), 'BIDSData', subject, 'vicon', subject + '_' + task + '_vicon.csv')
        emg_open_file = os.path.join(os.getcwd(), 'BIDSData', subject, 'sessantaquattro', subject + '_' + task + '_sessantaquattro.csv')

        with open(cyberglove_open_file) as cyberglove_o_f:
            op_cg = csv.reader(cyberglove_o_f)
            head_cg = next(op_cg)
            rows_cg = []
            for row_cg in op_cg:
                rows_cg.append(row_cg)
        cyberglove_list.extend(rows_cg)

        with open(vicon_open_file) as vicon_o_f:
            op_vc = csv.reader(vicon_o_f)
            head_vc = next(op_vc)
            rows_vc = []
            for row_vc in op_vc:
                rows_vc.append(row_vc)
        if len(rows_vc) > len(rows_cg): # vicon sometimes has more datapoints than cyberglove, sometimes has the same
            rows_vc = rows_vc[0:len(rows_cg)] # discard last datapoint
        vicon_list.extend(rows_vc)


        with open(emg_open_file) as emg_o_f:
            op_emg = csv.reader(emg_o_f)
            head_emg = next(op_emg)
            rows_emg = []
            for row_emg in op_emg:
                rows_emg.append(row_emg)
        emg_idx = np.linspace(0, len(rows_emg) - 1, len(rows_vc)).astype(int) # downsample emg
        sel_emg_datapoints = [rows_emg[idx] for idx in emg_idx]
        emg_list.extend(sel_emg_datapoints)

    print("Shape for cyberglove dataframe:", len(cyberglove_list))
    print("Shape for vicon dataframe:", len(vicon_list))
    print("Shape for emg dataframe:", len(emg_list))
    print("******************************************************\n")

    # check why subject 2 has one more cyberglove elements that vicon
    # add labels (EPs)
    # rectify emg
    # select kinematic variables
    # append everything into a pandas dataframe
    # return dataframe


