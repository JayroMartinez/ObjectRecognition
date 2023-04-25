import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    """This function evaluates the differences in the model weights between families over variables and time"""

    families = ['Mugs', 'Plates', 'Geometric', 'Cutlery', 'Ball']

    kin_cols = ['ThumbRotate', 'ThumbMPJ', 'ThumbIj', 'IndexMPJ', 'IndexPIJ',
                'MiddleMPJ', 'MiddlePIJ', 'RingMIJ', 'RingPIJ', 'PinkieMPJ',
                'PinkiePIJ', 'PalmArch', 'WristPitch', 'WristYaw', 'IndexAbd',
                'PinkieAbd', 'RingAbd', 'MiddleAbd',
                'ThumbAbd']
    flex = ['flexion_'+str(x) for x in range(0, 32)]
    ext = ['extension_' + str(x) for x in range(0, 32)]
    emg_cols = flex + ext

    tact_cols = ['rmo', 'mdo', 'rmi', 'mmo', 'pcim', 'ldd', 'rmm', 'rp', 'rdd', 'lmi', 'rdo', 'lmm', 'lp', 'rdm', 'ldm',
                 'ptip', 'idi', 'mdi', 'ido', 'mmm', 'ipi', 'mdm', 'idd', 'idm', 'imo', 'pdi', 'mmi', 'pdm', 'imm',
                 'mdd', 'pdii', 'mp', 'ptod', 'ptmd', 'tdo', 'pcid', 'imi', 'tmm', 'tdi', 'tmi', 'ptop', 'ptid', 'ptmp',
                 'tdm', 'tdd', 'tmo', 'pcip', 'ip', 'pcmp', 'rdi', 'ldi', 'lmo', 'pcmd', 'ldo', 'pdl', 'pdr', 'pdlo',
                 'lpo']

    ep_labs_cols = ['contour following', 'contour following + enclosure part',
       'edge following', 'enclosure', 'enclosure part',
       'enclosure part + function test', 'function test', 'pressure',
       'rotation', 'translation', 'weighting',
       'weighting + contour following']

    fam_obj = dict(
        Mugs = ['CeramicMug', 'Glass', 'MetalMug'],
        Plates = ['CeramicPlate', 'MetalPlate', 'PlasticPlate'],
        Geometric = ['Cube', 'Cylinder', 'Triangle'],
        Cutlery = ['Fork', 'Knife', 'Spoon'],
        Ball = ['PingPongBall', 'SquashBall', 'TennisBall']
    )

    for family in families:

        # print(family)

        ######################
        ## KIN              ##
        ######################

        # kin_w = []
        # kin_weights_file = result_file = open('./results/weights_Kin_' + family + '.csv', 'r')
        #
        # with open('./results/weights_Kin_' + family + '.csv', 'r') as kin_data:
        #
        #     kin_reader = csv.reader(kin_data)
        #     for row in kin_reader:
        #         kin_w.append(row)
        #
        # kin_w = np.asarray(kin_w, dtype=float)
        #
        # # absolute value for weights
        # kin_abs_w = np.absolute(kin_w)
        #
        # # mean over cross-validation folds
        # aux_kin_cv_1 = []
        # aux_kin_cv_2 = []
        # aux_kin_cv_3 = []
        # for i in range(0, int(len(kin_abs_w) / 3)):
        #     aux_kin_cv_1.append(kin_abs_w[3 * i])
        #     aux_kin_cv_2.append(kin_abs_w[3 * i + 1])
        #     aux_kin_cv_3.append(kin_abs_w[3 * i + 2])
        # kin_cv_w = []
        # kin_cv_w.append(np.mean(aux_kin_cv_1, axis=0))  # mean by column
        # kin_cv_w.append(np.mean(aux_kin_cv_2, axis=0))  # mean by column
        # kin_cv_w.append(np.mean(aux_kin_cv_3, axis=0))  # mean by column
        #
        # # WEIGHTS OVER TIME
        # # sum over objects
        # kin_obj_w = np.sum(kin_cv_w, axis=0)
        #
        # # sum over bins
        # kin_aux_w = np.reshape(kin_obj_w, (-1, len(kin_cols)))
        # kin_time_w = np.sum(kin_aux_w, axis=1)
        #
        # # plot for weight evolution over time bins
        # # kinematic data plot
        # plt.figure()
        # sns.lineplot(data=kin_time_w).set(title='Weight evolution over time bins for Kinematic Data\nFamily: ' + family)
        # plt.ylabel('Weight Sum')
        # plt.xlabel('Time Bins')
        # # plt.savefig('./results/time_weights_Kin_' + family + '.png', dpi=600)
        #
        # # WEIGHTS OVER VARIABLES
        # kin_bin_w = np.reshape(kin_cv_w, (-1, len(kin_cols)))
        # kin_lab = fam_obj[family] * int(len(kin_bin_w) / 3)
        # kin_w_df = pd.DataFrame(data=kin_bin_w, columns=kin_cols)
        # kin_w_df['Object'] = kin_lab
        #
        # kin_obj_w_df = kin_w_df.groupby('Object')[kin_cols].sum()
        # kin_tot_w = kin_obj_w_df.sum()
        # kin_fin_w = kin_obj_w_df.append(pd.Series(data=kin_tot_w, index=kin_cols, name='Total'))
        #
        # # heatmap weights
        # plt.figure()
        # g = sns.heatmap(data=kin_fin_w, annot=False, cmap="Greys")
        # g.set_xticklabels(labels=kin_fin_w.columns, rotation=45, size=4)
        # plt.title('Kinematic weights for family: ' + family)
        # plt.savefig('./results/var_weights_kin_' + family + '.png', dpi=600)
        # # plt.show()

        ######################
        ## EMG              ##
        ######################

        # emg_w = []
        # emg_weights_file = result_file = open('./results/weights_EMG_' + family + '.csv', 'r')
        #
        # with open('./results/weights_EMG_' + family + '.csv', 'r') as emg_data:
        #
        #     emg_reader = csv.reader(emg_data)
        #     for row in emg_reader:
        #         emg_w.append(row)

        # emg_w = np.asarray(emg_w, dtype=float)
        #
        # # absolute value for weights
        # emg_abs_w = np.absolute(emg_w)
        #
        # aux_emg_cv_1 = []
        # aux_emg_cv_2 = []
        # aux_emg_cv_3 = []
        # for i in range(0, int(len(emg_abs_w) / 3)):
        #     aux_emg_cv_1.append(emg_abs_w[3 * i])
        #     aux_emg_cv_2.append(emg_abs_w[3 * i + 1])
        #     aux_emg_cv_3.append(emg_abs_w[3 * i + 2])
        # emg_cv_w = []
        # emg_cv_w.append(np.mean(aux_emg_cv_1, axis=0))  # mean by column
        # emg_cv_w.append(np.mean(aux_emg_cv_2, axis=0))  # mean by column
        # emg_cv_w.append(np.mean(aux_emg_cv_3, axis=0))  # mean by column
        #
        # # WEIGHTS OVER TIME
        # # sum over objects
        # emg_obj_w = np.sum(emg_cv_w, axis=0)
        #
        # # sum over bins
        # emg_aux_w = np.reshape(emg_obj_w, (-1, len(emg_cols)))
        # emg_time_w = np.sum(emg_aux_w, axis=1)
        #
        # # plot for weight evolution over time bins
        # # EMG data plot
        # plt.figure()
        # sns.lineplot(data=emg_time_w).set(title='Weight evolution over time bins for EMG Data\nFamily: ' + family)
        # plt.ylabel('Weight Sum')
        # plt.xlabel('Time Bins')
        # # plt.savefig('./results/time_weights_EMG_' + family + '.png', dpi=600)
        #
        # # WEIGHTS OVER VARIABLES
        # emg_bin_w = np.reshape(emg_cv_w, (-1, len(emg_cols)))
        # emg_lab = fam_obj[family] * int(len(emg_bin_w) / 3)
        # emg_w_df = pd.DataFrame(data=emg_bin_w, columns=emg_cols)
        # emg_w_df['Object'] = emg_lab
        #
        # emg_obj_w_df = emg_w_df.groupby('Object')[emg_cols].sum()
        # emg_tot_w = emg_obj_w_df.sum()
        # emg_fin_w = emg_obj_w_df.append(pd.Series(data=emg_tot_w, index=emg_cols, name='Total'))
        #
        # # heatmap weights
        #
        # # flex_df = emg_fin_w[flex]
        # # ext_df = emg_fin_w[ext]
        # #
        # # plt.figure()
        # # h = sns.heatmap(data=flex_df, annot=False, cmap="Greys")
        # # # h.set_xticklabels(labels=flex_df.columns, rotation=45, size=2)
        # # plt.title('EMG weights for family: ' + family)
        # # plt.savefig('./results/var_weights_emg_' + family + '.png', dpi=600)
        # # # plt.show()
        #
        # plt.figure()
        # plt.pcolor(emg_fin_w, cmap='Greys')
        # plt.yticks(np.arange(0.5, len(emg_fin_w.index), 1), emg_fin_w.index)
        # plt.xticks(np.arange(0.5, len(emg_fin_w.columns), 1), emg_fin_w.columns)
        # plt.xticks(fontsize=5)
        # plt.xticks(rotation=90)
        # plt.yticks(fontsize=12)
        # plt.colorbar()
        # # plt.yticks(rotation=90)
        # plt.title('EMG weights for family: ' + family)
        # plt.savefig('./results/var_weights_emg_' + family + '.png', bbox_inches='tight', dpi=600)

        ######################
        ## TACT             ##
        ######################

        # tact_w = []
        # tact_weights_file = result_file = open('./results/weights_Tact_' + family + '.csv', 'r')
        #
        # with open('./results/weights_Tact_' + family + '.csv', 'r') as tact_data:
        #
        #     tact_reader = csv.reader(tact_data)
        #     for row in tact_reader:
        #         tact_w.append(row)
        #
        # tact_w = np.asarray(tact_w, dtype=float)
        #
        # # absolute value for weights
        # tact_abs_w = np.absolute(tact_w)
        #
        # aux_tact_cv_1 = []
        # aux_tact_cv_2 = []
        # aux_tact_cv_3 = []
        # for i in range(0, int(len(tact_abs_w) / 3)):
        #     aux_tact_cv_1.append(tact_abs_w[3 * i])
        #     aux_tact_cv_2.append(tact_abs_w[3 * i + 1])
        #     aux_tact_cv_3.append(tact_abs_w[3 * i + 2])
        # tact_cv_w = []
        # tact_cv_w.append(np.mean(aux_tact_cv_1, axis=0))  # mean by column
        # tact_cv_w.append(np.mean(aux_tact_cv_2, axis=0))  # mean by column
        # tact_cv_w.append(np.mean(aux_tact_cv_3, axis=0))  # mean by column
        #
        # # WEIGHTS OVER TIME
        # # sum over objects
        # tact_obj_w = np.sum(tact_cv_w, axis=0)
        #
        # # sum over bins
        # tact_aux_w = np.reshape(tact_obj_w, (-1, len(tact_cols)))
        # tact_time_w = np.sum(tact_aux_w, axis=1)
        #
        # # plot for weight evolution over time bins
        # # tactile data plot
        # plt.figure()
        # sns.lineplot(data=tact_time_w).set(title='Weight evolution over time bins for Tactile Data\nFamily: ' + family)
        # plt.ylabel('Weight Sum')
        # plt.xlabel('Time Bins')
        # plt.savefig('./results/time_weights_Tact_' + family + '.png', dpi=600)
        #
        # # WEIGHTS OVER VARIABLES
        # tact_bin_w = np.reshape(tact_cv_w, (-1, len(tact_cols)))
        # tact_lab = fam_obj[family] * int(len(tact_bin_w) / 3)
        # tact_w_df = pd.DataFrame(data=tact_bin_w, columns=tact_cols)
        # tact_w_df['Object'] = tact_lab
        #
        # tact_obj_w_df = tact_w_df.groupby('Object')[tact_cols].sum()
        # tact_tot_w = tact_obj_w_df.sum()
        # tact_fin_w = tact_obj_w_df.append(pd.Series(data=tact_tot_w, index=tact_cols, name='Total'))
        #
        # # heatmap weights
        # plt.figure()
        # plt.pcolor(tact_fin_w, cmap='Greys')
        # plt.yticks(np.arange(0.5, len(tact_fin_w.index), 1), tact_fin_w.index)
        # plt.xticks(np.arange(0.5, len(tact_fin_w.columns), 1), tact_fin_w.columns)
        # plt.xticks(fontsize=5)
        # plt.xticks(rotation=90)
        # plt.yticks(fontsize=12)
        # plt.colorbar()
        # # plt.yticks(rotation=90)
        # plt.title('Tact weights for family: ' + family)
        # plt.savefig('./results/var_weights_tact_' + family + '.png', bbox_inches='tight', dpi=600)

        ######################
        ## EP Labs          ##
        ######################

        # ep_labs_w = []
        # ep_labs_weights_file = result_file = open('./results/weights_EP_Labs_' + family + '.csv', 'r')
        #
        # with open('./results/weights_EP_Labs_' + family + '.csv', 'r') as ep_labs_data:
        #
        #     ep_labs_reader = csv.reader(ep_labs_data)
        #     for row in ep_labs_reader:
        #         ep_labs_w.append(row)
        #
        # ep_labs_w = np.asarray(ep_labs_w, dtype=float)
        #
        # # absolute value for weights
        # ep_labs_abs_w = np.absolute(ep_labs_w)
        #
        # aux_ep_labs_cv_1 = []
        # aux_ep_labs_cv_2 = []
        # aux_ep_labs_cv_3 = []
        # for i in range(0, int(len(ep_labs_abs_w) / 3)):
        #     aux_ep_labs_cv_1.append(ep_labs_abs_w[3 * i])
        #     aux_ep_labs_cv_2.append(ep_labs_abs_w[3 * i + 1])
        #     aux_ep_labs_cv_3.append(ep_labs_abs_w[3 * i + 2])
        # ep_labs_cv_w = []
        # ep_labs_cv_w.append(np.mean(aux_ep_labs_cv_1, axis=0))  # mean by column
        # ep_labs_cv_w.append(np.mean(aux_ep_labs_cv_2, axis=0))  # mean by column
        # ep_labs_cv_w.append(np.mean(aux_ep_labs_cv_3, axis=0))  # mean by column
        #
        # # sum over objects
        # ep_labs_obj_w = np.sum(ep_labs_cv_w, axis=0)
        #
        # # WEIGHTS OVER VARIABLES
        # ep_labs_lab = fam_obj[family] * int(len(ep_labs_cv_w) / 3)
        # ep_labs_w_df = pd.DataFrame(data=ep_labs_cv_w, columns=ep_labs_cols)
        # ep_labs_w_df['Object'] = ep_labs_lab
        #
        # ep_labs_obj_w_df = ep_labs_w_df.groupby('Object')[ep_labs_cols].sum()
        # ep_labs_tot_w = ep_labs_obj_w_df.sum()
        # ep_labs_fin_w = ep_labs_obj_w_df.append(pd.Series(data=ep_labs_tot_w, index=ep_labs_cols, name='Total'))
        #
        # # heatmap weights
        # plt.figure()
        # g = sns.heatmap(data=ep_labs_fin_w, annot=False, cmap="Greys")
        # g.set_xticklabels(labels=ep_labs_fin_w.columns, rotation=45, size=4)
        # plt.title('EP Labels weights for family: ' + family)
        # plt.tight_layout()
        # plt.savefig('./results/var_weights_ep_labs_' + family + '.png', dpi=600)
        # # plt.show()

        ######################
        ## EP Duration      ##
        ######################

        ep_dur_w = []
        ep_dur_weights_file = result_file = open('./results/weights_EP_Dur_' + family + '.csv', 'r')

        with open('./results/weights_EP_Dur_' + family + '.csv', 'r') as ep_dur_data:

            ep_dur_reader = csv.reader(ep_dur_data)
            for row in ep_dur_reader:
                ep_dur_w.append(row)

        ep_dur_w = np.asarray(ep_dur_w, dtype=float)

        # absolute value for weights
        ep_dur_abs_w = np.absolute(ep_dur_w)

        aux_ep_dur_cv_1 = []
        aux_ep_dur_cv_2 = []
        aux_ep_dur_cv_3 = []
        for i in range(0, int(len(ep_dur_abs_w) / 3)):
            aux_ep_dur_cv_1.append(ep_dur_abs_w[3 * i])
            aux_ep_dur_cv_2.append(ep_dur_abs_w[3 * i + 1])
            aux_ep_dur_cv_3.append(ep_dur_abs_w[3 * i + 2])
        ep_dur_cv_w = []
        ep_dur_cv_w.append(np.mean(aux_ep_dur_cv_1, axis=0))  # mean by column
        ep_dur_cv_w.append(np.mean(aux_ep_dur_cv_2, axis=0))  # mean by column
        ep_dur_cv_w.append(np.mean(aux_ep_dur_cv_3, axis=0))  # mean by column

        # sum over objects
        ep_dur_obj_w = np.sum(ep_dur_cv_w, axis=0)

        # WEIGHTS OVER VARIABLES
        ep_dur_lab = fam_obj[family] * int(len(ep_dur_cv_w) / 3)
        ep_dur_w_df = pd.DataFrame(data=ep_dur_cv_w, columns=ep_labs_cols)
        ep_dur_w_df['Object'] = ep_dur_lab

        ep_dur_obj_w_df = ep_dur_w_df.groupby('Object')[ep_labs_cols].sum()
        ep_dur_tot_w = ep_dur_obj_w_df.sum()
        ep_dur_fin_w = ep_dur_obj_w_df.append(pd.Series(data=ep_dur_tot_w, index=ep_labs_cols, name='Total'))

        # heatmap weights
        plt.figure()
        g = sns.heatmap(data=ep_dur_fin_w, annot=False, cmap="Greys")
        g.set_xticklabels(labels=ep_dur_fin_w.columns, rotation=45, size=4)
        plt.title('EP Labels weights for family: ' + family)
        plt.tight_layout()
        plt.savefig('./results/var_weights_ep_dur_' + family + '.png', dpi=600)
        # plt.show()

if __name__ == "__main__":
    main()
