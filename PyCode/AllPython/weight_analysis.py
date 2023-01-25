import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

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

    fam_obj = dict(
        Mugs = ['CeramicMug', 'Glass', 'MetalMug'],
        Plates = ['CeramicPlate', 'MetalPlate', 'PlasticPlate'],
        Geometric = ['Cube', 'Cylinder', 'Triangle'],
        Cutlery = ['Fork', 'Knife', 'Spoon'],
        Ball = ['PingPongBall', 'SquashBall', 'TennisBall']
    )

    for family in families:

        kin_w = []
        emg_w = []

        # kin_weights_file = result_file = open('./results/weights_Kin_' + family + '.csv', 'r')
        # emg_weights_file = result_file = open('./results/weights_EMG_' + family + '.csv', 'r')

        with open('./results/weights_Kin_' + family + '.csv', 'r') as kin_data:

            kin_reader = csv.reader(kin_data)
            for row in kin_reader:
                kin_w.append(row)

        with open('./results/weights_EMG_' + family + '.csv', 'r') as emg_data:

            emg_reader = csv.reader(emg_data)
            for row in emg_reader:
                emg_w.append(row)

        kin_w = np.asarray(kin_w, dtype=float)
        emg_w = np.asarray(emg_w, dtype=float)

        # absolute value for weights
        kin_abs_w = np.absolute(kin_w)
        emg_abs_w = np.absolute(emg_w)

        # mean over cross-validation folds
        aux_kin_cv_1 = []
        aux_kin_cv_2 = []
        aux_kin_cv_3 = []
        for i in range(0, int(len(kin_abs_w) / 3)):
            aux_kin_cv_1.append(kin_abs_w[3 * i])
            aux_kin_cv_2.append(kin_abs_w[3 * i + 1])
            aux_kin_cv_3.append(kin_abs_w[3 * i + 2])
        kin_cv_w = []
        kin_cv_w.append(np.mean(aux_kin_cv_1, axis=0))  # mean by column
        kin_cv_w.append(np.mean(aux_kin_cv_2, axis=0))  # mean by column
        kin_cv_w.append(np.mean(aux_kin_cv_3, axis=0))  # mean by column

        aux_emg_cv_1 = []
        aux_emg_cv_2 = []
        aux_emg_cv_3 = []
        for i in range(0, int(len(emg_abs_w) / 3)):
            aux_emg_cv_1.append(emg_abs_w[3 * i])
            aux_emg_cv_2.append(emg_abs_w[3 * i + 1])
            aux_emg_cv_3.append(emg_abs_w[3 * i + 2])
        emg_cv_w = []
        emg_cv_w.append(np.mean(aux_emg_cv_1, axis=0))  # mean by column
        emg_cv_w.append(np.mean(aux_emg_cv_2, axis=0))  # mean by column
        emg_cv_w.append(np.mean(aux_emg_cv_3, axis=0))  # mean by column

        # WEIGHTS OVER TIME
        # sum over objects
        kin_obj_w = np.sum(kin_cv_w, axis=0)
        emg_obj_w = np.sum(emg_cv_w, axis=0)

        # sum over bins
        kin_aux_w = np.reshape(kin_obj_w, (-1, len(kin_cols)))
        kin_time_w = np.sum(kin_aux_w, axis=1)
        emg_aux_w = np.reshape(emg_obj_w, (-1, len(emg_cols)))
        emg_time_w = np.sum(emg_aux_w, axis=1)

        # plot for weight evolution over time bins
        # kinematic data plot
        plt.figure()
        sns.lineplot(data=kin_time_w).set(title='Weight evolution over time bins for Kinematic Data\nFamily: ' + family)
        plt.ylabel('Weight Sum')
        plt.xlabel('Time Bins')
        # plt.savefig('./results/time_weights_Kin_' + family + '.png', dpi=600)

        # EMG data plot
        plt.figure()
        sns.lineplot(data=emg_time_w).set(title='Weight evolution over time bins for EMG Data\nFamily: ' + family)
        plt.ylabel('Weight Sum')
        plt.xlabel('Time Bins')
        # plt.savefig('./results/time_weights_EMG_' + family + '.png', dpi=600)

        # WEIGHTS OVER VARIABLES
        kin_bin_w = np.reshape(kin_cv_w, (-1, len(kin_cols)))
        kin_lab = fam_obj[family] * int(len(kin_bin_w) / 3)
        kin_w_df = pd.DataFrame(data=kin_bin_w, columns=kin_cols)
        kin_w_df['Object'] = kin_lab

        emg_bin_w = np.reshape(emg_cv_w, (-1, len(emg_cols)))
        emg_lab = fam_obj[family] * int(len(emg_bin_w) / 3)
        emg_w_df = pd.DataFrame(data=emg_bin_w, columns=emg_cols)
        emg_w_df['Object'] = emg_lab

        kin_obj_w_df = kin_w_df.groupby('Object')[kin_cols].sum()
        kin_tot_w = kin_obj_w_df.sum()
        kin_fin_w = kin_obj_w_df.append(pd.Series(data=kin_tot_w, index=kin_cols, name='Total'))

        emg_obj_w_df = emg_w_df.groupby('Object')[emg_cols].sum()
        emg_tot_w = emg_obj_w_df.sum()
        emg_fin_w = emg_obj_w_df.append(pd.Series(data=emg_tot_w, index=emg_cols, name='Total'))

        # heatmap weights
        plt.figure()
        g = sns.heatmap(data=kin_fin_w, annot=False, cmap="Greys")
        g.set_xticklabels(labels=kin_fin_w.columns, rotation=45, size=4)
        plt.title('Kinematic weights for family: ' + family)
        plt.savefig('./results/var_weights_kin_' + family + '.png', dpi=600)
        # plt.show()

        # flex_df = emg_fin_w[flex]
        # ext_df = emg_fin_w[ext]
        #
        # plt.figure()
        # h = sns.heatmap(data=flex_df, annot=False, cmap="Greys")
        # # h.set_xticklabels(labels=flex_df.columns, rotation=45, size=2)
        # plt.title('EMG weights for family: ' + family)
        # plt.savefig('./results/var_weights_emg_' + family + '.png', dpi=600)
        # # plt.show()

        plt.figure()
        plt.pcolor(emg_fin_w, cmap='Greys')
        plt.yticks(np.arange(0.5, len(emg_fin_w.index), 1), emg_fin_w.index)
        plt.xticks(np.arange(0.5, len(emg_fin_w.columns), 1), emg_fin_w.columns)
        plt.xticks(fontsize=5)
        plt.xticks(rotation=90)
        plt.yticks(fontsize=12)
        plt.colorbar()
        # plt.yticks(rotation=90)
        plt.title('EMG weights for family: ' + family)
        plt.savefig('./results/var_weights_emg_' + family + '.png', bbox_inches='tight', dpi=600)
        

if __name__ == "__main__":
    main()
