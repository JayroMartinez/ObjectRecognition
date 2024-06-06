import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def raw_weights():
    """This function evaluates the differences in the model weights between families over variables and time"""

    families = ['Ball','Cutlery', 'Geometric', 'Mugs', 'Plates']

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

    fam_obj = dict(
        Mugs = ['CeramicMug', 'Glass', 'MetalMug'],
        Plates = ['CeramicPlate', 'MetalPlate', 'PlasticPlate'],
        Geometric = ['Cube', 'Cylinder', 'Triangle'],
        Cutlery = ['Fork', 'Knife', 'Spoon'],
        Ball = ['PingPongBall', 'SquashBall', 'TennisBall']
    )

    for family in families:

        # print(family)

        #####################
        # KIN              ##
        #####################

        kin_w = []
        kin_weights_file = result_file = open('./results/Raw/weights_Kin_' + family + '.csv', 'r')

        with open('./results/Raw/weights_Kin_' + family + '.csv', 'r') as kin_data:

            kin_reader = csv.reader(kin_data)
            for row in kin_reader:
                kin_w.append(row)

        kin_w = np.asarray(kin_w, dtype=float)

        # absolute value for weights
        kin_abs_w = np.absolute(kin_w)

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

        # WEIGHTS OVER TIME
        # sum over objects
        kin_obj_w = np.sum(kin_cv_w, axis=0)

        # sum over bins
        kin_aux_w = np.reshape(kin_obj_w, (-1, len(kin_cols)))
        kin_time_w = np.sum(kin_aux_w, axis=1)

        # plot for weight evolution over time bins
        # kinematic data plot
        plt.figure()
        sns.lineplot(data=kin_time_w).set(title='Weight evolution over time bins for Kinematic Data\nFamily: ' + family)
        plt.ylabel('Weight Sum')
        plt.xlabel('Time Bins')
        plt.savefig('./results/Raw/time_weights_Kin_' + family + '.png', dpi=600)
        plt.close()

        # WEIGHTS OVER VARIABLES
        kin_bin_w = np.reshape(kin_cv_w, (-1, len(kin_cols)))
        kin_lab = fam_obj[family] * int(len(kin_bin_w) / 3)
        kin_w_df = pd.DataFrame(data=kin_bin_w, columns=kin_cols)
        kin_w_df['Object'] = kin_lab

        kin_obj_w_df = kin_w_df.groupby('Object')[kin_cols].sum()
        kin_tot_w = kin_obj_w_df.sum()
        kin_fin_w = kin_obj_w_df.append(pd.Series(data=kin_tot_w, index=kin_cols, name='Total'))

        # heatmap weights
        plt.figure()
        g = sns.heatmap(data=kin_fin_w, annot=False, cmap="Greys")
        g.set_yticklabels(labels=g.get_yticklabels(), rotation=45, size=8)  # Adjust the 'size' as needed
        g.set_xticklabels(labels=kin_fin_w.columns, rotation=45, size=8)
        plt.title('Kinematic weights for family: ' + family)
        plt.savefig('./results/Raw/var_weights_kin_' + family + '.png', dpi=600)
        # plt.show()
        plt.close()


def ep_weights():

    families = ['Ball', 'Cutlery', 'Geometric', 'Mugs', 'Plates']

    ep_labs_cols = ['contour following',
                    'enclosure', 'enclosure part',
                    'function test', 'pressure',
                    'rotation', 'translation', 'weighting']

    fam_obj = dict(
        Mugs=['CeramicMug', 'Glass', 'MetalMug'],
        Plates=['CeramicPlate', 'MetalPlate', 'PlasticPlate'],
        Geometric=['Cube', 'Cylinder', 'Triangle'],
        Cutlery=['Fork', 'Knife', 'Spoon'],
        Ball=['PingPongBall', 'SquashBall', 'TennisBall']
    )

    for family in families:
        """Here we don't analyse the results based on the asked object, just those based on the given object"""

        ##########################
        ## Obj EP Pres/Abs      ##
        ##########################

        ep_presabs_w = []
        ep_presabs_weights_file = result_file = open('./results/EP/weights/w_giv_EP_PresAbs_' + family + '.csv', 'r')

        with open('./results/EP/weights/w_giv_EP_PresAbs_' + family + '.csv', 'r') as ep_presabs_data:

            ep_presabs_reader = csv.reader(ep_presabs_data)
            for row in ep_presabs_reader:
                ep_presabs_w.append(row)

        ep_presabs_w = np.asarray(ep_presabs_w, dtype=float)

        # absolute value for weights
        ep_presabs_abs_w = ep_presabs_w

        aux_ep_presabs_obj_1 = []
        aux_ep_presabs_obj_2 = []
        aux_ep_presabs_obj_3 = []
        for i in range(0, int(len(ep_presabs_abs_w) / 3)):
            aux_ep_presabs_obj_1.append(ep_presabs_abs_w[3 * i])
            aux_ep_presabs_obj_2.append(ep_presabs_abs_w[3 * i + 1])
            aux_ep_presabs_obj_3.append(ep_presabs_abs_w[3 * i + 2])
        ep_presabs_cv_w = []
        ep_presabs_cv_w.append(np.mean(aux_ep_presabs_obj_1, axis=0))  # mean by column
        ep_presabs_cv_w.append(np.mean(aux_ep_presabs_obj_2, axis=0))  # mean by column
        ep_presabs_cv_w.append(np.mean(aux_ep_presabs_obj_3, axis=0))  # mean by column

        # WEIGHTS OVER VARIABLES
        ep_presabs_lab = fam_obj[family] * int(len(ep_presabs_cv_w) / 3)
        ep_presabs_w_df = pd.DataFrame(data=ep_presabs_cv_w, columns=ep_labs_cols)
        ep_presabs_w_df['Object'] = ep_presabs_lab

        ep_presabs_obj_w_df = ep_presabs_w_df.groupby('Object')[ep_labs_cols].sum()

        ep_presabs_tot_w = ep_presabs_obj_w_df.abs().sum()
        ep_presabs_fin_w = pd.DataFrame(data=ep_presabs_obj_w_df)
        aux_ep_presabs_tot_w = ep_presabs_tot_w.values.reshape(-1,1)
        norm_scaler = MinMaxScaler()
        aux_ep_presabs_tot_w = norm_scaler.fit_transform(aux_ep_presabs_tot_w)
        ep_presabs_fin_w.loc['Total'] = aux_ep_presabs_tot_w.flatten()

        # heatmap weights
        center = 0
        min = ep_presabs_fin_w.min().min()
        max = ep_presabs_fin_w.max().max()
        new_margin = np.maximum(min,max)
        normalize = mcolors.TwoSlopeNorm(vcenter=center, vmin=-new_margin, vmax=new_margin)

        colorblind_palette = sns.color_palette("colorblind")
        new_colormap = sns.color_palette("viridis", as_cmap=True)  # Use a continuous colorblind-friendly colormap

        plt.figure()
        g = sns.heatmap(data=ep_presabs_fin_w, annot=False, norm=normalize, cmap=new_colormap)
        g.set_xticklabels(labels=ep_presabs_fin_w.columns, rotation=45, size=8)
        g.set_yticklabels(labels=g.get_yticklabels(), rotation=45, size=8)  # Adjust the 'size' as needed
        plt.title('Pres/Abs EP Labels weights for family: ' + family)
        plt.tight_layout()
        plt.savefig('./results/EP/plots/var_weights_giv_ep_presabs_' + family + '.png', dpi=600)
        plt.close()

        ##########################
        ## Obj EP Duration      ##
        ##########################

        ep_dur_w = []
        ep_dur_weights_file = result_file = open('./results/EP/weights/w_giv_EP_Dur_' + family + '.csv', 'r')

        with open('./results/EP/weights/w_giv_EP_Dur_' + family + '.csv', 'r') as ep_dur_data:

            ep_dur_reader = csv.reader(ep_dur_data)
            for row in ep_dur_reader:
                ep_dur_w.append(row)

        ep_dur_w = np.asarray(ep_dur_w, dtype=float)

        # absolute value for weights
        ep_dur_abs_w = ep_dur_w

        aux_ep_dur_obj_1 = []
        aux_ep_dur_obj_2 = []
        aux_ep_dur_obj_3 = []
        for i in range(0, int(len(ep_dur_abs_w) / 3)):
            aux_ep_dur_obj_1.append(ep_dur_abs_w[3 * i])
            aux_ep_dur_obj_2.append(ep_dur_abs_w[3 * i + 1])
            aux_ep_dur_obj_3.append(ep_dur_abs_w[3 * i + 2])
        ep_dur_cv_w = []
        ep_dur_cv_w.append(np.mean(aux_ep_dur_obj_1, axis=0))  # mean by column
        ep_dur_cv_w.append(np.mean(aux_ep_dur_obj_2, axis=0))  # mean by column
        ep_dur_cv_w.append(np.mean(aux_ep_dur_obj_3, axis=0))  # mean by column

        # WEIGHTS OVER VARIABLES
        ep_dur_lab = fam_obj[family] * int(len(ep_dur_cv_w) / 3)
        ep_dur_w_df = pd.DataFrame(data=ep_dur_cv_w, columns=ep_labs_cols)
        ep_dur_w_df['Object'] = ep_dur_lab

        ep_dur_obj_w_df = ep_dur_w_df.groupby('Object')[ep_labs_cols].sum()
        ep_dur_tot_w = ep_dur_obj_w_df.abs().mean()
        ep_dur_fin_w = pd.DataFrame(data=ep_dur_obj_w_df)
        aux_ep_dur_tot_w = ep_dur_tot_w.values.reshape(-1, 1)
        norm_scaler = MinMaxScaler()
        aux_ep_dur_tot_w = norm_scaler.fit_transform(aux_ep_dur_tot_w)
        ep_dur_fin_w.loc['Total'] = aux_ep_dur_tot_w.flatten()

        # heatmap weights
        center = 0
        min = ep_dur_fin_w.min().min()
        max = ep_dur_fin_w.max().max()
        new_margin = np.maximum(min, max)
        normalize = mcolors.TwoSlopeNorm(vcenter=center, vmin=-new_margin, vmax=new_margin)

        colorblind_palette = sns.color_palette("colorblind")
        new_colormap = sns.color_palette("viridis", as_cmap=True)  # Use a continuous colorblind-friendly colormap

        plt.figure()
        g = sns.heatmap(data=ep_dur_fin_w, annot=False, norm=normalize, cmap=new_colormap)
        g.set_xticklabels(labels=ep_dur_fin_w.columns, rotation=45, size=8)
        g.set_yticklabels(labels=g.get_yticklabels(), rotation=45, size=8)  # Adjust the 'size' as needed
        plt.title('Duration EP Labels weights for family: ' + family)
        plt.tight_layout()
        plt.savefig('./results/EP/plots/var_weights_giv_ep_dur_' + family + '.png', dpi=600)
        plt.close()

        ##########################
        ## Obj EP Count         ##
        ##########################

        ep_count_w = []
        ep_count_weights_file = result_file = open('./results/EP/weights/w_giv_EP_Count_' + family + '.csv', 'r')

        with open('./results/EP/weights/w_giv_EP_Count_' + family + '.csv', 'r') as ep_count_data:

            ep_count_reader = csv.reader(ep_count_data)
            for row in ep_count_reader:
                ep_count_w.append(row)

        ep_count_w = np.asarray(ep_count_w, dtype=float)

        ep_count_abs_w = ep_count_w

        aux_ep_count_obj_1 = []
        aux_ep_count_obj_2 = []
        aux_ep_count_obj_3 = []
        for i in range(0, int(len(ep_count_abs_w) / 3)):
            aux_ep_count_obj_1.append(ep_count_abs_w[3 * i])
            aux_ep_count_obj_2.append(ep_count_abs_w[3 * i + 1])
            aux_ep_count_obj_3.append(ep_count_abs_w[3 * i + 2])
        ep_count_cv_w = []
        ep_count_cv_w.append(np.mean(aux_ep_count_obj_1, axis=0))  # mean by column
        ep_count_cv_w.append(np.mean(aux_ep_count_obj_2, axis=0))  # mean by column
        ep_count_cv_w.append(np.mean(aux_ep_count_obj_3, axis=0))  # mean by column

        # WEIGHTS OVER VARIABLES
        ep_count_lab = fam_obj[family] * int(len(ep_count_cv_w) / 3)
        ep_count_w_df = pd.DataFrame(data=ep_count_cv_w, columns=ep_labs_cols)
        ep_count_w_df['Object'] = ep_count_lab

        ep_count_obj_w_df = ep_count_w_df.groupby('Object')[ep_labs_cols].sum()
        ep_count_tot_w = ep_count_obj_w_df.abs().mean()
        ep_count_fin_w = pd.DataFrame(data=ep_count_obj_w_df)
        aux_ep_count_tot_w = ep_count_tot_w.values.reshape(-1, 1)
        norm_scaler = MinMaxScaler()
        aux_ep_count_tot_w = norm_scaler.fit_transform(aux_ep_count_tot_w)
        ep_count_fin_w.loc['Total'] = aux_ep_count_tot_w.flatten()

        # heatmap weights
        center = 0
        min = ep_count_fin_w.min().min()
        max = ep_count_fin_w.max().max()
        new_margin = np.maximum(min, max)
        normalize = mcolors.TwoSlopeNorm(vcenter=center, vmin=-new_margin, vmax=new_margin)

        colorblind_palette = sns.color_palette("colorblind")
        new_colormap = sns.color_palette("viridis", as_cmap=True)  # Use a continuous colorblind-friendly colormap

        plt.figure()
        g = sns.heatmap(data=ep_count_fin_w, annot=False, norm=normalize, cmap=new_colormap)
        g.set_xticklabels(labels=ep_count_fin_w.columns, rotation=45, size=8)
        g.set_yticklabels(labels=g.get_yticklabels(), rotation=45, size=8)  # Adjust the 'size' as needed
        plt.title('Count EP Labels weights for family: ' + family)
        plt.tight_layout()
        plt.savefig('./results/EP/plots/var_weights_giv_ep_count_' + family + '.png', dpi=600)
        plt.close()

        ##########################
        ## Fam EP Pres/Abs      ##
        ##########################

        ep_presabs_w_fam = []
        ep_presabs_weights_file_fam = result_file_fam = open('./results/EP/weights/w_fam_EP_PresAbs.csv', 'r')

        with open('./results/EP/weights/w_fam_EP_PresAbs.csv', 'r') as ep_presabs_data_fam:

            ep_presabs_reader_fam = csv.reader(ep_presabs_data_fam)
            for row_fam in ep_presabs_reader_fam:
                ep_presabs_w_fam.append(row_fam)

        ep_presabs_w_fam = np.asarray(ep_presabs_w_fam, dtype=float)

        # absolute value for weights
        ep_presabs_abs_w_fam = ep_presabs_w_fam

        aux_ep_presabs_ball = []
        aux_ep_presabs_cutlery = []
        aux_ep_presabs_geometric = []
        aux_ep_presabs_mugs = []
        aux_ep_presabs_plates = []

        for i in range(0, int(len(ep_presabs_abs_w_fam) / 5)):
            aux_ep_presabs_ball.append(ep_presabs_abs_w_fam[5 * i])
            aux_ep_presabs_cutlery.append(ep_presabs_abs_w_fam[5 * i + 1])
            aux_ep_presabs_geometric.append(ep_presabs_abs_w_fam[5 * i + 2])
            aux_ep_presabs_mugs.append(ep_presabs_abs_w_fam[5 * i + 3])
            aux_ep_presabs_plates.append(ep_presabs_abs_w_fam[5 * i + 4])
        ep_presabs_cv_w_fam = []
        ep_presabs_cv_w_fam.append(np.mean(aux_ep_presabs_ball, axis=0))  # mean by column
        ep_presabs_cv_w_fam.append(np.mean(aux_ep_presabs_cutlery, axis=0))  # mean by column
        ep_presabs_cv_w_fam.append(np.mean(aux_ep_presabs_geometric, axis=0))  # mean by column
        ep_presabs_cv_w_fam.append(np.mean(aux_ep_presabs_mugs, axis=0))  # mean by column
        ep_presabs_cv_w_fam.append(np.mean(aux_ep_presabs_plates, axis=0))  # mean by column

        # WEIGHTS OVER VARIABLES
        ep_presabs_lab_fam = families * int(len(ep_presabs_cv_w_fam) / 5)
        ep_presabs_w_df_fam = pd.DataFrame(data=ep_presabs_cv_w_fam, columns=ep_labs_cols)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        ep_presabs_w_df_fam[ep_labs_cols] = scaler.fit_transform(ep_presabs_w_df_fam[ep_labs_cols])
        ep_presabs_w_df_fam['Family'] = ep_presabs_lab_fam

        ep_presabs_obj_w_df_fam = ep_presabs_w_df_fam.groupby('Family')[ep_labs_cols].sum()
        ep_presabs_tot_w_fam = ep_presabs_obj_w_df_fam.abs().mean()
        ep_presabs_fin_w_fam = pd.DataFrame(data=ep_presabs_obj_w_df_fam)
        idx = ep_presabs_fin_w_fam.index.tolist()
        idx.append('Total')
        ep_presabs_fin_w_fam.loc[len(ep_presabs_fin_w_fam.index)] = pd.Series(data=ep_presabs_tot_w_fam, index=ep_labs_cols,
                                                                  name='Total')
        ep_presabs_fin_w_fam.index = idx

        # heatmap weights
        center = 0
        min = ep_presabs_fin_w_fam.min().min()
        max = ep_presabs_fin_w_fam.max().max()
        new_margin = np.maximum(min, max)
        normalize = mcolors.TwoSlopeNorm(vcenter=center, vmin=-new_margin, vmax=new_margin)
        new_colormap = sns.color_palette("viridis", as_cmap=True)  # Use a continuous colorblind-friendly colormap

        plt.figure()
        g = sns.heatmap(data=ep_presabs_fin_w_fam, annot=False, norm=normalize, cmap=new_colormap)
        g.set_xticklabels(labels=ep_presabs_fin_w_fam.columns, rotation=45, size=8)
        g.set_yticklabels(labels=g.get_yticklabels(), rotation=45, size=8)  # Adjust the 'size' as needed
        plt.title('Pres/Abs EP Labels weights for object family')
        plt.tight_layout()
        plt.savefig('./results/EP/plots/var_weights_fam_ep_presabs_family.png', dpi=600)
        # plt.show()
        plt.close()

        ##########################
        ## Fam EP Duration      ##
        ##########################

        ep_dur_w_fam = []
        ep_dur_weights_file_fam = result_file_fam = open('./results/EP/weights/w_fam_EP_Dur.csv', 'r')

        with open('./results/EP/weights/w_fam_EP_Dur.csv', 'r') as ep_dur_data_fam:

            ep_dur_reader_fam = csv.reader(ep_dur_data_fam)
            for row_fam in ep_dur_reader_fam:
                ep_dur_w_fam.append(row_fam)

        ep_dur_w_fam = np.asarray(ep_dur_w_fam, dtype=float)

        # absolute value for weights
        ep_dur_abs_w_fam = ep_dur_w_fam

        aux_ep_dur_ball = []
        aux_ep_dur_cutlery = []
        aux_ep_dur_geometric = []
        aux_ep_dur_mugs = []
        aux_ep_dur_plates = []

        for i in range(0, int(len(ep_dur_abs_w_fam) / 5)):
            aux_ep_dur_ball.append(ep_dur_abs_w_fam[5 * i])
            aux_ep_dur_cutlery.append(ep_dur_abs_w_fam[5 * i + 1])
            aux_ep_dur_geometric.append(ep_dur_abs_w_fam[5 * i + 2])
            aux_ep_dur_mugs.append(ep_dur_abs_w_fam[5 * i + 3])
            aux_ep_dur_plates.append(ep_dur_abs_w_fam[5 * i + 4])
        ep_dur_cv_w_fam = []
        ep_dur_cv_w_fam.append(np.mean(aux_ep_dur_ball, axis=0))  # mean by column
        ep_dur_cv_w_fam.append(np.mean(aux_ep_dur_cutlery, axis=0))  # mean by column
        ep_dur_cv_w_fam.append(np.mean(aux_ep_dur_geometric, axis=0))  # mean by column
        ep_dur_cv_w_fam.append(np.mean(aux_ep_dur_mugs, axis=0))  # mean by column
        ep_dur_cv_w_fam.append(np.mean(aux_ep_dur_plates, axis=0))  # mean by column

        # WEIGHTS OVER VARIABLES
        ep_dur_lab_fam = families * int(len(ep_dur_cv_w_fam) / 5)
        ep_dur_w_df_fam = pd.DataFrame(data=ep_dur_cv_w_fam, columns=ep_labs_cols)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        ep_dur_w_df_fam[ep_labs_cols] = scaler.fit_transform(ep_dur_w_df_fam[ep_labs_cols])
        ep_dur_w_df_fam['Family'] = ep_dur_lab_fam

        ep_dur_obj_w_df_fam = ep_dur_w_df_fam.groupby('Family')[ep_labs_cols].sum()
        ep_dur_tot_w_fam = ep_dur_obj_w_df_fam.abs().mean()
        ep_dur_fin_w_fam = pd.DataFrame(data=ep_dur_obj_w_df_fam)
        idx = ep_dur_fin_w_fam.index.tolist()
        idx.append('Total')
        ep_dur_fin_w_fam.loc[len(ep_dur_fin_w_fam.index)] = pd.Series(data=ep_dur_tot_w_fam,
                                                                              index=ep_labs_cols,
                                                                              name='Total')
        ep_dur_fin_w_fam.index = idx

        # heatmap weights
        center = 0
        min = ep_dur_fin_w_fam.min().min()
        max = ep_dur_fin_w_fam.max().max()
        new_margin = np.maximum(min, max)
        normalize = mcolors.TwoSlopeNorm(vcenter=center, vmin=-new_margin, vmax=new_margin)
        new_colormap = sns.color_palette("viridis", as_cmap=True)  # Use a continuous colorblind-friendly colormap

        plt.figure()
        g = sns.heatmap(data=ep_dur_fin_w_fam, annot=False, norm=normalize, cmap=new_colormap)
        g.set_xticklabels(labels=ep_dur_fin_w_fam.columns, rotation=45, size=8)
        g.set_yticklabels(labels=g.get_yticklabels(), rotation=45, size=8)  # Adjust the 'size' as needed
        plt.title('Duration EP Labels weights for object family')
        plt.tight_layout()
        plt.savefig('./results/EP/plots/var_weights_fam_ep_dur_family.png', dpi=600)
        # plt.show()
        plt.close()

        ##########################
        ## Fam EP Count      ##
        ##########################

        ep_count_w_fam = []
        ep_count_weights_file_fam = result_file_fam = open('./results/EP/weights/w_fam_EP_Count.csv', 'r')

        with open('./results/EP/weights/w_fam_EP_Count.csv', 'r') as ep_count_data_fam:

            ep_count_reader_fam = csv.reader(ep_count_data_fam)
            for row_fam in ep_count_reader_fam:
                ep_count_w_fam.append(row_fam)

        ep_count_w_fam = np.asarray(ep_count_w_fam, dtype=float)

        # absolute value for weights
        ep_count_abs_w_fam = ep_count_w_fam

        aux_ep_count_ball = []
        aux_ep_count_cutlery = []
        aux_ep_count_geometric = []
        aux_ep_count_mugs = []
        aux_ep_count_plates = []

        for i in range(0, int(len(ep_count_abs_w_fam) / 5)):
            aux_ep_count_ball.append(ep_count_abs_w_fam[5 * i])
            aux_ep_count_cutlery.append(ep_count_abs_w_fam[5 * i + 1])
            aux_ep_count_geometric.append(ep_count_abs_w_fam[5 * i + 2])
            aux_ep_count_mugs.append(ep_count_abs_w_fam[5 * i + 3])
            aux_ep_count_plates.append(ep_count_abs_w_fam[5 * i + 4])
        ep_count_cv_w_fam = []
        ep_count_cv_w_fam.append(np.mean(aux_ep_count_ball, axis=0))  # mean by column
        ep_count_cv_w_fam.append(np.mean(aux_ep_count_cutlery, axis=0))  # mean by column
        ep_count_cv_w_fam.append(np.mean(aux_ep_count_geometric, axis=0))  # mean by column
        ep_count_cv_w_fam.append(np.mean(aux_ep_count_mugs, axis=0))  # mean by column
        ep_count_cv_w_fam.append(np.mean(aux_ep_count_plates, axis=0))  # mean by column

        # WEIGHTS OVER VARIABLES
        ep_count_lab_fam = families * int(len(ep_count_cv_w_fam) / 5)
        ep_count_w_df_fam = pd.DataFrame(data=ep_count_cv_w_fam, columns=ep_labs_cols)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        ep_count_w_df_fam[ep_labs_cols] = scaler.fit_transform(ep_count_w_df_fam[ep_labs_cols])
        ep_count_w_df_fam['Family'] = ep_count_lab_fam

        ep_count_obj_w_df_fam = ep_count_w_df_fam.groupby('Family')[ep_labs_cols].sum()
        ep_count_tot_w_fam = ep_count_obj_w_df_fam.abs().mean()
        ep_count_fin_w_fam = pd.DataFrame(data=ep_count_obj_w_df_fam)
        idx = ep_count_fin_w_fam.index.tolist()
        idx.append('Total')
        ep_count_fin_w_fam.loc[len(ep_count_fin_w_fam.index)] = pd.Series(data=ep_count_tot_w_fam,
                                                                              index=ep_labs_cols,
                                                                              name='Total')
        ep_count_fin_w_fam.index = idx

        # heatmap weights
        # generate new colormap
        center = 0
        min = ep_count_fin_w_fam.min().min()
        max = ep_count_fin_w_fam.max().max()
        new_margin = np.maximum(min, max)
        normalize = mcolors.TwoSlopeNorm(vcenter=center, vmin=-new_margin, vmax=new_margin)
        new_colormap = sns.color_palette("viridis", as_cmap=True)  # Use a continuous colorblind-friendly colormap

        plt.figure()
        g = sns.heatmap(data=ep_count_fin_w_fam, annot=False, norm=normalize, cmap=new_colormap)
        g.set_xticklabels(labels=ep_count_fin_w_fam.columns, rotation=45, size=8)
        g.set_yticklabels(labels=g.get_yticklabels(), rotation=45, size=8)  # Adjust the 'size' as needed
        plt.title('Count EP Labels weights for object family')
        plt.tight_layout()
        plt.savefig('./results/EP/plots/var_weights_fam_ep_count_family.png', dpi=600)
        # plt.show()
        plt.close()
