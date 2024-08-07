import pandas as pd
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import itertools
import warnings
import scipy.stats as stats

def variance(data):
    """This function is to compare the differences in model weights with the differences in the raw data variables"""

    kin_cols = ['ThumbRotate', 'ThumbMPJ', 'ThumbIj', 'IndexMPJ', 'IndexPIJ',
                'MiddleMPJ', 'MiddlePIJ', 'RingMIJ', 'RingPIJ', 'PinkieMPJ',
                'PinkiePIJ', 'PalmArch', 'WristPitch', 'WristYaw', 'Index_Proj_J1_Z',
                'Pinkie_Proj_J1_Z', 'Ring_Proj_J1_Z', 'Middle_Proj_J1_Z',
                'Thumb_Proj_J1_Z']

    emg_cols = [col for col in data.columns if ('flexion' in col) or ('extension' in col)]
    flex_cols = [col for col in data.columns if 'flexion' in col]
    ext_cols = [col for col in data.columns if 'extension' in col]

    tact_cols = ['rmo', 'mdo', 'rmi', 'mmo', 'pcim', 'ldd', 'rmm', 'rp', 'rdd', 'lmi', 'rdo', 'lmm', 'lp', 'rdm',
                 'ldm', 'ptip', 'idi', 'mdi', 'ido', 'mmm', 'ipi', 'mdm', 'idd', 'idm', 'imo', 'pdi', 'mmi', 'pdm',
                 'imm', 'mdd', 'pdii', 'mp', 'ptod', 'ptmd', 'tdo', 'pcid', 'imi', 'tmm', 'tdi', 'tmi', 'ptop',
                 'ptid', 'ptmp', 'tdm', 'tdd', 'tmo', 'pcip', 'ip', 'pcmp', 'rdi', 'ldi', 'lmo', 'pcmd', 'ldo',
                 'pdl', 'pdr', 'pdlo', 'lpo']

    families = np.unique(data['Family'])

    fam_kin_acc = dict(
        Mugs=45.5,
        Plates=59.39,
        Geometric=45.76,
        Cutlery=39.68,
        Ball=59.92
    )

    fam_emg_acc = dict(
        Mugs=34.53,
        Plates=59.43,
        Geometric=45.14,
        Cutlery=34.73,
        Ball=39.92
    )

    fam_tact_acc = dict(
        Mugs=45.08,
        Plates=60.61,
        Geometric=39.65,
        Cutlery=35.97,
        Ball=62.4
    )

    # # BOXPLOT KINEMATIC WHOLE DATASET
    # g = sns.boxplot(data=data[kin_cols])
    # g.set_xticklabels(labels=kin_cols, rotation=45, size=4)
    # plt.ylabel('Raw Kinematic Data Value')
    # plt.savefig('./results/raw_boxplot_kin.png', dpi=600)

    # # BOXPLOT EMG WHOLE DATASET
    # h = sns.violinplot(data=data[emg_cols])
    # h.set_xticklabels(labels=emg_cols, rotation=45, size=4)
    # plt.ylabel('Raw EMG Data Value')
    # plt.savefig('./results/raw_boxplot_emg.png', dpi=600)
    #
    # i = sns.violinplot(data=data[flex_cols])
    # i.set_xticklabels(labels=flex_cols, rotation=45, size=4)
    # plt.ylabel('Raw EMG Flexion Data Value')
    # plt.savefig('./results/raw_boxplot_emg_flex.png', dpi=600)
    #
    # j = sns.violinplot(data=data[ext_cols])
    # j.set_xticklabels(labels=ext_cols, rotation=45, size=4)
    # plt.ylabel('Raw EMG Extension Data Value')
    # plt.savefig('./results/raw_boxplot_emg_ext.png', dpi=600)

    # # BOXPLOT TACTILE WHOLE DATASET
    # k = sns.violin(data=data[tact_cols])
    # k.set_xticklabels(labels=tact_cols, rotation=45, size=4)
    # plt.ylabel('Raw Tactile Data Value')
    # plt.savefig('./results/raw_boxplot_tact.png', dpi=600)

    for family in families:
        #############
        #############
        ## WEIGHTS
        #############
        #############

        # OPEN WEIGHT FILES

        kin_weights = []
        emg_weights = []
        tact_weights = []

        # with open('./results/weights_Kin_' + family + '.csv', 'r') as kin_data:
        #     kin_reader = csv.reader(kin_data)
        #     for row in kin_reader:
        #         kin_weights.append(row)
        #
        # with open('./results/weights_EMG_' + family + '.csv', 'r') as emg_data:
        #     emg_reader = csv.reader(emg_data)
        #     for row in emg_reader:
        #         emg_weights.append(row)

        with open('./results/weights_Tact_' + family + '.csv', 'r') as tact_data:
            tact_reader = csv.reader(tact_data)
            for row in tact_reader:
                tact_weights.append(row)


        # kin_w = np.asarray(kin_weights, dtype=float)
        # emg_w = np.asarray(emg_weights, dtype=float)
        tact_w = np.asarray(tact_weights, dtype=float)

        # obtain number of bins
        # kin_bins = int(len(kin_w[0])/19)
        # emg_bins = int(len(emg_w[0])/64)
        tact_bins = int(len(tact_w[0]) / 58)

        # absolute value for weights
        # kin_abs_w = np.absolute(kin_w)
        # emg_abs_w = np.absolute(emg_w)
        tact_abs_w = np.absolute(tact_w)

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

        aux_tact_cv_1 = []
        aux_tact_cv_2 = []
        aux_tact_cv_3 = []
        for i in range(0, int(len(tact_abs_w) / 3)):
            aux_tact_cv_1.append(tact_abs_w[3 * i])
            aux_tact_cv_2.append(tact_abs_w[3 * i + 1])
            aux_tact_cv_3.append(tact_abs_w[3 * i + 2])
        tact_cv_w = []
        tact_cv_w.append(np.mean(aux_tact_cv_1, axis=0))  # mean by column
        tact_cv_w.append(np.mean(aux_tact_cv_2, axis=0))  # mean by column
        tact_cv_w.append(np.mean(aux_tact_cv_3, axis=0))  # mean by column

        # sum over objects
        # kin_obj_weights = np.sum(kin_cv_w, axis=0)
        # emg_obj_weights = np.sum(emg_cv_w, axis=0)
        tact_obj_weights = np.sum(tact_cv_w, axis=0)

        # reshape from [num_vars * num_bins] to [num_vars] using sum()
        # aux_kin_w = np.reshape(kin_obj_weights, (-1, len(kin_cols)))
        # kin_final_weights = np.sum(aux_kin_w, axis=0)
        #
        # aux_emg_w = np.reshape(emg_obj_weights, (-1, len(emg_cols)))
        # emg_final_weights = np.sum(aux_emg_w, axis=0)

        aux_tact_w = np.reshape(tact_obj_weights, (-1, len(tact_cols)))
        tact_final_weights = np.sum(aux_tact_w, axis=0)

        # # reshape from [num_vars * num_bins] to [num_vars] using mean()
        # aux_kin_w = np.reshape(kin_obj_weights, (-1, len(kin_cols)))
        # kin_final_weights = np.mean(aux_kin_w, axis=0)
        #
        # aux_emg_w = np.reshape(emg_obj_weights, (-1, len(emg_cols)))
        # emg_final_weights = np.mean(aux_emg_w, axis=0)

        #############
        #############
        ## VARIABLE VALUES
        #############
        #############

        # SELECT & PREPROCESS RAW DATA
        selected_df = data.loc[data['Family'] == family]  # select particular family
        selected_df.dropna(axis=0, inplace=True)  # drop rows containing NaN values

        eps_to_iter = np.unique(selected_df['EP total'].values)

        # ep_labels = []
        kin_eps = []
        emg_eps = []
        tact_eps = []
        given_objects =[]

        dropped = 0

        for ep in eps_to_iter:

            ep_data = selected_df.loc[selected_df['EP total'].astype(int) == ep]

            # ep_kin_data = ep_data[kin_cols]
            # kin_in_bins = np.array_split(ep_kin_data, kin_bins)
            #
            # ep_emg_data = ep_data[emg_cols]
            # emg_in_bins = np.array_split(ep_emg_data, emg_bins)

            ep_tact_data = ep_data[tact_cols]
            tact_in_bins = np.array_split(ep_tact_data, tact_bins)

            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    # kin_bin_mean = [np.nanmean(x, axis=0) for x in kin_in_bins]  # size = [num_bins] X [sensors]
                    # flat_kin_mean = list(
                    #     itertools.chain.from_iterable(kin_bin_mean))  # size = [num_bins X sensors] (unidimensional)
                    #
                    # emg_bin_mean = [np.nanmean(x, axis=0) for x in emg_in_bins]  # size = [num_bins] X [sensors]
                    # flat_emg_mean = list(
                    #     itertools.chain.from_iterable(emg_bin_mean))  # size = [num_bins X sensors] (unidimensional)

                    tact_bin_mean = [np.nanmean(x, axis=0) for x in tact_in_bins]  # size = [num_bins] X [sensors]
                    flat_tact_mean = list(
                        itertools.chain.from_iterable(tact_bin_mean))  # size = [num_bins X sensors] (unidimensional)

                    # kin_eps.append(flat_kin_mean)
                    # emg_eps.append(flat_emg_mean)
                    tact_eps.append(flat_tact_mean)
                    given_objects.append(np.unique(ep_data['Given Object'])[0])

                except RuntimeWarning:
                    # print("Dropped EP", trn_iter, "from family ", family)
                    dropped += 1
        # print("Dropped:", dropped, "for family", family, ".", int(np.ceil(dropped/len(eps_to_iter)*100)), "%")


        # kin_df = pd.DataFrame(data=kin_eps, dtype=float)
        # kin_df['Given Object'] = given_objects
        #
        # emg_df = pd.DataFrame(data=emg_eps, dtype=float)
        # emg_df['Given Object'] = given_objects

        tact_df = pd.DataFrame(data=tact_eps, dtype=float)
        tact_df['Given Object'] = given_objects

        #############
        #############
        ## STATS FOR WHOLE FAMILY
        #############
        #############

        # for loop over kin cols

        kin_anova = []
        emg_anova = []
        tact_anova = []
        objects = np.unique(selected_df['Given Object'].values)

        # for kin_c in range(0, len(kin_df.columns)-1):
        #     fval_kin, pval_kin = stats.f_oneway(kin_df[kin_c].loc[kin_df['Given Object'] == objects[0]],
        #                                 kin_df[kin_c].loc[kin_df['Given Object'] == objects[1]],
        #                                 kin_df[kin_c].loc[kin_df['Given Object'] == objects[2]])
        #
        #     kin_anova.append(pval_kin)
        #     # kin_anova.append(fval_kin)
        #
        # for emg_c in range(0, len(emg_df.columns) - 1):
        #     fval_emg, pval_emg = stats.f_oneway(emg_df[emg_c].loc[emg_df['Given Object'] == objects[0]],
        #                                 emg_df[emg_c].loc[emg_df['Given Object'] == objects[1]],
        #                                 emg_df[emg_c].loc[emg_df['Given Object'] == objects[2]])
        #
        #     emg_anova.append(pval_emg)
        #     # emg_anova.append(fval_emg)

        for tact_c in range(0, len(tact_df.columns) - 1):
            fval_tact, pval_tact = stats.f_oneway(tact_df[tact_c].loc[tact_df['Given Object'] == objects[0]],
                                        tact_df[tact_c].loc[tact_df['Given Object'] == objects[1]],
                                        tact_df[tact_c].loc[tact_df['Given Object'] == objects[2]])

            tact_anova.append(pval_tact)
            # taact_anova.append(fval_tact)

        # resh_kin_anova = np.mean(np.reshape(kin_anova, (-1, len(kin_cols))), axis=0)
        # resh_emg_anova = np.mean(np.reshape(emg_anova, (-1, len(emg_cols))), axis=0)
        resh_tact_anova = np.mean(np.reshape(tact_anova, (-1, len(tact_cols))), axis=0)


        # kin_to_plot = pd.DataFrame({'Pval':resh_kin_anova, 'Weights':kin_final_weights})
        # emg_to_plot = pd.DataFrame({'Pval': resh_emg_anova, 'Weights': emg_final_weights})

        # kin_to_plot = pd.DataFrame({'Pval': kin_anova, 'Weights': kin_obj_weights})
        # emg_to_plot = pd.DataFrame({'Pval': resh_emg_anova, 'Weights': emg_final_weights})

        # k_wXval = kin_obj_weights * kin_df.drop(columns=['Given Object']).abs().mean(axis=0)
        # e_wXval = emg_obj_weights * emg_df.drop(columns=['Given Object']).abs().mean(axis=0)
        t_wXval = tact_obj_weights * tact_df.drop(columns=['Given Object']).abs().mean(axis=0)

        # kin_to_plot_wXval = pd.DataFrame({'Pval': kin_anova, 'Weights': k_wXval})
        # emg_to_plot_wXval = pd.DataFrame({'Pval': emg_anova, 'Weights': e_wXval})
        tact_to_plot_wXval = pd.DataFrame({'Pval': tact_anova, 'Weights': t_wXval})

        # scatter plot anova p-val vs. model weight
        # tit = 'Kinematic data, family: ' + family + str(fam_kin_acc[family])
        # plt.figure()
        # sns.scatterplot(data=kin_to_plot, x='Pval', y='Weights').set(title=tit)
        # plt.xlabel('ANOVA p-value')
        # plt.ylabel('Model Weight')
        # # plt.savefig('./results/EMG_l1.png', dpi=600)
        # plt.show()

        # # scatter plot anova p-val vs. model weight * variable mean kinematic data
        # tit = 'Kinematic data, family: ' + family + ". (Accuracy: " + str(fam_kin_acc[family]) + "%)"
        # plt.figure()
        # sns.scatterplot(data=kin_to_plot_wXval, x='Pval', y='Weights').set(title=tit)
        # plt.xlabel('ANOVA p-value')
        # plt.ylabel('Model Weight * Variable mean')
        # fig_file = './results/pVal_weight_Kin_' + family + '.png'
        # plt.savefig(fig_file, dpi=600)
        # # plt.show()
        #
        # # scatter plot anova p-val vs. model weight * variable mean EMG data
        # tit = 'EMG data, family: ' + family + ". (Accuracy: " + str(fam_kin_acc[family]) + "%)"
        # plt.figure()
        # sns.scatterplot(data=emg_to_plot_wXval, x='Pval', y='Weights').set(title=tit)
        # plt.xlabel('ANOVA p-value')
        # plt.ylabel('Model Weight * Variable mean')
        # fig_file = './results/pVal_weight_EMG_' + family + '.png'
        # plt.savefig(fig_file, dpi=600)
        # # plt.show()

        # scatter plot anova p-val vs. model weight * variable mean Tactile data
        tit = 'Tactile data, family: ' + family + ". (Accuracy: " + str(fam_tact_acc[family]) + "%)"
        plt.figure()
        sns.scatterplot(data=tact_to_plot_wXval, x='Pval', y='Weights').set(title=tit)
        plt.xlabel('ANOVA p-value')
        plt.ylabel('Model Weight * Variable mean')
        # fig_file = './results/pVal_weight_Tact_' + family + '.png'
        fig_file = './results/pVal_weight_Tact_' + family + '.svg'
        plt.savefig(fig_file, format='svg', dpi=600)
        # plt.show()



def check_kinematics(data):

    kin_cols = ['ThumbRotate', 'ThumbMPJ', 'ThumbIj', 'IndexMPJ', 'IndexPIJ',
                'MiddleMPJ', 'MiddlePIJ', 'RingMIJ', 'RingPIJ', 'PinkieMPJ',
                'PinkiePIJ', 'PalmArch', 'WristPitch', 'WristYaw', 'Index_Proj_J1_Z',
                'Pinkie_Proj_J1_Z', 'Ring_Proj_J1_Z', 'Middle_Proj_J1_Z',
                'Thumb_Proj_J1_Z']

    subjects = data['Subject'].unique()

    for subject in subjects:

        plt.clf()
        plt.close()

        selected_data = data.loc[data['Subject'] == subject]

        # g = sns.boxplot(data=selected_data[kin_cols])
        # g.set_xticklabels(labels=kin_cols, rotation=45, size=5)
        # plt.ylabel('Raw Kinematic Data Values')
        # plt.title('Subject: ' + subject)
        # plt.tight_layout()
        # # plt.show()
        # save_file = './results/checks/kinematic_data_' + subject + '.png'
        # plt.savefig(save_file, dpi=600)

        a=1



