import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():

    bins = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    l1VSl2 = [0, 0.25, 0.5, 0.75, 1]
    c_param = [0.01, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5]

    plt.close('all')  # to clean the screen

    result_file = open('./results/Raw/accuracy/raw_results.csv', 'r')

    res_cols = ['Source', 'Family', 'Bins', 'L1', 'C', 'Scores', 'Mean Score']
    res_df = pd.read_csv(result_file, names=res_cols)
    c_str = [str(x) for x in res_df['C']]
    res_df['C'] = c_str
    res_df['C'] = res_df['C'].astype('category')
    res_df['Bins'] = res_df['Bins'].astype('int')
    res_df['L1'] = res_df['L1'].astype('float')


    emg_df = res_df.loc[res_df['Source'] == 'EMG']
    kin_df = res_df.loc[res_df['Source'] == 'Kin']
    tact_df = res_df.loc[res_df['Source'] == 'Tactile']
    # multi_df = res_df.loc[res_df['Source'] == 'Multimodal']
    # hierarchical_df = res_df.loc[res_df['Source'] == 'Hierarchical']

    ###############
    # BEST ACC
    ###############

    kin_max_acc = 0
    kin_max_param = []
    emg_max_acc = 0
    emg_max_param = []
    tact_max_acc = 0
    tact_max_param = []
    # multi_max_acc = 0
    # multi_max_param = []

    for b in bins:
        for l in l1VSl2:
            for c in c_param:

                emg_sel_res = emg_df.loc[(emg_df['Bins'] == b) & (emg_df['L1'] == l) & (emg_df['C'].astype(str) == str(c))]
                emg_sel_mean_acc = emg_sel_res['Mean Score'].mean()

                if emg_sel_mean_acc > emg_max_acc:

                    emg_max_acc = emg_sel_mean_acc
                    emg_max_param = [b, l, c]

                kin_sel_res = kin_df.loc[(kin_df['Bins'] == b) & (kin_df['L1'] == l) & (kin_df['C'].astype(str) == str(c))]
                kin_sel_mean_acc = kin_sel_res['Mean Score'].mean()

                if kin_sel_mean_acc > kin_max_acc:
                    kin_max_acc = kin_sel_mean_acc
                    kin_max_param = [b, l, c]

                tact_sel_res = tact_df.loc[
                    (tact_df['Bins'] == b) & (tact_df['L1'] == l) & (tact_df['C'].astype(str) == str(c))]
                tact_sel_mean_acc = tact_sel_res['Mean Score'].mean()

                if tact_sel_mean_acc > tact_max_acc:
                    tact_max_acc = tact_sel_mean_acc
                    tact_max_param = [b, l, c]

                # multi_sel_res = multi_df.loc[(multi_df['Bins'] == b) & (multi_df['L1'] == l) & (multi_df['C'].astype(str) == str(c))]
                # multi_sel_mean_acc = multi_sel_res['Mean Score'].mean()
                #
                # if multi_sel_mean_acc > multi_max_acc:
                #     multi_max_acc = multi_sel_mean_acc
                #     multi_max_param = [b, l, c]

                # STILL MISSING HIERARCHICAL (MAYBE MAKES NO SENSE)

    print("Best accuracy for EMG data: ", round(emg_max_acc, 2), "% with", emg_max_param[0], "bins, L1 =",
          emg_max_param[1], "and C =", emg_max_param[2])
    print("Best accuracy for Kinematic data: ", round(kin_max_acc, 2), "% with", kin_max_param[0], "bins, L1 =",
          kin_max_param[1], "and C =", kin_max_param[2])
    print("Best accuracy for Tactile data: ", round(tact_max_acc, 2), "% with", tact_max_param[0], "bins, L1 =",
          tact_max_param[1], "and C =", tact_max_param[2])
    # print("Best accuracy for Multisource data: ", round(multi_max_acc, 2), "% with", multi_max_param[0], "bins, L1 =",
    #       multi_max_param[1], "and C =", multi_max_param[2])

    a=1


    ###############
    # EMG PLOTS
    ###############
    # Over Bins
    plt.figure()
    sns.lineplot(data=emg_df, x='Bins', y='Mean Score', hue='Family').set(title='EMG score over bins')
    plt.axhline(33.33, color='k', linestyle='--')
    plt.ylabel('Score (95% Confidence Interval)')
    plt.savefig('./results/Raw/plots/EMG_bins.png', dpi=600)
    # plt.show()


    # Over C
    plt.figure()
    sns.lineplot(data=emg_df, x='C', y='Mean Score', hue='Family').set(title='EMG score over C param')
    plt.axhline(33.33, color='k', linestyle='--')
    plt.ylabel('Score (95% Confidence Interval)')
    plt.savefig('./results/Raw/plots/EMG_c.png', dpi=600)
    # plt.show()

    # Over L1
    plt.figure()
    sns.lineplot(data=emg_df, x='L1', y='Mean Score', hue='Family').set(title='EMG score over L1 param')
    plt.axhline(33.33, color='k', linestyle='--')
    plt.ylabel('Score (95% Confidence Interval)')
    plt.savefig('./results/Raw/plots/EMG_l1.png', dpi=600)
    # plt.show()

    ###############
    # KIN PLOTS
    ###############
    # Over Bins
    plt.figure()
    sns.lineplot(data=kin_df, x='Bins', y='Mean Score', hue='Family').set(title='Kinematic score over bins')
    plt.axhline(33.33, color='k', linestyle='--')
    plt.ylabel('Score (95% Confidence Interval)')
    plt.savefig('./results/Raw/plots/KIN_bins.png', dpi=600)
    # plt.show()

    # Over C
    plt.figure()
    sns.lineplot(data=kin_df, x='C', y='Mean Score', hue='Family').set(title='Kinematic score over C param')
    plt.axhline(33.33, color='k', linestyle='--')
    plt.ylabel('Score (95% Confidence Interval)')
    plt.savefig('./results/Raw/plots/KIN_c.png', dpi=600)
    # plt.show()

    # Over L1
    plt.figure()
    sns.lineplot(data=kin_df, x='L1', y='Mean Score', hue='Family').set(title='Kinematic score over L1 param')
    plt.axhline(33.33, color='k', linestyle='--')
    plt.ylabel('Score (95% Confidence Interval)')
    plt.savefig('./results/Raw/plots/KIN_l1.png', dpi=600)
    # plt.show()

    ###############
    # TACTILE PLOTS
    ###############
    # Over Bins
    plt.figure()
    sns.lineplot(data=tact_df, x='Bins', y='Mean Score', hue='Family').set(title='Tactile score over bins')
    plt.axhline(33.33, color='k', linestyle='--')
    plt.ylabel('Score (95% Confidence Interval)')
    plt.savefig('./results/Raw/plots/Tact_bins.png', dpi=600)
    # plt.show()

    # Over C
    plt.figure()
    sns.lineplot(data=tact_df, x='C', y='Mean Score', hue='Family').set(title='Tactile score over C param')
    plt.axhline(33.33, color='k', linestyle='--')
    plt.ylabel('Score (95% Confidence Interval)')
    plt.savefig('./results/Raw/plots/Tact_c.png', dpi=600)
    # plt.show()

    # Over L1
    plt.figure()
    sns.lineplot(data=tact_df, x='L1', y='Mean Score', hue='Family').set(title='Tactile score over L1 param')
    plt.axhline(33.33, color='k', linestyle='--')
    plt.ylabel('Score (95% Confidence Interval)')
    plt.savefig('./results/Raw/plots/Tact_l1.png', dpi=600)
    # plt.show()

    # ###############
    # # MULTIMODAL PLOTS
    # ###############
    # # Over Bins
    # plt.figure()
    # sns.lineplot(data=multi_df, x='Bins', y='Mean Score', hue='Family').set(title='Multimodal score over bins')
    # plt.axhline(33.33, color='k', linestyle='--')
    # plt.ylabel('Score (95% Confidence Interval)')
    # plt.savefig('./results/Raw/plots/MULTI_bins.png', dpi=600)
    # # plt.show()
    #
    # # Over C
    # plt.figure()
    # sns.lineplot(data=multi_df, x='C', y='Mean Score', hue='Family').set(title='Multimodal score over C param')
    # plt.axhline(33.33, color='k', linestyle='--')
    # plt.ylabel('Score (95% Confidence Interval)')
    # plt.savefig('./results/Raw/plots/MULTI_c.png', dpi=600)
    # # plt.show()
    #
    # # Over L1
    # plt.figure()
    # sns.lineplot(data=multi_df, x='L1', y='Mean Score', hue='Family').set(title='Multimodal score over L1 param')
    # plt.axhline(33.33, color='k', linestyle='--')
    # plt.ylabel('Score (95% Confidence Interval)')
    # plt.savefig('./results/Raw/plots/MULTI_l1.png', dpi=600)
    # # plt.show()

    # ###############
    # # HIERARCHICAL PLOTS
    # ###############
    # # Over C
    # plt.figure()
    # sns.lineplot(data=multi_df, x='C', y='Mean Score', hue='Family').set(title='Hierarchical score over C param')
    # plt.axhline(33.33, color='k', linestyle='--')
    # plt.ylabel('Score (95% Confidence Interval)')
    # plt.savefig('./results/Raw/plots/HIER_c.png', dpi=600)
    # # plt.show()

if __name__ == "__main__":
    main()
