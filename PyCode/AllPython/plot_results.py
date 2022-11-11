import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():

    bins = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    l1VSl2 = [0, 0.25, 0.5, 0.75, 1]
    c_param = [0.01, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5]
    # c_par_list = [str(x) for x in c_param]

    plt.close('all')  # to clen the screen

    result_file = open('./results/results_file.csv', 'r')  # Open file in append mode

    res_cols = ['Source', 'Family', 'Bins', 'L1', 'C', 'Scores', 'Mean Score']
    res_df = pd.read_csv(result_file, names=res_cols)
    c_str = [str(x) for x in res_df['C']]
    res_df['C'] = c_str
    res_df['C'] = res_df['C'].astype('category')

    emg_df = res_df.loc[res_df['Source'] == 'EMG']
    kin_df = res_df.loc[res_df['Source'] == 'Kin']
    multi_df = res_df.loc[res_df['Source'] == 'Multimodal']

    ###############
    # EMG PLOTS
    ###############
    # Over Bins
    plt.figure()
    sns.lineplot(data=emg_df, x='Bins', y='Mean Score', hue='Family').set(title='EMG score over bins')
    plt.axhline(33.33, color='k', linestyle='--')
    plt.ylabel('Score (95% Confidence Interval)')
    plt.savefig('./results/EMG_bins.png')
    # plt.show()


    # Over C
    plt.figure()
    sns.lineplot(data=emg_df, x='C', y='Mean Score', hue='Family').set(title='EMG score over C param')
    plt.axhline(33.33, color='k', linestyle='--')
    plt.ylabel('Score (95% Confidence Interval)')
    plt.savefig('./results/EMG_c.png')
    # plt.show()

    # Over L1
    plt.figure()
    sns.lineplot(data=emg_df, x='L1', y='Mean Score', hue='Family').set(title='EMG score over L1 param')
    plt.axhline(33.33, color='k', linestyle='--')
    plt.ylabel('Score (95% Confidence Interval)')
    plt.savefig('./results/EMG_l1.png')
    # plt.show()

    ###############
    # KIN PLOTS
    ###############
    # Over Bins
    plt.figure()
    sns.lineplot(data=kin_df, x='Bins', y='Mean Score', hue='Family').set(title='Kinematic score over bins')
    plt.axhline(33.33, color='k', linestyle='--')
    plt.ylabel('Score (95% Confidence Interval)')
    plt.savefig('./results/KIN_bins.png')
    # plt.show()

    # Over C
    plt.figure()
    sns.lineplot(data=kin_df, x='C', y='Mean Score', hue='Family').set(title='Kinematic score over C param')
    plt.axhline(33.33, color='k', linestyle='--')
    plt.ylabel('Score (95% Confidence Interval)')
    plt.savefig('./results/KIN_c.png')
    # plt.show()

    # Over L1
    plt.figure()
    sns.lineplot(data=kin_df, x='L1', y='Mean Score', hue='Family').set(title='Kinematic score over L1 param')
    plt.axhline(33.33, color='k', linestyle='--')
    plt.ylabel('Score (95% Confidence Interval)')
    plt.savefig('./results/KIN_l1.png')
    # plt.show()

    ###############
    # MULTIMODAL PLOTS
    ###############
    # Over Bins
    plt.figure()
    sns.lineplot(data=multi_df, x='Bins', y='Mean Score', hue='Family').set(title='Multimodal score over bins')
    plt.axhline(33.33, color='k', linestyle='--')
    plt.ylabel('Score (95% Confidence Interval)')
    plt.savefig('./results/MULTI_bins.png')
    # plt.show()

    # Over C
    plt.figure()
    sns.lineplot(data=multi_df, x='C', y='Mean Score', hue='Family').set(title='Multimodal score over C param')
    plt.axhline(33.33, color='k', linestyle='--')
    plt.ylabel('Score (95% Confidence Interval)')
    plt.savefig('./results/MULTI_c.png')
    # plt.show()

    # Over L1
    plt.figure()
    sns.lineplot(data=multi_df, x='L1', y='Mean Score', hue='Family').set(title='Multimodal score over L1 param')
    plt.axhline(33.33, color='k', linestyle='--')
    plt.ylabel('Score (95% Confidence Interval)')
    plt.savefig('./results/MULTI_l1.png')
    # plt.show()


    a=1


if __name__ == "__main__":
    main()

