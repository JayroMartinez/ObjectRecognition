import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from statannotations.Annotator import Annotator
from statannot import add_stat_annotation, statannot


def main():

    bins = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    l1VSl2 = [0, 0.25, 0.5, 0.75, 1]
    c_param = [0.01, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5]

    kinds = ["Kin", "EMG", "Tactile", "Multimodal", "Multimodal_KE", "Multimodal_KT", "Multimodal_ET", "Hierarchical", "Hierarchical_KE", "Hierarchical_KT", "Hierarchical_ET"]
    pairs = list(combinations(kinds, r=2))

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
    multi_df = res_df.loc[res_df['Source'] == 'Multimodal']
    multi_KE_df = res_df.loc[res_df['Source'] == 'Multimodal_KE']
    multi_KT_df = res_df.loc[res_df['Source'] == 'Multimodal_KT']
    multi_ET_df = res_df.loc[res_df['Source'] == 'Multimodal_ET']
    hierarchical_df = res_df.loc[res_df['Source'] == 'Hierarchical']
    hierarchical_KE_df = res_df.loc[res_df['Source'] == 'Hierarchical_KE']
    hierarchical_KT_df = res_df.loc[res_df['Source'] == 'Hierarchical_KT']
    hierarchical_ET_df = res_df.loc[res_df['Source'] == 'Hierarchical_ET']

    ###############
    # BEST ACC
    ###############

    kin_max_acc = 0
    kin_max_param = []
    emg_max_acc = 0
    emg_max_param = []
    tact_max_acc = 0
    tact_max_param = []
    multi_max_acc = 0
    multi_max_param = []
    multi_KE_max_acc = 0
    multi_KE_max_param = []
    multi_KT_max_acc = 0
    multi_KT_max_param = []
    multi_ET_max_acc = 0
    multi_ET_max_param = []
    hier_max_acc = 0
    hier_max_c = []
    hier_KE_max_acc = 0
    hier_KE_max_c = []
    hier_KT_max_acc = 0
    hier_KT_max_c = []
    hier_ET_max_acc = 0
    hier_ET_max_c = []

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

                multi_sel_res = multi_df.loc[(multi_df['Bins'] == b) & (multi_df['L1'] == l) & (multi_df['C'].astype(str) == str(c))]
                multi_sel_mean_acc = multi_sel_res['Mean Score'].mean()

                if multi_sel_mean_acc > multi_max_acc:
                    multi_max_acc = multi_sel_mean_acc
                    multi_max_param = [b, l, c]

                multi_KE_sel_res = multi_KE_df.loc[
                    (multi_KE_df['Bins'] == b) & (multi_KE_df['L1'] == l) & (multi_KE_df['C'].astype(str) == str(c))]
                multi_KE_sel_mean_acc = multi_KE_sel_res['Mean Score'].mean()

                if multi_KE_sel_mean_acc > multi_KE_max_acc:
                    multi_KE_max_acc = multi_KE_sel_mean_acc
                    multi_KE_max_param = [b, l, c]

                multi_KT_sel_res = multi_KT_df.loc[
                    (multi_KT_df['Bins'] == b) & (multi_KT_df['L1'] == l) & (multi_KT_df['C'].astype(str) == str(c))]
                multi_KT_sel_mean_acc = multi_KT_sel_res['Mean Score'].mean()

                if multi_KT_sel_mean_acc > multi_KT_max_acc:
                    multi_KT_max_acc = multi_KT_sel_mean_acc
                    multi_KT_max_param = [b, l, c]

                multi_ET_sel_res = multi_ET_df.loc[
                    (multi_ET_df['Bins'] == b) & (multi_ET_df['L1'] == l) & (multi_ET_df['C'].astype(str) == str(c))]
                multi_ET_sel_mean_acc = multi_ET_sel_res['Mean Score'].mean()

                if multi_ET_sel_mean_acc > multi_ET_max_acc:
                    multi_ET_max_acc = multi_ET_sel_mean_acc
                    multi_ET_max_param = [b, l, c]

    for c in c_param:

        hier_sel_res = hierarchical_df.loc[
            (hierarchical_df['C'].astype(str) == str(c))]
        hier_sel_mean_acc = hier_sel_res['Mean Score'].mean()

        if hier_sel_mean_acc > hier_max_acc:
            hier_max_acc = hier_sel_mean_acc
            hier_max_c = c

        hier_KE_sel_res = hierarchical_KE_df.loc[
            (hierarchical_KE_df['C'].astype(str) == str(c))]
        hier_KE_sel_mean_acc = hier_KE_sel_res['Mean Score'].mean()

        if hier_KE_sel_mean_acc > hier_KE_max_acc:
            hier_KE_max_acc = hier_KE_sel_mean_acc
            hier_KE_max_c = c

        hier_KT_sel_res = hierarchical_KT_df.loc[
            (hierarchical_KT_df['C'].astype(str) == str(c))]
        hier_KT_sel_mean_acc = hier_KT_sel_res['Mean Score'].mean()

        if hier_KT_sel_mean_acc > hier_KT_max_acc:
            hier_KT_max_acc = hier_KT_sel_mean_acc
            hier_KT_max_c = c

        hier_ET_sel_res = hierarchical_ET_df.loc[
            (hierarchical_ET_df['C'].astype(str) == str(c))]
        hier_ET_sel_mean_acc = hier_ET_sel_res['Mean Score'].mean()

        if hier_ET_sel_mean_acc > hier_ET_max_acc:
            hier_ET_max_acc = hier_ET_sel_mean_acc
            hier_ET_max_c = c

    # a=1



    # print("Best accuracy for EMG data: ", round(emg_max_acc, 2), "% with", emg_max_param[0], "bins, L1 =",
    #       emg_max_param[1], "and C =", emg_max_param[2])
    # print("Best accuracy for Kinematic data: ", round(kin_max_acc, 2), "% with", kin_max_param[0], "bins, L1 =",
    #       kin_max_param[1], "and C =", kin_max_param[2])
    # print("Best accuracy for Tactile data: ", round(tact_max_acc, 2), "% with", tact_max_param[0], "bins, L1 =",
    #       tact_max_param[1], "and C =", tact_max_param[2])
    # print("Best accuracy for Multisource data: ", round(multi_max_acc, 2), "% with", multi_max_param[0], "bins, L1 =",
    #       multi_max_param[1], "and C =", multi_max_param[2])
    # print("Best accuracy for Multisource KE data: ", round(multi_KE_max_acc, 2), "% with", multi_KE_max_param[0], "bins, L1 =",
    #       multi_KE_max_param[1], "and C =", multi_KE_max_param[2])
    # print("Best accuracy for Multisource KT data: ", round(multi_KT_max_acc, 2), "% with", multi_KT_max_param[0], "bins, L1 =",
    #       multi_KT_max_param[1], "and C =", multi_KT_max_param[2])
    # print("Best accuracy for Multisource ET data: ", round(multi_ET_max_acc, 2), "% with", multi_ET_max_param[0], "bins, L1 =",
    #       multi_ET_max_param[1], "and C =", multi_ET_max_param[2])
    # print("Best accuracy for Hierarchical data: ", round(hier_max_acc, 2), "% with C = ", hier_max_c)
    # print("Best accuracy for Hierarchical KE data: ", round(hier_KE_max_acc, 2), "% with C = ", hier_KE_max_c)
    # print("Best accuracy for Hierarchical KT data: ", round(hier_KT_max_acc, 2), "% with C = ", hier_KT_max_c)
    # print("Best accuracy for Hierarchical ET data: ", round(hier_ET_max_acc, 2), "% with C = ", hier_ET_max_c)



    # ###############
    # # EMG PLOTS
    # ###############
    # # Over Bins
    # plt.figure()
    # sns.lineplot(data=emg_df, x='Bins', y='Mean Score', hue='Family').set(title='EMG score over bins')
    # plt.axhline(33.33, color='k', linestyle='--')
    # plt.ylabel('Score (95% Confidence Interval)')
    # plt.savefig('./results/Raw/plots/EMG_bins.png', dpi=600)
    # # plt.show()
    # plt.close()


    # # Over C
    # plt.figure()
    # sns.lineplot(data=emg_df, x='C', y='Mean Score', hue='Family').set(title='EMG score over C param')
    # plt.axhline(33.33, color='k', linestyle='--')
    # plt.ylabel('Score (95% Confidence Interval)')
    # plt.savefig('./results/Raw/plots/EMG_c.png', dpi=600)
    # # plt.show()
    # plt.close()

    # # Over L1
    # plt.figure()
    # sns.lineplot(data=emg_df, x='L1', y='Mean Score', hue='Family').set(title='EMG score over L1 param')
    # plt.axhline(33.33, color='k', linestyle='--')
    # plt.ylabel('Score (95% Confidence Interval)')
    # plt.savefig('./results/Raw/plots/EMG_l1.png', dpi=600)
    # # plt.show()
    # plt.close()

    # ###############
    # # KIN PLOTS
    # ###############
    # # Over Bins
    # plt.figure()
    # sns.lineplot(data=kin_df, x='Bins', y='Mean Score', hue='Family').set(title='Kinematic score over bins')
    # plt.axhline(33.33, color='k', linestyle='--')
    # plt.ylabel('Score (95% Confidence Interval)')
    # plt.savefig('./results/Raw/plots/KIN_bins.png', dpi=600)
    # # plt.show()
    # plt.close()

    # # Over C
    # plt.figure()
    # sns.lineplot(data=kin_df, x='C', y='Mean Score', hue='Family').set(title='Kinematic score over C param')
    # plt.axhline(33.33, color='k', linestyle='--')
    # plt.ylabel('Score (95% Confidence Interval)')
    # plt.savefig('./results/Raw/plots/KIN_c.png', dpi=600)
    # # plt.show()
    # plt.close()

    # # Over L1
    # plt.figure()
    # sns.lineplot(data=kin_df, x='L1', y='Mean Score', hue='Family').set(title='Kinematic score over L1 param')
    # plt.axhline(33.33, color='k', linestyle='--')
    # plt.ylabel('Score (95% Confidence Interval)')
    # plt.savefig('./results/Raw/plots/KIN_l1.png', dpi=600)
    # # plt.show()
    # plt.close()

    # ###############
    # # TACTILE PLOTS
    # ###############
    # # Over Bins
    # plt.figure()
    # sns.lineplot(data=tact_df, x='Bins', y='Mean Score', hue='Family').set(title='Tactile score over bins')
    # plt.axhline(33.33, color='k', linestyle='--')
    # plt.ylabel('Score (95% Confidence Interval)')
    # plt.savefig('./results/Raw/plots/Tact_bins.png', dpi=600)
    # # plt.show()
    # plt.close()

    # # Over C
    # plt.figure()
    # sns.lineplot(data=tact_df, x='C', y='Mean Score', hue='Family').set(title='Tactile score over C param')
    # plt.axhline(33.33, color='k', linestyle='--')
    # plt.ylabel('Score (95% Confidence Interval)')
    # plt.savefig('./results/Raw/plots/Tact_c.png', dpi=600)
    # # plt.show()
    # plt.close()

    # # Over L1
    # plt.figure()
    # sns.lineplot(data=tact_df, x='L1', y='Mean Score', hue='Family').set(title='Tactile score over L1 param')
    # plt.axhline(33.33, color='k', linestyle='--')
    # plt.ylabel('Score (95% Confidence Interval)')
    # plt.savefig('./results/Raw/plots/Tact_l1.png', dpi=600)
    # # plt.show()
    # plt.close()

    # ###############
    # # MULTIMODAL PLOTS
    # ###############
    # # Over Bins
    # plt.figure()
    # sns.lineplot(data=multi_df, x='Bins', y='Mean Score', hue='Family').set(title='Multimodal All score over bins')
    # plt.axhline(33.33, color='k', linestyle='--')
    # plt.ylabel('Score (95% Confidence Interval)')
    # plt.savefig('./results/Raw/plots/MULTI_bins.png', dpi=600)
    # # plt.show()
    # plt.close()

    # # Over C
    # plt.figure()
    # sns.lineplot(data=multi_df, x='C', y='Mean Score', hue='Family').set(title='Multimodal All score over C param')
    # plt.axhline(33.33, color='k', linestyle='--')
    # plt.ylabel('Score (95% Confidence Interval)')
    # plt.savefig('./results/Raw/plots/MULTI_c.png', dpi=600)
    # # plt.show()
    # plt.close()

    # # Over L1
    # plt.figure()
    # sns.lineplot(data=multi_df, x='L1', y='Mean Score', hue='Family').set(title='Multimodal All score over L1 param')
    # plt.axhline(33.33, color='k', linestyle='--')
    # plt.ylabel('Score (95% Confidence Interval)')
    # plt.savefig('./results/Raw/plots/MULTI_l1.png', dpi=600)
    # # plt.show()
    # plt.close()

    # ###############
    # # MULTIMODAL KE PLOTS
    # ###############
    # # Over Bins
    # plt.figure()
    # sns.lineplot(data=multi_KE_df, x='Bins', y='Mean Score', hue='Family').set(title='Multimodal KE score over bins')
    # plt.axhline(33.33, color='k', linestyle='--')
    # plt.ylabel('Score (95% Confidence Interval)')
    # plt.savefig('./results/Raw/plots/MULTI_KE_bins.png', dpi=600)
    # # plt.show()
    # plt.close()

    # # Over C
    # plt.figure()
    # sns.lineplot(data=multi_KE_df, x='C', y='Mean Score', hue='Family').set(title='Multimodal KE score over C param')
    # plt.axhline(33.33, color='k', linestyle='--')
    # plt.ylabel('Score (95% Confidence Interval)')
    # plt.savefig('./results/Raw/plots/MULTI_KE_c.png', dpi=600)
    # # plt.show()
    # plt.close()

    # # Over L1
    # plt.figure()
    # sns.lineplot(data=multi_KE_df, x='L1', y='Mean Score', hue='Family').set(title='Multimodal KE score over L1 param')
    # plt.axhline(33.33, color='k', linestyle='--')
    # plt.ylabel('Score (95% Confidence Interval)')
    # plt.savefig('./results/Raw/plots/MULTI_KE_l1.png', dpi=600)
    # # plt.show()
    # plt.close()

    # ###############
    # # MULTIMODAL KT PLOTS
    # ###############
    # # Over Bins
    # plt.figure()
    # sns.lineplot(data=multi_KT_df, x='Bins', y='Mean Score', hue='Family').set(title='Multimodal KT score over bins')
    # plt.axhline(33.33, color='k', linestyle='--')
    # plt.ylabel('Score (95% Confidence Interval)')
    # plt.savefig('./results/Raw/plots/MULTI_KT_bins.png', dpi=600)
    # # plt.show()
    # plt.close()

    # # Over C
    # plt.figure()
    # sns.lineplot(data=multi_KT_df, x='C', y='Mean Score', hue='Family').set(title='Multimodal KT score over C param')
    # plt.axhline(33.33, color='k', linestyle='--')
    # plt.ylabel('Score (95% Confidence Interval)')
    # plt.savefig('./results/Raw/plots/MULTI_KT_c.png', dpi=600)
    # # plt.show()
    # plt.close()

    # # Over L1
    # plt.figure()
    # sns.lineplot(data=multi_KT_df, x='L1', y='Mean Score', hue='Family').set(title='Multimodal KT score over L1 param')
    # plt.axhline(33.33, color='k', linestyle='--')
    # plt.ylabel('Score (95% Confidence Interval)')
    # plt.savefig('./results/Raw/plots/MULTI_KT_l1.png', dpi=600)
    # # plt.show()
    # plt.close()

    # ###############
    # # MULTIMODAL ET PLOTS
    # ###############
    # # Over Bins
    # plt.figure()
    # sns.lineplot(data=multi_ET_df, x='Bins', y='Mean Score', hue='Family').set(title='Multimodal ET score over bins')
    # plt.axhline(33.33, color='k', linestyle='--')
    # plt.ylabel('Score (95% Confidence Interval)')
    # plt.savefig('./results/Raw/plots/MULTI_ET_bins.png', dpi=600)
    # # plt.show()
    # plt.close()

    # # Over C
    # plt.figure()
    # sns.lineplot(data=multi_ET_df, x='C', y='Mean Score', hue='Family').set(title='Multimodal ET score over C param')
    # plt.axhline(33.33, color='k', linestyle='--')
    # plt.ylabel('Score (95% Confidence Interval)')
    # plt.savefig('./results/Raw/plots/MULTI_ET_c.png', dpi=600)
    # # plt.show()
    # plt.close()
    #
    # # Over L1
    # plt.figure()
    # sns.lineplot(data=multi_ET_df, x='L1', y='Mean Score', hue='Family').set(title='Multimodal ET score over L1 param')
    # plt.axhline(33.33, color='k', linestyle='--')
    # plt.ylabel('Score (95% Confidence Interval)')
    # plt.savefig('./results/Raw/plots/MULTI_ET_l1.png', dpi=600)
    # # plt.show()
    # plt.close()

    # ###############
    # # HIERARCHICAL PLOTS
    # ###############
    # # Over C
    # plt.figure()
    # sns.lineplot(data=hierarchical_df, x='C', y='Mean Score', hue='Family').set(title='Hierarchical All score over C param')
    # plt.axhline(33.33, color='k', linestyle='--')
    # plt.ylabel('Score (95% Confidence Interval)')
    # plt.savefig('./results/Raw/plots/HIER_c.png', dpi=600)
    # # plt.show()
    # plt.close()

    # ###############
    # # HIERARCHICAL KE PLOTS
    # ###############
    # # Over C
    # plt.figure()
    # sns.lineplot(data=hierarchical_KE_df, x='C', y='Mean Score', hue='Family').set(title='Hierarchical KE score over C param')
    # plt.axhline(33.33, color='k', linestyle='--')
    # plt.ylabel('Score (95% Confidence Interval)')
    # plt.savefig('./results/Raw/plots/HIER_KE_c.png', dpi=600)
    # # plt.show()
    # plt.close()

    # ###############
    # # HIERARCHICAL KT PLOTS
    # ###############
    # # Over C
    # plt.figure()
    # sns.lineplot(data=hierarchical_KT_df, x='C', y='Mean Score', hue='Family').set(title='Hierarchical KT score over C param')
    # plt.axhline(33.33, color='k', linestyle='--')
    # plt.ylabel('Score (95% Confidence Interval)')
    # plt.savefig('./results/Raw/plots/HIER_KT_c.png', dpi=600)
    # # plt.show()
    # plt.close()

    # ###############
    # # HIERARCHICAL ET PLOTS
    # ###############
    # # Over C
    # plt.figure()
    # sns.lineplot(data=hierarchical_ET_df, x='C', y='Mean Score', hue='Family').set(title='Hierarchical ET score over C param')
    # plt.axhline(33.33, color='k', linestyle='--')
    # plt.ylabel('Score (95% Confidence Interval)')
    # plt.savefig('./results/Raw/plots/HIER_ET_c.png', dpi=600)
    # # plt.show()
    # plt.close()

    # ###############
    # # ALL COMPARISON PLOTS
    # ###############
    # all_sources_df = pd.concat([emg_df, kin_df, tact_df, multi_df, multi_KE_df, multi_KT_df, multi_ET_df, hierarchical_df, hierarchical_KE_df, hierarchical_KT_df, hierarchical_ET_df])
    # plt.figure()
    # i = sns.barplot(data=all_sources_df, x="Source", y="Mean Score")
    # # sns.set(rc={'figure.figsize': (234, 165)})
    # # annotator_i = Annotator(i, pairs, data=all_sources_df, x="Source", y="Mean Score")
    # # annotator_i.configure(test="Mann-Whitney", text_format="star", show_test_name=False)
    # # annotator_i.apply_and_annotate()
    # i.set(ylabel="Accuracy (95% ci)")
    # i.set(xlabel=None)
    # plt.xticks(rotation=45, size=4)
    # # i.axhline(20, color='r')
    # plt.title('Given Object classification accuracy comparison')
    # plt.savefig('./results/Raw/plots/class_acc_source_comp.png', dpi=600)
    # plt.tight_layout()
    # # plt.show()
    # plt.close()

    ###############
    # ALL  vs Multi_KE COMPARISON PLOTS
    ###############
    all_sources_df = pd.concat([emg_df, kin_df, tact_df, multi_df, multi_KE_df, multi_KT_df, multi_ET_df, hierarchical_df, hierarchical_KE_df, hierarchical_KT_df, hierarchical_ET_df])
    plt.figure()
    i = sns.barplot(data=all_sources_df, x="Source", y="Mean Score")
    pairs_vs_multKE = [("Multimodal_KE","Hierarchical"),("Multimodal_KE","Hierarchical_KT"),("Multimodal_KE","Hierarchical_KE")]
    annotator_i = Annotator(i, pairs_vs_multKE, data=all_sources_df, x="Source", y="Mean Score")
    annotator_i.configure(test="Mann-Whitney", text_format="simple", show_test_name=False)
    annotator_i.apply_and_annotate()
    i.set(ylabel="Accuracy (95% ci)")
    i.set(xlabel=None)
    plt.xticks(rotation=45, size=4)
    # i.axhline(20, color='r')
    plt.title('Given Object classification accuracy comparison')
    plt.savefig('./results/Raw/plots/class_acc_vs_multiKE_comp.png', dpi=600)
    plt.tight_layout()
    # plt.show()
    plt.close()

    ###############
    # ALL  vs Kinematic COMPARISON PLOTS
    ###############
    all_sources_df = pd.concat(
        [emg_df, kin_df, tact_df, multi_df, multi_KE_df, multi_KT_df, multi_ET_df, hierarchical_df, hierarchical_KE_df,
         hierarchical_KT_df, hierarchical_ET_df])
    plt.figure()
    i = sns.barplot(data=all_sources_df, x="Source", y="Mean Score")
    pairs_vs_kin = [("Kin", "Hierarchical"), ("Kin", "Hierarchical_KT"),
                     ("Kin", "Hierarchical_KE")]
    annotator_i = Annotator(i, pairs_vs_kin, data=all_sources_df, x="Source", y="Mean Score")
    annotator_i.configure(test="Mann-Whitney", text_format="simple", show_test_name=False)
    annotator_i.apply_and_annotate()
    i.set(ylabel="Accuracy (95% ci)")
    i.set(xlabel=None)
    plt.xticks(rotation=45, size=4)
    # i.axhline(20, color='r')
    plt.title('Given Object classification accuracy comparison')
    plt.savefig('./results/Raw/plots/class_acc_vs_Kin_comp.png', dpi=600)
    plt.tight_layout()
    # plt.show()
    plt.close()

    # ##############
    # # COMPARISON MULTI PLOTS
    # ###############
    # kinds_multi = ["Multimodal", "Multimodal_KE", "Multimodal_KT", "Multimodal_ET"]
    # pairs_multi = list(combinations(kinds_multi, r=2))
    # all_sources_df = pd.concat([multi_df, multi_KE_df, multi_KT_df, multi_ET_df])
    # plt.figure()
    # i = sns.barplot(data=all_sources_df, x="Source", y="Mean Score")
    # annotator_i = Annotator(i, pairs_multi, data=all_sources_df, x="Source", y="Mean Score")
    # annotator_i.configure(test="Mann-Whitney", text_format="simple", show_test_name=False)
    # annotator_i.apply_and_annotate()
    # i.set(ylabel="Accuracy (95% ci)")
    # # i.axhline(20, color='r')
    # plt.title('Given Object classification accuracy multimodal comparison')
    # plt.savefig('./results/Raw/plots/class_acc_multi_comp.png', dpi=600)
    # # plt.show()
    # plt.close()

    # ##############
    # # COMPARISON HIER PLOTS
    # ###############
    # kinds_multi = ["Hierarchical", "Hierarchical_KE", "Hierarchical_KT", "Hierarchical_ET"]
    # pairs_multi = list(combinations(kinds_multi, r=2))
    # all_sources_df = pd.concat([hierarchical_df, hierarchical_KE_df, hierarchical_KT_df, hierarchical_ET_df])
    # plt.figure()
    # i = sns.barplot(data=all_sources_df, x="Source", y="Mean Score")
    # annotator_i = Annotator(i, pairs_multi, data=all_sources_df, x="Source", y="Mean Score")
    # annotator_i.configure(test="Mann-Whitney", text_format="simple", show_test_name=False)
    # annotator_i.apply_and_annotate()
    # i.set(ylabel="Accuracy (95% ci)")
    # # i.axhline(20, color='r')
    # plt.title('Given Object classification accuracy hierarchical comparison')
    # plt.savefig('./results/Raw/plots/class_acc_hier_comp.png', dpi=600)
    # # plt.show()
    # plt.close()

    # ##############
    # # COMPARISON MULTI HIER PLOTS
    # ###############
    # # kinds_mulhier = ["Multimodal_KE", "Multimodal_KT", "Multimodal_ET","Hierarchical_KE", "Hierarchical_KT", "Hierarchical_ET"]
    # pairs_mulhier = [("Multimodal_KE","Hierarchical_KE"),("Multimodal_KT","Hierarchical_KT"),("Multimodal_ET","Hierarchical_ET"),]
    # all_sources_df = pd.concat([multi_KE_df, multi_KT_df, multi_ET_df, hierarchical_KE_df, hierarchical_KT_df, hierarchical_ET_df])
    # plt.figure()
    # i = sns.barplot(data=all_sources_df, x="Source", y="Mean Score")
    # # sns.set(font_scale=0.5)
    # annotator_i = Annotator(i, pairs_mulhier, data=all_sources_df, x="Source", y="Mean Score")
    # annotator_i.configure(test="Mann-Whitney", text_format="simple", show_test_name=False)
    # annotator_i.apply_and_annotate()
    # i.set(ylabel="Accuracy (95% ci)")
    # i.set(xlabel=None)
    # # i.axhline(20, color='r')
    # plt.xticks(size=6)
    # plt.title('Given Object classification accuracy multimodal/hierarchical comparison')
    # plt.savefig('./results/Raw/plots/class_acc_mult_hier_comp.png', dpi=600)
    # # plt.show()
    # plt.close()

    # ##############
    # # COMPARISON KE PLOTS
    # ###############
    # pairs_ke = [("Kin","Multimodal_KE"),("Kin","Hierarchical_KE"),("EMG","Multimodal_KE"),("EMG","Hierarchical_KE"),("Multimodal_KE","Hierarchical_KE")]
    # all_sources_df = pd.concat([kin_df, emg_df, multi_KE_df, hierarchical_KE_df])
    # plt.figure()
    # i = sns.barplot(data=all_sources_df, x="Source", y="Mean Score")
    # # sns.set(font_scale=0.5)
    # annotator_i = Annotator(i, pairs_ke, data=all_sources_df, x="Source", y="Mean Score")
    # annotator_i.configure(test="Mann-Whitney", text_format="simple", show_test_name=False)
    # annotator_i.apply_and_annotate()
    # i.set(ylabel="Accuracy (95% ci)")
    # i.set(xlabel=None)
    # # i.axhline(20, color='r')
    # plt.xticks(size=6)
    # plt.title('Given Object classification accuracy Kinematic/EMG comparison')
    # plt.savefig('./results/Raw/plots/class_acc_KE_comp.png', dpi=600)
    # # plt.show()
    # plt.close()

    # ##############
    # # COMPARISON KT PLOTS
    # ###############
    # pairs_kt = [("Kin", "Multimodal_KT"), ("Kin", "Hierarchical_KT"), ("Tactile", "Multimodal_KT"),
    #             ("Tactile", "Hierarchical_KT"), ("Multimodal_KT", "Hierarchical_KT")]
    # all_sources_df = pd.concat([kin_df, tact_df, multi_KT_df, hierarchical_KT_df])
    # plt.figure()
    # i = sns.barplot(data=all_sources_df, x="Source", y="Mean Score")
    # # sns.set(font_scale=0.5)
    # annotator_i = Annotator(i, pairs_kt, data=all_sources_df, x="Source", y="Mean Score")
    # annotator_i.configure(test="Mann-Whitney", text_format="simple", show_test_name=False)
    # annotator_i.apply_and_annotate()
    # i.set(ylabel="Accuracy (95% ci)")
    # i.set(xlabel=None)
    # # i.axhline(20, color='r')
    # plt.xticks(size=6)
    # plt.title('Given Object classification accuracy Kinematic/Tactile comparison')
    # plt.savefig('./results/Raw/plots/class_acc_KT_comp.png', dpi=600)
    # # plt.show()
    # plt.close()

    # ##############
    # # COMPARISON ET PLOTS
    # ###############
    # pairs_et = [("EMG", "Multimodal_ET"), ("EMG", "Hierarchical_ET"), ("Tactile", "Multimodal_ET"),
    #             ("Tactile", "Hierarchical_ET"), ("Multimodal_ET", "Hierarchical_ET")]
    # all_sources_df = pd.concat([emg_df, tact_df, multi_ET_df, hierarchical_ET_df])
    # plt.figure()
    # i = sns.barplot(data=all_sources_df, x="Source", y="Mean Score")
    # # sns.set(font_scale=0.5)
    # annotator_i = Annotator(i, pairs_et, data=all_sources_df, x="Source", y="Mean Score")
    # annotator_i.configure(test="Mann-Whitney", text_format="simple", show_test_name=False)
    # annotator_i.apply_and_annotate()
    # i.set(ylabel="Accuracy (95% ci)")
    # i.set(xlabel=None)
    # # i.axhline(20, color='r')
    # plt.xticks(size=6)
    # plt.title('Given Object classification accuracy EMG?Tactile comparison')
    # plt.savefig('./results/Raw/plots/class_acc_ET_comp.png', dpi=600)
    # # plt.show()
    # plt.close()


if __name__ == "__main__":
    main()
