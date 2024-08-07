import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import numpy as np
from itertools import combinations
from statannotations.Annotator import Annotator
from statannot import add_stat_annotation, statannot
from matplotlib.patches import Patch

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def ep_stats_classification_plots():

    plt.rcParams.update({'font.size':16})

    kinds = ["pres/abs", "count", "duration"]
    pairs = list(combinations(kinds, r=2))

    accuracies_df = pd.DataFrame()
    rand_accuracies_df = pd.DataFrame()

    # GIVEN PRES/ABS
    obj_presabs_acc = []
    obj_presabs_rand_acc = []
    obj_presabs = './results/EP/accuracy/ep_presabs_giv_results_file.csv'

    with open(obj_presabs, 'r') as obj_presabs_data:
        obj_presabs_reader = csv.reader(obj_presabs_data)
        for row in obj_presabs_reader:
            row_o_pa = row[0].replace("[", "")
            row_o_pa = row_o_pa.replace("]", "")
            if row[1] != 'Random':
                obj_presabs_acc = np.append(obj_presabs_acc, np.fromstring(row_o_pa, dtype=float, sep=","))
            else:
                obj_presabs_rand_acc = np.append(obj_presabs_rand_acc, np.fromstring(row_o_pa, dtype=float, sep=","))

    obj_label = ["given"] * len(obj_presabs_acc)
    kind_label = ["pres/abs"] * len(obj_presabs_acc)
    obj_presabs_acc_df = pd.DataFrame({"Accuracy": obj_presabs_acc, "Obj/Fam": obj_label, "Kind": kind_label})
    accuracies_df = pd.concat([accuracies_df, obj_presabs_acc_df], ignore_index=True)
    obj_presabs_rand_acc_df = pd.DataFrame({"Accuracy": obj_presabs_rand_acc, "Obj/Fam": obj_label, "Kind": kind_label})
    rand_accuracies_df = pd.concat([rand_accuracies_df, obj_presabs_rand_acc_df], ignore_index=True)

    # GIVEN COUNT
    obj_count_acc = []
    obj_count_rand_acc = []
    obj_count = './results/EP/accuracy/ep_count_giv_results_file.csv'

    with open(obj_count, 'r') as obj_count_data:
        obj_count_reader = csv.reader(obj_count_data)
        for row in obj_count_reader:
            row_o_c = row[0].replace("[", "")
            row_o_c = row_o_c.replace("]", "")
            if row[1] != 'Random':
                obj_count_acc = np.append(obj_count_acc, np.fromstring(row_o_c, dtype=float, sep=","))
            else:
                obj_count_rand_acc = np.append(obj_count_rand_acc, np.fromstring(row_o_c, dtype=float, sep=","))

    obj_label = ["given"] * len(obj_count_acc)
    kind_label = ["count"] * len(obj_count_acc)
    obj_count_acc_df = pd.DataFrame({"Accuracy": obj_count_acc, "Obj/Fam": obj_label, "Kind": kind_label})
    accuracies_df = pd.concat([accuracies_df, obj_count_acc_df], ignore_index=True)
    obj_count_rand_acc_df = pd.DataFrame({"Accuracy": obj_count_rand_acc, "Obj/Fam": obj_label, "Kind": kind_label})
    rand_accuracies_df = pd.concat([rand_accuracies_df, obj_count_rand_acc_df], ignore_index=True)

    # GIVEN DURATION
    obj_dur_acc = []
    obj_dur_rand_acc = []
    obj_dur = './results/EP/accuracy/ep_dur_giv_results_file.csv'

    with open(obj_dur, 'r') as obj_dur_data:
        obj_dur_reader = csv.reader(obj_dur_data)
        for row in obj_dur_reader:
            row_o_d = row[0].replace("[", "")
            row_o_d = row_o_d.replace("]", "")
            if row[1] != 'Random':
                obj_dur_acc = np.append(obj_count_acc, np.fromstring(row_o_d, dtype=float, sep=","))
            else:
                obj_dur_rand_acc = np.append(obj_count_rand_acc, np.fromstring(row_o_d, dtype=float, sep=","))

    obj_label = ["given"] * len(obj_dur_acc)
    kind_label = ["duration"] * len(obj_dur_acc)
    obj_dur_acc_df = pd.DataFrame({"Accuracy": obj_dur_acc, "Obj/Fam": obj_label, "Kind": kind_label})
    accuracies_df = pd.concat([accuracies_df, obj_dur_acc_df], ignore_index=True)
    obj_dur_rand_acc_df = pd.DataFrame({"Accuracy": obj_dur_rand_acc, "Obj/Fam": obj_label, "Kind": kind_label})
    rand_accuracies_df = pd.concat([rand_accuracies_df, obj_dur_rand_acc_df], ignore_index=True)

    # BOXPLOT EP GIVEN
    plt.figure()
    g = sns.barplot(data=accuracies_df.loc[accuracies_df["Obj/Fam"] == 'given'], x="Kind", y="Accuracy")
    g.set(xlabel='')
    annotator_g = Annotator(g, pairs, data=accuracies_df.loc[accuracies_df["Obj/Fam"] == 'given'], x="Kind",
                          y="Accuracy")
    annotator_g.configure(test="Mann-Whitney", text_format="simple", show_test_name=False)
    annotator_g.apply_and_annotate()
    g.set(ylabel="Accuracy (95% ci)")
    g.axhline(33.33, color='r')
    # plt.title('Classification Accuracy for Given Object Classification')
    # plt.autoscale()
    # plt.tight_layout()
    plt.savefig('./results/EP/plots/EP_giv_class_acc.svg', format='svg', dpi=600)
    # # plt.show()
    # plt.close()

    # ASKED PRES/ABS
    obj_presabs_acc = []
    obj_presabs_rand_acc = []
    obj_presabs = './results/EP/accuracy/ep_presabs_ask_results_file.csv'

    with open(obj_presabs, 'r') as obj_presabs_data:
        obj_presabs_reader = csv.reader(obj_presabs_data)
        for row in obj_presabs_reader:
            row_o_pa = row[0].replace("[", "")
            row_o_pa = row_o_pa.replace("]", "")
            if row[1] != 'Random':
                obj_presabs_acc = np.append(obj_presabs_acc, np.fromstring(row_o_pa, dtype=float, sep=","))
            else:
                obj_presabs_rand_acc = np.append(obj_presabs_rand_acc, np.fromstring(row_o_pa, dtype=float, sep=","))

    obj_label = ["asked"] * len(obj_presabs_acc)
    kind_label = ["pres/abs"] * len(obj_presabs_acc)
    obj_presabs_acc_df = pd.DataFrame({"Accuracy": obj_presabs_acc, "Obj/Fam": obj_label, "Kind": kind_label})
    accuracies_df = pd.concat([accuracies_df, obj_presabs_acc_df], ignore_index=True)
    obj_presabs_rand_acc_df = pd.DataFrame({"Accuracy": obj_presabs_rand_acc, "Obj/Fam": obj_label, "Kind": kind_label})
    rand_accuracies_df = pd.concat([rand_accuracies_df, obj_presabs_rand_acc_df], ignore_index=True)

    # ASKED COUNT
    obj_count_acc = []
    obj_count_rand_acc = []
    obj_count = './results/EP/accuracy/ep_count_ask_results_file.csv'

    with open(obj_count, 'r') as obj_count_data:
        obj_count_reader = csv.reader(obj_count_data)
        for row in obj_count_reader:
            row_o_c = row[0].replace("[", "")
            row_o_c = row_o_c.replace("]", "")
            if row[1] != 'Random':
                obj_count_acc = np.append(obj_count_acc, np.fromstring(row_o_c, dtype=float, sep=","))
            else:
                obj_count_rand_acc = np.append(obj_count_rand_acc, np.fromstring(row_o_c, dtype=float, sep=","))

    obj_label = ["asked"] * len(obj_count_acc)
    kind_label = ["count"] * len(obj_count_acc)
    obj_count_acc_df = pd.DataFrame({"Accuracy": obj_count_acc, "Obj/Fam": obj_label, "Kind": kind_label})
    accuracies_df = pd.concat([accuracies_df, obj_count_acc_df], ignore_index=True)
    obj_count_rand_acc_df = pd.DataFrame({"Accuracy": obj_count_rand_acc, "Obj/Fam": obj_label, "Kind": kind_label})
    rand_accuracies_df = pd.concat([rand_accuracies_df, obj_count_rand_acc_df], ignore_index=True)

    # ASKED DURATION
    obj_dur_acc = []
    obj_dur_rand_acc = []
    obj_dur = './results/EP/accuracy/ep_dur_ask_results_file.csv'

    with open(obj_dur, 'r') as obj_dur_data:
        obj_dur_reader = csv.reader(obj_dur_data)
        for row in obj_dur_reader:
            row_o_d = row[0].replace("[", "")
            row_o_d = row_o_d.replace("]", "")
            if row[1] != 'Random':
                obj_dur_acc = np.append(obj_dur_acc, np.fromstring(row_o_d, dtype=float, sep=","))
            else:
                obj_dur_rand_acc = np.append(obj_dur_rand_acc, np.fromstring(row_o_d, dtype=float, sep=","))

    obj_label = ["asked"] * len(obj_dur_acc)
    kind_label = ["duration"] * len(obj_dur_acc)
    obj_dur_acc_df = pd.DataFrame({"Accuracy": obj_dur_acc, "Obj/Fam": obj_label, "Kind": kind_label})
    accuracies_df = pd.concat([accuracies_df, obj_dur_acc_df], ignore_index=True)
    obj_dur_rand_acc_df = pd.DataFrame({"Accuracy": obj_dur_rand_acc, "Obj/Fam": obj_label, "Kind": kind_label})
    rand_accuracies_df = pd.concat([rand_accuracies_df, obj_dur_rand_acc_df], ignore_index=True)

    # BOXPLOT EP ASKED
    # plt.figure()
    # h = sns.barplot(data=accuracies_df.loc[accuracies_df["Obj/Fam"] == 'asked'], x="Kind", y="Accuracy")
    # annotator_h = Annotator(h, pairs, data=accuracies_df.loc[accuracies_df["Obj/Fam"] == 'asked'], x="Kind", y="Accuracy")
    # annotator_h.configure(test="Mann-Whitney", text_format="simple", show_test_name=False)
    # annotator_h.apply_and_annotate()
    # h.set(ylabel="Accuracy (95% ci)")
    # h.axhline(33.33, color='r')
    # plt.title('Classification Accuracy for Asked Object Classification')
    # plt.savefig('./results/EP/plots/EP_ask_class_acc.png', dpi=600)
    # # plt.show()
    # plt.close()

    asked_acc = accuracies_df.loc[accuracies_df["Obj/Fam"] == 'asked']
    # ask_label = ["real"] * len(asked_acc)
    # asked_acc['real/rnd'] = ask_label
    rand_asked_acc = rand_accuracies_df.loc[rand_accuracies_df["Obj/Fam"] == 'asked']
    rand_label = ["random"] * len(rand_asked_acc)
    rand_asked_acc["Obj/Fam"] = rand_label
    rand_asked_acc["Kind"] = rand_label
    asked = pd.concat([asked_acc, rand_asked_acc])

    new_pairs = [(x,'random') for x in kinds]

    plt.figure()
    h = sns.barplot(data=asked, x="Kind", y="Accuracy")
    h.set(xlabel='')
    annotator_h = Annotator(h, new_pairs, data=asked, x="Kind", y="Accuracy")
    annotator_h.configure(test="Mann-Whitney", text_format="simple", show_test_name=False)
    annotator_h.apply_and_annotate()
    h.set(ylabel="Accuracy (95% ci)")
    h.axhline(33.33, color='r')
    # plt.title('Classification Accuracy for Asked Object Classification')
    # plt.savefig('./results/EP/plots/EP_ask_class_acc_rand_comp.png', dpi=600)
    # plt.autoscale()
    # plt.tight_layout()
    plt.savefig('./results/EP/plots/EP_ask_class_acc_rand_comp.svg', format='svg', dpi=600)
    # plt.show()
    # plt.close()

    # FAMILY PRES/ABS
    fam_presabs_acc = []
    fam_presabs_rand_acc = []
    fam_presabs = './results/EP/accuracy/ep_presabs_fam_results_file.csv'

    with open(fam_presabs, 'r') as fam_presabs_data:
        fam_presabs_reader = csv.reader(fam_presabs_data)
        for row in fam_presabs_reader:
            row_f_pa = row[0].replace("[", "")
            row_f_pa = row_f_pa.replace("]", "")
            if row[1] != 'Random':
                fam_presabs_acc = np.append(fam_presabs_acc, np.fromstring(row_f_pa, dtype=float, sep=","))
            else:
                fam_presabs_rand_acc = np.append(fam_presabs_rand_acc, np.fromstring(row_f_pa, dtype=float, sep=","))

    fam_label = ["family"] * len(fam_presabs_acc)
    kind_label = ["pres/abs"] * len(fam_presabs_acc)
    fam_presabs_acc_df = pd.DataFrame({"Accuracy": fam_presabs_acc, "Obj/Fam": fam_label, "Kind": kind_label})
    accuracies_df = pd.concat([accuracies_df, fam_presabs_acc_df], ignore_index=True)
    fam_presabs_rand_acc_df = pd.DataFrame({"Accuracy": fam_presabs_rand_acc, "Obj/Fam": fam_label, "Kind": kind_label})
    rand_accuracies_df = pd.concat([rand_accuracies_df, fam_presabs_rand_acc_df], ignore_index=True)

    # FAMILY COUNT
    fam_count_acc = []
    fam_count_rand_acc = []
    fam_count = './results/EP/accuracy/ep_count_fam_results_file.csv'

    with open(fam_count, 'r') as fam_count_data:
        fam_count_reader = csv.reader(fam_count_data)
        for row in fam_count_reader:
            row_f_c = row[0].replace("[", "")
            row_f_c = row_f_c.replace("]", "")
            if row[1] != 'Random':
                fam_count_acc = np.append(fam_count_acc, np.fromstring(row_f_c, dtype=float, sep=","))
            else:
                fam_count_rand_acc = np.append(fam_count_rand_acc, np.fromstring(row_f_c, dtype=float, sep=","))

    fam_label = ["family"] * len(fam_count_acc)
    kind_label = ["count"] * len(fam_count_acc)
    fam_count_acc_df = pd.DataFrame({"Accuracy": fam_count_acc, "Obj/Fam": fam_label, "Kind": kind_label})
    accuracies_df = pd.concat([accuracies_df, fam_count_acc_df], ignore_index=True)
    fam_count_rand_acc_df = pd.DataFrame({"Accuracy": fam_count_rand_acc, "Obj/Fam": fam_label, "Kind": kind_label})
    rand_accuracies_df = pd.concat([rand_accuracies_df, fam_count_rand_acc_df], ignore_index=True)

    # FAMILY DURATION
    fam_dur_acc = []
    fam_dur_rand_acc = []
    fam_dur = './results/EP/accuracy/ep_dur_fam_results_file.csv'
    with open(fam_dur, 'r') as fam_dur_data:
        fam_dur_reader = csv.reader(fam_dur_data)
        for row in fam_dur_reader:
            row_f_d = row[0].replace("[", "")
            row_f_d = row_f_d.replace("]", "")
            if row[1] != 'Random':
                fam_dur_acc = np.append(fam_dur_acc, np.fromstring(row_f_d, dtype=float, sep=","))
            else:
                fam_dur_rand_acc = np.append(fam_dur_rand_acc, np.fromstring(row_f_d, dtype=float, sep=","))

    fam_label = ["family"] * len(fam_dur_acc)
    kind_label = ["duration"] * len(fam_dur_acc)
    fam_dur_acc_df = pd.DataFrame({"Accuracy": fam_dur_acc, "Obj/Fam": fam_label, "Kind": kind_label})
    accuracies_df = pd.concat([accuracies_df, fam_dur_acc_df], ignore_index=True)
    fam_dur_rand_acc_df = pd.DataFrame({"Accuracy": fam_dur_rand_acc, "Obj/Fam": fam_label, "Kind": kind_label})
    rand_accuracies_df = pd.concat([rand_accuracies_df, fam_dur_rand_acc_df], ignore_index=True)

    # BOXPLOT EP FAMILY
    plt.figure()
    i = sns.barplot(data=accuracies_df.loc[accuracies_df["Obj/Fam"] == 'family'], x="Kind", y="Accuracy")
    i.set(xlabel='')
    annotator_i = Annotator(i, pairs, data=accuracies_df.loc[accuracies_df["Obj/Fam"] == 'family'], x="Kind", y="Accuracy")
    annotator_i.configure(test="Mann-Whitney", text_format="simple", show_test_name=False)
    annotator_i.apply_and_annotate()
    i.set(ylabel="Accuracy (95% ci)")
    i.axhline(20, color='r')
    # plt.title('Classification Accuracy for Family Classification')
    # plt.autoscale()
    # plt.tight_layout()
    plt.savefig('./results/EP/plots/EP_fam_class_acc.svg', format='svg', dpi=600)
    # # plt.show()
    # plt.close()

    # BOXPLOT ALL COMPARISON
    plt.figure()
    # sns.set_palette("colorblind")
    hatches = ['...', '', '///']
    # Create the barplot
    j = sns.barplot(data=accuracies_df, x="Kind", y="Accuracy", hue="Obj/Fam", hue_order=["asked", "given", "family"], palette=["white"]*3)
    j.set(xlabel='')
    # Each bar represented by j.patches is ordered by the 'Kind' and within each 'Kind' by 'Obj/Fam'
    hue_order = ["asked", "given", "family"]
    num_hues = len(hue_order)
    num_kinds = len(accuracies_df['Kind'].unique())
    bars_per_group = num_hues * num_kinds
    for i, bar in enumerate(j.patches):
        hue_index = (i // num_kinds) % num_hues  # Calculate which hue within the kind
        hatch = hatches[hue_index]
        bar.set_hatch(hatch)
        bar.set_edgecolor('black')
    # Add statistical annotations
    statannot.add_stat_annotation(j, data=accuracies_df, x="Kind", y="Accuracy", hue="Obj/Fam",
                                  hue_order=["asked", "given", "family"],
                                  box_pairs=[(("pres/abs", "asked"), ("pres/abs", "given")),
                                             (("pres/abs", "asked"), ("pres/abs", "family")),
                                             (("pres/abs", "given"), ("pres/abs", "family")),
                                             (("count", "asked"), ("count", "given")),
                                             (("count", "asked"), ("count", "family")),
                                             (("count", "given"), ("count", "family")),
                                             (("duration", "asked"), ("duration", "given")),
                                             (("duration", "asked"), ("duration", "family")),
                                             (("duration", "given"), ("duration", "family"))],
                                  test="Mann-Whitney",
                                  text_format="star",
                                  show_test_name=False,
                                  color='black'
                                  )
    # Set y-axis label
    j.set(ylabel="Accuracy (95% ci)")
    # Set plot title
    plt.title('Classification Accuracy Comparison')
    # Adjust the legend to ensure clarity in black and white
    handles, labels = j.get_legend_handles_labels()
    plt.legend(handles, labels, loc="lower right")
    # Save the plot
    # plt.autoscale()
    # plt.tight_layout()
    plt.savefig('./results/EP/plots/EP_all_comp_acc_star.svg', format='svg', dpi=600)
    plt.close()

    # BOXPLOT COMPARISON ASKED-GIVEN
    # plt.figure()
    # k = sns.barplot(data=accuracies_df.loc[accuracies_df["Obj/Fam"] != "family"], x="Kind", y="Accuracy", hue="Obj/Fam",
    #                 hue_order=list(["asked", "given"]))
    # statannot.add_stat_annotation(k, data=accuracies_df.loc[accuracies_df["Obj/Fam"] != "family"], x="Kind", y="Accuracy", hue="Obj/Fam",
    #                               box_pairs=[(("pres/abs", "asked"), ("pres/abs", "given")),
    #                                          (("count", "asked"), ("count", "given")),
    #                                          (("duration", "asked"), ("duration", "given"))],
    #                               test="Mann-Whitney",
    #                               text_format="simple",
    #                               show_test_name=False,
    #                               )
    # k.set(ylabel="Accuracy (95% ci)")
    # plt.title('Classification Accuracy Comparison')
    # sns.move_legend(k, "lower right")
    # plt.savefig('./results/EP/plots/EP_ask_giv_comp_acc.png', dpi=600)
    # # plt.show()
    # plt.close()

    # BOXPLOT COMPARISON ASKED-FAMILY
    # plt.figure()
    # l = sns.barplot(data=accuracies_df.loc[accuracies_df["Obj/Fam"] != "given"], x="Kind", y="Accuracy", hue="Obj/Fam",
    #                 hue_order=list(["asked", "family"]))
    # statannot.add_stat_annotation(l, data=accuracies_df.loc[accuracies_df["Obj/Fam"] != "given"], x="Kind",
    #                               y="Accuracy", hue="Obj/Fam",
    #                               box_pairs=[(("pres/abs", "asked"), ("pres/abs", "family")),
    #                                          (("count", "asked"), ("count", "family")),
    #                                          (("duration", "asked"), ("duration", "family"))],
    #                               test="Mann-Whitney",
    #                               text_format="simple",
    #                               show_test_name=False,
    #                               )
    # l.set(ylabel="Accuracy (95% ci)")
    # plt.title('Classification Accuracy Comparison')
    # sns.move_legend(l, "lower right")
    # plt.savefig('./results/EP/plots/EP_ask_fam_comp_acc.png', dpi=600)
    # # plt.show()
    # plt.close()

    # BOXPLOT COMPARISON GIVEN-FAMILY
    # plt.figure()
    # m = sns.barplot(data=accuracies_df.loc[accuracies_df["Obj/Fam"] != "asked"], x="Kind", y="Accuracy", hue="Obj/Fam",
    #                 hue_order=list(["given", "family"]))
    # statannot.add_stat_annotation(m, data=accuracies_df.loc[accuracies_df["Obj/Fam"] != "asked"], x="Kind",
    #                               y="Accuracy", hue="Obj/Fam",
    #                               box_pairs=[(("pres/abs", "given"), ("pres/abs", "family")),
    #                                          (("count", "given"), ("count", "family")),
    #                                          (("duration", "given"), ("duration", "family"))],
    #                               test="Mann-Whitney",
    #                               text_format="simple",
    #                               show_test_name=False,
    #                               )
    # m.set(ylabel="Accuracy (95% ci)")
    # plt.title('Classification Accuracy Comparison')
    # sns.move_legend(m, "lower right")
    # plt.savefig('./results/EP/plots/EP_giv_fam_comp_acc.png', dpi=600)
    # # plt.show()
    # plt.close()

    # Filter the DataFrame for only presence/absence results
    pres_abs_df = accuracies_df[accuracies_df["Kind"] == "pres/abs"]

    # BOXPLOT PRES/ABS COMPARISON
    # Start the figure
    plt.figure()
    # Define hatches for "asked," "given," and "family"
    hatches = ['...', '', '///']
    # Create the barplot
    j = sns.barplot(data=pres_abs_df, x="Kind", y="Accuracy", hue="Obj/Fam", hue_order=["asked", "given", "family"],
                    palette=["white"] * 3)
    j.set(xlabel='')
    # Apply hatches to the bars
    num_kinds = len(pres_abs_df["Kind"].unique())
    num_hues = len(hatches)
    bars_per_group = len(j.patches) // num_kinds
    for i, bar in enumerate(j.patches):
        # Calculate which hue category within the Kind we are in
        hue_index = (i // num_kinds) % num_hues
        bar.set_hatch(hatches[hue_index])
        bar.set_edgecolor('black')
    # Add statistical annotations
    statannot.add_stat_annotation(j, data=pres_abs_df, x="Kind", y="Accuracy", hue="Obj/Fam",
                                  hue_order=["asked", "given", "family"],
                                  box_pairs=[(("pres/abs", "asked"), ("pres/abs", "given")),
                                             (("pres/abs", "asked"), ("pres/abs", "family")),
                                             (("pres/abs", "given"), ("pres/abs", "family"))],
                                  test="Mann-Whitney",
                                  text_format="star",
                                  show_test_name=False,
                                  color='black')
    # Set y-axis label
    j.set(ylabel="Accuracy (95% ci)")
    # Set plot title
    plt.title('Presence/Absence Classification Accuracy Comparison')
    # Define legend handles using the defined hatches and corresponding colors
    legend_handles = [Patch(facecolor='white', edgecolor='black', label='asked', hatch=hatches[0]),
                      Patch(facecolor='white', edgecolor='black', label='given', hatch=hatches[1]),
                      Patch(facecolor='white', edgecolor='black', label='family', hatch=hatches[2])]

    # Set the legend with these custom handles
    plt.legend(handles=legend_handles, loc="lower right")
    # Move legend to the appropriate location
    sns.move_legend(j, "lower right")
    # plt.autoscale()
    # plt.tight_layout()
    # Save the plot
    plt.savefig('./results/EP/plots/EP_pres_abs_comp_acc_star.svg', format='svg', dpi=600)
    plt.close()