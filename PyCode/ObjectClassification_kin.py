import json
import numpy as np
from sklearn.model_selection import StratifiedKFold

def main():

    # VARIABLES TO ADJUST
    family_to_select = 'Cutlery'
    num_bins = 5
    cv = 5

    # READING FILES AND CREATING DATASET
    file_data = "../Data/filtered_data.json"
    file_eps = "../Data/ep_labels.json"
    file_task = "../Data/task_labels.json"

    with open(file_data, "r") as f:
        data = json.load(f)  # data[subjects][trials][joints]

    with open(file_eps, "r") as g:
        eps = json.load(g)  # eps[subjects][trials]

    with open(file_task, "r") as h:
        task = json.load(h)  # task[subjects][trials]

    # VECTORIZING DATA

    vectorized_task = [x for sublist in task for x in sublist]  # Vectorization of tasks
    vectorized_eps = [x for sublist in eps for x in sublist]  # Vectorization of eps
    vectorized_data = [x for sublist in data for x in sublist]  # Vectorization of trials
    vectorized_data = np.array(vectorized_data, dtype=float) # Conversion to float to we replace 'None' with 'NaN'
    given_object = [x.split("_")[0] for x in vectorized_task]  # Vectorized given objects
    ask_object = [x.split("_")[1] for x in vectorized_task]  # Vectorized asked objects

    # SPLIT BY TRIALS

    tr_idx = [index for index, _ in enumerate(vectorized_task) if vectorized_task[index] != vectorized_task[index-1]]
    tr_idx.append(len(vectorized_task))

    # all these lists are [trials]x[timepoints per trial]
    spl_task = [vectorized_task[tr_idx[i]:tr_idx[i+1]] for i, _ in enumerate(tr_idx) if i != len(tr_idx)-1]
    spl_eps = [vectorized_eps[tr_idx[i]:tr_idx[i + 1]] for i, _ in enumerate(tr_idx) if i != len(tr_idx) - 1]
    spl_dat = [vectorized_data[tr_idx[i]:tr_idx[i + 1]] for i, _ in enumerate(tr_idx) if i != len(tr_idx) - 1]
    spl_given = [given_object[tr_idx[i]:tr_idx[i + 1]] for i, _ in enumerate(tr_idx) if i != len(tr_idx) - 1]
    spl_ask = [ask_object[tr_idx[i]:tr_idx[i + 1]] for i, _ in enumerate(tr_idx) if i != len(tr_idx) - 1]

    # SELECT TRIALS BY FAMILY

    obj_fam = dict(
        CeramicMug = 'Mugs',
        Glass = 'Mugs',
        MetalMug = 'Mugs',
        CeramicPlate = 'Plates',
        MetalPlate = 'Plates',
        PlasticPlate = 'Plates',
        Cube = 'Geometric',
        Cylinder ='Geometric',
        Triangle ='Geometric',
        Fork = 'Cutlery',
        Knife ='Cutlery',
        Spoon ='Cutlery',
        PingPongBall = 'Ball',
        SquashBall='Ball',
        TennisBall='Ball'
    )

    fam_idx = list()
    for it in range(len(spl_given)):
        if obj_fam[spl_given[it][0]] == family_to_select:
            fam_idx.append(it)

    selected_task = [spl_task[idx] for idx in fam_idx]
    selected_eps = [spl_eps[idx] for idx in fam_idx]
    selected_dat = [spl_dat[idx] for idx in fam_idx]
    selected_given = [spl_given[idx] for idx in fam_idx]
    selected_ask = [spl_ask[idx] for idx in fam_idx]

    # DIVIDE BY EPs

    # all these list are [trials]x[eps per trial]x[timepoints per ep]
    final_task = list()
    final_eps = list()
    final_given = list()
    final_ask = list()
    # this list is [trials]x[eps per trial]x[bins per ep]x[joints]
    final_dat = list()

    for e in range(len(selected_eps)):

        ch_idx = [index for index, _ in enumerate(selected_eps[e]) if selected_eps[e][index] != selected_eps[e][index-1]]
        ch_idx.append(len(selected_eps[e]))
        if 0 not in ch_idx:
            ch_idx.insert(0, 0)

        sel_task = [selected_task[e][ch_idx[i]:ch_idx[i + 1]] for i, _ in enumerate(ch_idx) if i != len(ch_idx) - 1]
        final_task.append(sel_task)
        sel_eps = [selected_eps[e][ch_idx[i]:ch_idx[i + 1]] for i, _ in enumerate(ch_idx) if i != len(ch_idx) - 1]
        final_eps.append(sel_eps)
        sel_given = [selected_given[e][ch_idx[i]:ch_idx[i + 1]] for i, _ in enumerate(ch_idx) if i != len(ch_idx) - 1]
        final_given.append(sel_given)
        sel_ask = [selected_ask[e][ch_idx[i]:ch_idx[i + 1]] for i, _ in enumerate(ch_idx) if i != len(ch_idx) - 1]
        final_ask.append(sel_ask)

        sel_dat = [selected_dat[e][ch_idx[i]:ch_idx[i + 1]] for i, _ in enumerate(ch_idx) if i != len(ch_idx) - 1]

        aux = list()
        for j in range(len(sel_dat)):
            div_dat = np.array_split(sel_dat[j], 5)
            me = [np.nanmean(x, axis=0) for x in div_dat]
            aux.append(me)

        final_dat.append(aux)

    # SELECT TRIALS CV

    skf = StratifiedKFold(n_splits=cv)

    a=1











if __name__ == "__main__":
        main()