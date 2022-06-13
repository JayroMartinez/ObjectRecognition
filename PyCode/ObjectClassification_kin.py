import json
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import glmnet_python
from glmnet_python import glmnet
from glmnet_python import glmnetPredict
from glmnet_python import glmnetPlot
from glmnet_python import cvglmnet
from glmnet_python import cvglmnetPredict
from glmnet_python import cvglmnetPlot
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.neural_network import MLPClassifier



def main():

    # VARIABLES TO ADJUST
    # 'Mugs', 'Plates', 'Geometric', 'Cutlery', 'Ball'
    family_to_select = 'Ball'
    num_bins = 5
    cv = 3

    # READING FILES AND CREATING DATASET
    file_data = "../Data/filtered_data.json"
    file_eps = "../Data/ep_labels.json"
    file_task = "../Data/task_labels.json"
    file_allSyn = "../Data/all_subjects_scores.json"

    with open(file_data, "r") as f:
        data = json.load(f)  # data[subjects][trials][joints]

    with open(file_eps, "r") as g:
        eps = json.load(g)  # eps[subjects][trials]

    with open(file_task, "r") as h:
        task = json.load(h)  # task[subjects][trials]

    with open(file_allSyn, "r") as h:
        all_syn = json.load(h)  # all_syn[subjects X trials][joints]


    # VECTORIZING DATA

    vectorized_task = [x for sublist in task for x in sublist]  # Vectorization of tasks
    vectorized_eps = [x for sublist in eps for x in sublist]  # Vectorization of eps
    vectorized_data = [x for sublist in data for x in sublist]  # Vectorization of trials
    vectorized_data = np.array(vectorized_data, dtype=float) # Conversion to float to we replace 'None' with 'NaN'
    given_object = [x.split("_")[0] for x in vectorized_task]  # Vectorized given objects
    ask_object = [x.split("_")[1] for x in vectorized_task]  # Vectorized asked objects
    all_syn = np.array(all_syn, dtype=float) # Conversion to float to we replace 'None' with 'NaN'

    # SPLIT BY TRIALS

    tr_idx = [index for index, _ in enumerate(vectorized_task) if vectorized_task[index] != vectorized_task[index-1]]
    tr_idx.append(len(vectorized_task))

    # all these lists are [trials]x[timepoints per trial]
    spl_task = [vectorized_task[tr_idx[i]:tr_idx[i+1]] for i, _ in enumerate(tr_idx) if i != len(tr_idx)-1]
    spl_eps = [vectorized_eps[tr_idx[i]:tr_idx[i + 1]] for i, _ in enumerate(tr_idx) if i != len(tr_idx) - 1]
    spl_dat = [vectorized_data[tr_idx[i]:tr_idx[i + 1]] for i, _ in enumerate(tr_idx) if i != len(tr_idx) - 1]
    spl_given = [given_object[tr_idx[i]:tr_idx[i + 1]] for i, _ in enumerate(tr_idx) if i != len(tr_idx) - 1]
    spl_ask = [ask_object[tr_idx[i]:tr_idx[i + 1]] for i, _ in enumerate(tr_idx) if i != len(tr_idx) - 1]
    spl_syn = [all_syn[tr_idx[i]:tr_idx[i + 1]] for i, _ in enumerate(tr_idx) if i != len(tr_idx) - 1]

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
    selected_syn = [spl_syn[idx] for idx in fam_idx]

    # DIVIDE BY EPs

    # all these list are [trials]x[eps per trial]x[timepoints per ep]
    final_task = list()
    final_eps = list()
    final_given = list()
    final_ask = list()
    # this lists are [trials]x[eps per trial]x[bins per ep]x[joints]
    final_dat = list()
    final_syn = list()

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
        sel_syn = [selected_syn[e][ch_idx[i]:ch_idx[i + 1]] for i, _ in enumerate(ch_idx) if i != len(ch_idx) - 1]

        aux = list()
        aux_syn = list()
        for j in range(len(sel_dat)):
            div_dat = np.array_split(sel_dat[j], num_bins)
            div_syn = np.array_split(sel_syn[j], num_bins)
            # print(len(sel_dat[j]), len(div_dat[0]))
            me = [np.nanmean(x, axis=0) for x in div_dat]
            me2 = [np.nanmean(x, axis=0) for x in div_syn]
            aux.append(me)
            aux_syn.append(me2)

        final_dat.append(aux)
        final_syn.append(aux_syn)

    # SELECT TRIALS CV

    total_acc = list()
    unique_given = [final_given[i][0][0] for i in range(len(final_given))]  # unique object per trial
    skf = StratifiedKFold(n_splits=cv)
    for train, test in skf.split(final_dat, unique_given):

        train_given = [unique_given[x] for x in train]
        test_given = [unique_given[y] for y in test]
        train_data = [final_dat[x] for x in train]
        test_data = [final_dat[y] for y in test]
        train_syn = [final_syn[x] for x in train]
        test_syn = [final_syn[y] for y in test]

        trn_dat = list()
        trn_lab = list()
        trn_syn = list()

        for i in range(len(train_data)):
            for j in range(len(train_data[i])):
                aux = list()
                aux_syn = list()
                for k in range(len(train_data[i][j])):
                    aux = np.append(aux, train_data[i][j][k])
                    aux_syn = np.append(aux_syn, train_syn[i][j][k])
                if np.count_nonzero(np.isnan(aux)) == 0:
                    trn_dat.append(aux)
                    trn_lab.append(train_given[i])
                    trn_syn.append(aux_syn)

        tst_dat = list()
        tst_lab = list()
        tst_syn = list()
        for i2 in range(len(test_data)):
            for j2 in range(len(test_data[i2])):
                aux2 = list()
                aux2_syn = list()
                for k2 in range(len(test_data[i2][j2])):
                    aux2 = np.append(aux2, test_data[i2][j2][k2])
                    aux2_syn = np.append(aux2_syn, test_syn[i2][j2][k2])
                if np.count_nonzero(np.isnan(aux2)) == 0:
                    tst_dat.append(aux2)
                    tst_lab.append(test_given[i2])
                    tst_syn.append(aux2_syn)

        # USING LOGISTIC REGRESSION
        log_model = LogisticRegression(penalty='elasticnet', C=0.2, solver='saga', max_iter=25000, multi_class='multinomial', l1_ratio=0.5)
        log_model.fit(trn_syn, trn_lab)
        pred = log_model.predict_proba(tst_syn)

        cl = log_model.classes_
        hits = 0
        for i in range(len(pred)):
            if cl[np.argmax(pred[i])] == tst_lab[i]:
                hits += 1
        acc = round((hits/len(tst_lab))*100, 2)
        total_acc.append(acc)
        print("Hits: ", hits, " out of ", len(tst_lab), ". ", acc, "%")

        #  USING GLMNET
        # lbl = LabelEncoder()
        # dum = np.float64(lbl.fit_transform(trn_lab))
        #
        # cvfit = cvglmnet(x=np.array(trn_dat), y=dum, family='multinomial', alpha=0.5)
        # pred = cvglmnetPredict(cvfit, np.array(tst_dat), ptype='class')
        #
        # dum_res = np.float64(lbl.fit_transform(tst_lab))
        # hits = 0
        # for i in range(len(pred)):
        #     if pred[i] == dum_res[i]:
        #         hits += 1
        # acc = round((hits/len(tst_lab))*100, 2)
        # total_acc.append(acc)
        # print("Hits: ", hits, " out of ", len(tst_lab), ". ", acc, "%")

        # USING SVM
        # clf = svm.SVC(C=25, kernel='poly', degree=3, decision_function_shape='ovo')  # 45.91
        # clf.fit(np.array(trn_dat), trn_lab)
        # pred = clf.predict(np.array(tst_dat))
        # hits = 0
        # for i in range(len(pred)):
        #     if pred[i] == tst_lab[i]:
        #         hits += 1
        # acc = round((hits/len(tst_lab))*100, 2)
        # total_acc.append(acc)
        # print("Hits: ", hits, " out of ", len(tst_lab), ". ", acc, "%")

        # USING ANN
        # num_hidden_layers = 4
        # layers = np.ceil(np.geomspace(len(trn_dat[0]), 3, num_hidden_layers+2)).astype(int)
        # clf = MLPClassifier(solver='adam', alpha=20, hidden_layer_sizes=(layers[1:len(layers)-1]), random_state=1, max_iter=100000, activation='tanh')
        # clf.fit(np.array(trn_dat), trn_lab)
        # pred = clf.predict(np.array(tst_dat))
        # hits = 0
        # for i in range(len(pred)):
        #     if pred[i] == tst_lab[i]:
        #         hits += 1
        # acc = round((hits/len(tst_lab))*100, 2)
        # total_acc.append(acc)
        # print("Hits: ", hits, " out of ", len(tst_lab), ". ", acc, "%")

    print("|| FAMILY: ", family_to_select, " || Mean accuracy after ", cv, " folds with ", num_bins, " bins per EP: ", round(np.mean(total_acc), 2), " % ||")










if __name__ == "__main__":
        main()