import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np

def split(data):

    tasks = np.array(data['Task'])
    eps = np.array(data['EP'])

    task_num = 0
    ep_num = 0
    trial_list = []
    ep_list = []

    trial_list.extend([task_num])
    ep_list.extend([ep_num])

    for iter in range(1, len(tasks)):

        if tasks[iter] != tasks[iter - 1]:
            task_num += 1
            ep_num = 0
        elif eps[iter] != eps[iter - 1]:
            ep_num += 1

        trial_list.extend([task_num])
        ep_list.extend([ep_num])

    data['Trial num'] = trial_list
    data['EP num'] = ep_list

    given = [item.split("_")[0] for item in tasks]
    data['Given Object'] = given

    obj_fam = dict( CeramicMug = 'Mugs',
                    Glass = 'Mugs',
                    MetalMug = 'Mugs',
                    CeramicPlate='Plates',
                    MetalPlate = 'Plates',
                    PlasticPlate = 'Plates',
                    Cube='Geometric',
                    Cylinder = 'Geometric',
                    Triangle = 'Geometric',
                    Fork = 'Cutlery',
                    Knife = 'Cutlery',
                    Spoon = 'Cutlery',
                    PingPongBall = 'Ball',
                    SquashBall = 'Ball',
                    TennisBall = 'Ball',
                   )

    family = [obj_fam[x] for x in given]
    data['Family'] = family

    return data
