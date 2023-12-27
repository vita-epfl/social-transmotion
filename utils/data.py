import os
import numpy as np
from utils.utils import path_to_data
from utils.trajnetplusplustools import Reader_jta_all_visual_cues, Reader_jrdb_2dbox


def load_data_jrdb_2dbox(split):
    joint_and_mask=[]
    ############## change dataset path
    name = 'jrdb_2dbox'
    train_scenes, _, _ = prepare_data('data/jrdb_2dbox/', subset=split, sample=1.0, goals=False, dataset_name=name)
    for scene_i, (filename, scene_id, paths) in enumerate(train_scenes):
        scene_train = Reader_jrdb_2dbox.paths_to_xy(paths)
        scene_train = drop_ped_with_missing_frame(scene_train)
        scene_train, _ = drop_distant_far(scene_train)
        
        scene_train_real = scene_train.reshape(scene_train.shape[0],scene_train.shape[1],-1,4) ##(21, n, 16, 4) #jjjjjjjjjjjjjjjjjjjjjj 2j
        scene_train_real_ped = np.transpose(scene_train_real,(1,0,2,3)) ## (n, 21, 16, 3)

        scene_train_mask = np.ones(scene_train_real_ped.shape[:-1])
        joint_and_mask.append((np.asarray(scene_train_real_ped), np.asarray(scene_train_mask)))

    return joint_and_mask

def load_data_jta_all_visual_cues(split):
    joint_and_mask=[]
    ############## change dataset path
    name = 'jta_all_visual_cues'

    train_scenes, _, _ = prepare_data('data/jta_all_visual_cues/', subset=split, sample=1.0, goals=False, dataset_name=name)
    for scene_i, (filename, scene_id, paths) in enumerate(train_scenes):
        scene_train = Reader_jta_all_visual_cues.paths_to_xy(paths)
        scene_train = drop_ped_with_missing_frame(scene_train)
        scene_train, _ = drop_distant_far(scene_train)
        scene_train_real = scene_train.reshape(scene_train.shape[0],scene_train.shape[1],-1,4) 
        scene_train_real_ped = np.transpose(scene_train_real,(1,0,2,3)) 

        scene_train_mask = np.ones(scene_train_real_ped.shape[:-1])
        joint_and_mask.append((np.asarray(scene_train_real_ped), np.asarray(scene_train_mask)))

    return joint_and_mask


def prepare_data(path, subset='/train/', sample=1.0, goals=True, dataset_name=''):

    all_scenes = []

    ## List file names
    files = [f.split('.')[-2] for f in os.listdir(path + subset) if f.endswith('.ndjson')]
    ## Iterate over file names
    if dataset_name == 'jta_all_visual_cues':
        for file in files:
            reader = Reader_jta_all_visual_cues(path + subset + '/' + file + '.ndjson', scene_type='paths')
            scene = [(file, s_id, s) for s_id, s in reader.scenes(sample=sample)]
            all_scenes += scene
        return all_scenes, None, True
    elif dataset_name == 'jrdb_2dbox':
        for file in files:
            reader = Reader_jrdb_2dbox(path + subset + '/' + file + '.ndjson', scene_type='paths')
            scene = [(file, s_id, s) for s_id, s in reader.scenes(sample=sample)]
            all_scenes += scene
        return all_scenes, None, True
    else:
        print('not implement this dataset, error from utils/data.py')
        exit()

def drop_ped_with_missing_frame(xy):
    xy_n_t = np.transpose(xy, (1, 0, 2)) 
    mask = np.ones(xy_n_t.shape[0], dtype=bool)
    for n in range(xy_n_t.shape[0]-1):
        for t in range(9):
            if np.isnan(xy_n_t[n+1, t, 0]) == True:
                mask[n+1] = False
                break
    return np.transpose(xy_n_t[mask], (1, 0, 2))

def drop_distant_far(xy, r=6):
    distance_2 = np.sum(np.square(xy[:, :, 0:2] - xy[:, 0:1, 0:2]), axis=2)
    mask = np.nanmin(distance_2, axis=0) < r**2
    return xy[:, mask], mask

