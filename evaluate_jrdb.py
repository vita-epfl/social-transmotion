import argparse
import torch
import random
import numpy as np
from progress.bar import Bar
from torch.utils.data import DataLoader

from dataset_jrdb import batch_process_coords, create_dataset, collate_batch
from model_jrdb import create_model
from utils.utils import create_logger

def inference(model, config, input_joints, padding_mask, out_len=14):
    model.eval()
    
    with torch.no_grad():
        pred_joints = model(input_joints, padding_mask)

    output_joints = pred_joints[:,-out_len:]

    return output_joints


def evaluate_ade_fde(model, modality_selection, dataloader, bs, config, logger, return_all=False, bar_prefix="", per_joint=False, show_avg=False):
    in_F, out_F = config['TRAIN']['input_track_size'], config['TRAIN']['output_track_size']
    bar = Bar(f"EVAL ADE_FDE", fill="#", max=len(dataloader))

    batch_size = bs
    batch_id = 0
    ade = 0
    fde = 0
    ade_batch = 0 
    fde_batch = 0
    for i, batch in enumerate(dataloader):
        joints, masks, padding_mask = batch
        padding_mask = padding_mask.to(config["DEVICE"])
   
        in_joints, in_masks, out_joints, out_masks, padding_mask = batch_process_coords(joints, masks, padding_mask, config, modality_selection)
        pred_joints = inference(model, config, in_joints, padding_mask, out_len=out_F)

        out_joints = out_joints.cpu() 

        pred_joints = pred_joints.cpu().reshape(out_joints.size(0), 12, 1, 2)    
        
        for k in range(len(out_joints)):

            person_out_joints = out_joints[k,:,0:1]
            person_pred_joints = pred_joints[k,:,0:1]

            gt_xy = person_out_joints[:,0,:2]
            pred_xy = person_pred_joints[:,0,:2]
            sum_ade = 0
                
            for t in range(12):
                d1 = (gt_xy[t,0].detach().cpu().numpy() - pred_xy[t,0].detach().cpu().numpy())
                d2 = (gt_xy[t,1].detach().cpu().numpy() - pred_xy[t,1].detach().cpu().numpy())
             
                dist_ade = [d1,d2]
                sum_ade += np.linalg.norm(dist_ade)
            sum_ade /= 12
            ade_batch += sum_ade
            d3 = (gt_xy[-1,0].detach().cpu().numpy() - pred_xy[-1,0].detach().cpu().numpy())
            d4 = (gt_xy[-1,1].detach().cpu().numpy() - pred_xy[-1,1].detach().cpu().numpy())
            dist_fde = [d3,d4]
            scene_fde = np.linalg.norm(dist_fde)

            fde_batch += scene_fde
        batch_id+=1

    ade = ade_batch/((batch_id-1)*batch_size+len(out_joints))
    fde = fde_batch/((batch_id-1)*batch_size+len(out_joints))
    return ade, fde

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,  help="checkpoint path")
    parser.add_argument("--split", type=str, default="test", help="Split to use. one of [train, test, valid]")
    parser.add_argument("--metric", type=str, default="vim", help="Evaluation metric. One of (vim, mpjpe)")
    parser.add_argument("--modality", type=str, default="traj+2dbox", help="available modality combination from['traj','traj+2dbox']")

    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
        
    ################################
    # Load checkpoint
    ################################

    logger = create_logger('')
    logger.info(f'Loading checkpoint from {args.ckpt}') 
    ckpt = torch.load(args.ckpt, map_location = torch.device('cpu'))
    config = ckpt['config']
    
    if torch.cuda.is_available():
        config["DEVICE"] = f"cuda:{torch.cuda.current_device()}"
        torch.cuda.manual_seed(0)
    else:
        config["DEVICE"] = "cpu"


    logger.info("Initializing with config:")
    logger.info(config)

    ################################
    # Initialize model
    ################################

    model = create_model(config, logger)
    model.load_state_dict(ckpt['model']) 
    ################################
    # Load data
    ################################

    in_F, out_F = config['TRAIN']['input_track_size'], config['TRAIN']['output_track_size']
    assert in_F == 9
    assert out_F == 12

    name = config['DATA']['train_datasets']
    
    dataset = create_dataset(name[0], logger, split=args.split, track_size=(in_F+out_F), track_cutoff=in_F)

    
 
    bs = config['TRAIN']['batch_size']
    dataloader = DataLoader(dataset, batch_size=bs, num_workers=config['TRAIN']['num_workers'], shuffle=False, collate_fn=collate_batch)
    ade,fde = evaluate_ade_fde(model, args.modality, dataloader, bs, config, logger, return_all=True)


    print('ADE: ', ade)
    print('FDE: ', fde)
    

