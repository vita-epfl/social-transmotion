<div align="center">
<h1> Social-Transmotion:<br>  Promptable Human Trajectory Prediction </h1>
<h3>Saeed Saadatnejad*, Yang Gao*, Kaouther Messaoud, Alexandre Alahi
</h3>

 
<image src="docs/social-transmotion.png" width="600">
</div>

<div align="center"> <h3> Abstract </h3>  </div>
<div align="justify">

Accurate human trajectory prediction is crucial for applications such as autonomous vehicles, robotics, and surveillance systems. Yet, existing models often fail to fully leverage the non-verbal social cues human subconsciously communicate when navigating the space.
To address this, we introduce Social-Transmotion, a generic model that exploits the power of transformers to handle diverse and numerous visual cues, capturing the multi-modal nature of human behavior. We translate the idea of a prompt from Natural Language Processing (NLP) to the task of human trajectory prediction, where a prompt can be a sequence of x-y coordinates on the ground, bounding boxes or body poses. This, in turn, augments trajectory data, leading to enhanced human trajectory prediction.
Our model exhibits flexibility and adaptability by capturing spatiotemporal interactions between pedestrians based on the available visual cues, whether they are poses, bounding boxes, or a combination thereof.
By the masking technique, we ensure our model's effectiveness even when certain visual cues are unavailable, although performance is further boosted with the presence of comprehensive visual data.
</br>


# Getting Started

Install the requirements using `pip`:
```
pip install -r requirements.txt
```

We have conveniently added the preprocessed data to the release section of the repository.
Place the data subdirectory of JTA under `data/jta_all_visual_cues` and the data subdirectory of JRDB under `data/jrdb_2dbox` of the repository.

# Training and Testing

## JTA dataset
You can train the Social-Transmotion model on this dataset using the following command:
```
python train_jta.py --cfg configs/jta_all_visual_cues.yaml --exp_name jta
```


To evaluate the trained model, use the following command:
```
python evaluate_jta.py --ckpt ./experiments/jta/checkpoints/checkpoint.pth.tar --metric ade_fde --modality traj+all
```
Please note that the evaluation modality can be any of `[traj, traj+2dbox, traj+3dpose, traj+2dpose, traj+3dpose+3dbox, traj+all]`.
For the ease of use, we have also provided the trained model in the release section of this repo. In order to use that, you should pass the address of the saved checkpoint via `--ckpt`.

## JRDB dataset
You can train the Social-Transmotion model on this dataset using the following command:
```
python train_jrdb.py --cfg configs/jrdb_2dbox.yaml --exp_name jrdb
```

To evaluate the trained model, use the following command:
```
python evaluate_jrdb.py --ckpt ./experiments/jrdb/checkpoints/checkpoint.pth.tar --metric ade_fde --modality traj+2dbox
```
Please note that the evaluation modality can be one any of `[traj, traj+2dbox]`.
For the ease of use, we have also provided the trained model in the release section of this repo. In order to use that, you should pass the address of the saved checkpoint via `--ckpt`.

# Work in Progress

This repository is work-in-progress and will continue to get updated and improved over the coming months.
