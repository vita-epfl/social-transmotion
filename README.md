<div align="center">
<h1> Social-Transmotion:<br>  Promptable Human Trajectory Prediction </h1>
<h3>Saeed Saadatnejad*, Yang Gao*, Kaouther Messaoud, Alexandre Alahi
</h3>
<h4> <i> International Conference on Learning Representations (ICLR), Austria, May 2024 </i></h4>

[[Paper](https://arxiv.org/abs/2312.16168)] [[ICLR page](https://iclr.cc/virtual/2024/poster/18604)] [[Poster](docs/Poster.pdf)] [[Slides](docs/iclr_slides.pdf)]


<image src="docs/social-transmotion.png" width="500">

</div>

<div align="center"> <h3> Abstract </h3>  </div>
<div align="justify">

Accurate human trajectory prediction is crucial for applications such as autonomous vehicles, robotics, and surveillance systems. Yet, existing models often fail to fully leverage the non-verbal social cues human subconsciously communicate when navigating the space. To address this, we introduce Social-Transmotion, a generic Transformer-based model that exploits diverse and numerous visual cues to predict human behavior. We translate the idea of a prompt from Natural Language Processing (NLP) to the task of human trajectory prediction, where a prompt can be a sequence of x-y coordinates on the ground, bounding boxes in the image plane, or body pose keypoints in either 2D or 3D. This, in turn, augments trajectory data, leading to enhanced human trajectory prediction. Using masking technique, our model exhibits flexibility and adaptability by capturing spatiotemporal interactions between agents based on the available visual cues. We delve into the merits of using 2D versus 3D poses, and a limited set of poses. Additionally, we investigate the spatial and temporal attention map to identify which keypoints and time-steps in the sequence are vital for optimizing human trajectory prediction. Our approach is validated on multiple datasets, including JTA, JRDB, Pedestrians and Cyclists in Road Traffic, and ETH-UCY.
</br>


# Getting Started

Install the requirements using `pip`:
```
pip install -r requirements.txt
```

We have conveniently added the preprocessed data to the release section of the repository (for license details, please refer to the original papers).
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

```
@InProceedings{saadatnejad2024socialtransmotion,
      title={Social-Transmotion: Promptable Human Trajectory Prediction}, 
      author={Saeed Saadatnejad and Yang Gao and Kaouther Messaoud and Alexandre Alahi},
      year={2024},
      booktitle={International Conference on Learning Representations (ICLR)},
}
```
