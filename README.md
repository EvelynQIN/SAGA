<h1 align="center">
SAGA + FLEX
</h1>

- <strong>Body Models</strong>  
Download [SMPL-X body model and vposer v1.0 model](https://smpl-x.is.tue.mpg.de/index.html) and put them under /body_utils/body_models folder as below:
```
SAGA
│
└───body_utils
    │
    └───body_models 
        │
        └───smplx 
        │   └───SMPLX_FEMALE.npz
        │   └───...
        │
        └───mano (from flex)
        |   └───MANO_LEFT.pkl
        |   └───MANO_RIGHT.pkl
        |
        |
        └───vposer_v1_0
        │   └───snapshots
        │       └───TR00_E096.pt
        │   └───...
        │
        └───VPoser
        │   └───vposerDecoderWeights.npz
        │   └───vposerEnccoderWeights.npz
        │   └───vposerMeanPose.npz
    │
    └───...
│
└───...
```

## Dataset
### 
Download [GRAB](https://grab.is.tue.mpg.de/) object mesh

Download dataset for the first stage (GraspPose) from [[Google Drive]](https://drive.google.com/uc?export=download&id=1OfSGa3Y1QwkbeXUmAhrfeXtF89qvZj54)

Put them under /dataset as below,
```
SAGA
│
└───dataset 
    │
    └───GraspPose
    │   └───train
    │       └───s1
    │       └───...
    │   └───eval
    │       └───s1
    │       └───...
    │   └───test
    │       └───s1
    │       └───...
    │   
    └───contact_meshes
    │   └───airplane.ply
    │   └───...
│   └───replicagrasp (from flex)
    │   └───dset_info.npz
    │   └───receptacles.npz
    └───sbj (from flex)
    │   └───...
    │  
```
    
## Pretrained models
Put pretrained models under /pretrained_model as below,
```
SAGA
│
└───pretrained_model
    │
    └───male_grasppose_model.pt
    │   
    └───pgp.pth (from flex, group prior)
```


## Train
### First Stage: WholeGrasp-VAE training
```
python train_grasppose.py --data_path ./dataset/GraspPose --gender male --exp_name male
```

## Inference
### WholeGrasp-VAE sampling + SAGA GraspPose-Opt + FLEX opt
```
python opt_grasppose_saga.py --exp_name sage_with_flex_test --gender male --pose_ckpt_path pretrained_model/male_grasppose_model.pt --n_rand_samples_per_object 1 --object camera
```

## Visualization
Visualize the final optimization after flex opt. Fitting result is saved at (by default) _/results/$EXP_NAME/GraspPose/$OBJECT/$RECEPT/$ORNT_$INDEX/flex_fitting_results.npz.
```
cd visualization
python vis_flex.py --exp_name saga_with_flex_test --gender male --object camera --receptacle_name receptacle_aabb_Tbl2_Top2_frl_apartment_table_02 --ornt up --index 0
```

Visualize the intermediate optimization after saga opt. Fitting result is saved at (by default) _/results/$EXP_NAME/GraspPose/$OBJECT/$RECEPT/$ORNT_$INDEX/saga_fitting_results.npz.
```
cd visualization
python vis_saga.py --exp_name saga_with_flex_test --gender male --object camera --receptacle_name receptacle_aabb_Tbl2_Top2_frl_apartment_table_02 --ornt up --index 0
```
