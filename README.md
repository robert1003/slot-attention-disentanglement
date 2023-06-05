# Exploring the Role of the Bottleneck in Slot-Based Models Through Covariance Regularization

This repo contains code adapted from [here](https://github.com/evelinehong/slot-attention-pytorch) for the following work:

"Improving Slot Attention Through Disentanglement"

[Andrew Stange](https://www.linkedin.com/in/andrewstange1/), [Robert Lo](https://robertlo.tech), [Abishek Sridhar](https://www.linkedin.com/in/abishek-sridhar5/), [Kousik Rajesh](https://www.linkedin.com/in/kousik-rajesh/)

## Abstract

In this project we attempt to make slot-based models with an image reconstruction objective competitive with those that use a feature reconstruction objectives on real world datasets. We propose a loss-based approach to constricting the bottleneck of slot-based models, allowing larger-capacity encoder networks to be used with Slot Attention without producing degenerate stripe-shaped masks. 
    
We find that our proposed method offers an improvement over the baseline Slot Attention model but does not reach the performance of DINOSAUR on the COCO2017 dataset. Throughout this project, we confirm the superiority of a feature reconstruction objective over an image reconstruction objective and explore the role of the architectural bottleneck in slot-based models.

## File Structure

* `./pytorch`: Main experiment code.
* `./tf`: Cloned from official impl. Contains tfdataset to pytorch dataset conversion script.

## Dataset

We used two dataset in our work, ColoredMdSprite (cmdsprite) and COCO (coco).

### ColoredMdSprite

```
# Download colored background dataset
wget https://storage.googleapis.com/multi-object-datasets/multi_dsprites/multi_dsprites_colored_on_colored.tfrecords

# Convert dataset from tfrecords to .npy format. This will generate the training set
python3 tf/tfrecord_convert.py --infile <tfrecord path> --outdir ~/data/colored --type=colored_on_colored

# For evaluation set:
python3 tf/tfrecord_convert.py --infile <tfrecord path> --outdir ~/data/colored --type=colored_on_colored --start-idx 60000 –num=320
```

### COCO

```
# Download the official COCO dataset. Val size: 778 MB, 5k images. Train size: 18 GB, ~118k images
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# Generate ViT embeddings of COCO dataset
python3 experiments.py --output_dir [coco embedding path] --batch_size 64 --num_workers 16 --dataset_path [coco path] --split [val/train]
```

## Experiments

Because we developed our code on the fly with our experiments, to reproduce results in our work, please first checkout to the corresponding git hash then run the command.

Format: [git hash], [command]

* Table 1: ARI on ColoredMdSprites and COCO
```
* Slot-Attn on..
  * ..ColoredMdSprites: `ec1a6fb`, `python3 train.py --model_dir [checkpoint path] --batch_size 64 --num_slots 5 --hid_dim 64 --dataset multid --dataset_path [cmdsprite path] --num_workers 16 --base`
  * ..COCO: Numbers taken from DINOSAUR paper.
* CovLoss on..
  * ..ColoredMdSprites: `ec1a6fb`, `python3 train.py --model_dir [checkpoint path] --batch_size 64 --num_slots 5 --hid_dim 64 --dataset multid --dataset_path [cmdsprite path] --num_workers 16 --std_target 0 --proj_weight 1.0 --var_weight 0 --cov_weight 1.0 --cov-div-sq --proj-layernorm`
  * ..COCO: `af7b5f8`, `python3 train.py --model_dir [checkpoint path] --batch_size 64 --num_slots 7 --hid_dim 64 --dinosaur --grad-clip 1 --dataset_path [coco path] --embed_path [coco embedding path] --dinosaur-downsample --num_workers 16 --std_target 0 --proj_weight 1.0 --var_weight 0 --cov_weight 1.0 --cov-div-sq --proj-layernorm --decoder_init_size 8 --decoder_hid_dim 32 --decoder_num_conv_layers 6 --decoder_type coco-adaptive --print_model`
* CosineLoss on..
  * ..ColoredMdSprites: `be8d9c8`, `python3 train.py --model_dir [checkpoint path] --batch_size 64 --num_slots 5 --hid_dim 64 --dataset multid --dataset_path [cmdsprite path] --num_workers 8 --info-nce --info-nce-weight 0.01 --info-nce-warmup 0 --store_freq 1000 --identity-proj --proj_dim 64`
  * ..COCO: `d0e2c0`, `python3 train.py --model_dir [checkpoint path] --batch_size 64 --num_slots 7 --hid_dim 128 --dinosaur --grad-clip 1 --dataset_path [coco path] --embed_path [coco embedding path] --dinosaur-downsample --num_workers 16 --info-nce --info-nce-weight 0.01 --info-nce-warmup 0 --store_freq 1000 --identity-proj --proj_dim 128 --decoder_init_size 8 --decoder_hid_dim 32 --decoder_num_conv_layers 6 --decoder_type coco-adaptive --print_model --store_freq 1000`
```

* Table 3: ARI on COCO for DINOSAUR
```
* DINOSAUR: `db5b29a`, `python3 train.py --model_dir [checkpoint path] --batch_size 64 --num_slots 7 --hid_dim 256 --dinosaur-emb --decoder_hid_dim 3072 --grad-clip 1 --dataset_path [coco path] --embed_path [coco embedding path] --dinosaur-downsample --num_workers 16 --std_target 0 --proj_weight 0 --var_weight 0 --cov_weight 0 --identity-proj --proj_dim 256 --print_model`
* DINOSAUR + CovLoss: `db5b29a`, `python3 train.py --model_dir [checkpoint path] --batch_size 64 --num_slots 7 --hid_dim 256 --dinosaur-emb --decoder_hid_dim 3072 --grad-clip 1 --dataset_path [coco path] --embed_path [coco embedding path] --dinosaur-downsample --num_workers 16 --std_target 1.0 --proj_weight 1.0 --var_weight 1.0 --cov_weight 1.0 --identity-proj --proj_dim 256 --print_model`
* DINOSAUR with large dim: `db5b29a`, `python3 train.py --model_dir [checkpoint path] --batch_size 64 --num_slots 7 --hid_dim 1024 --dinosaur-emb --decoder_hid_dim 3072 --grad-clip 1 --dataset_path [coco path] --embed_path [coco embedding path] --dinosaur-downsample --num_workers 16 --std_target 0 --proj_weight 0 --var_weight 0 --cov_weight 0 --identity-proj --proj_dim 1024 --print_model`
* DINOSAUR with large dim + CovLoss: `db5b29a`, `python3 train.py --model_dir [checkpoint path] --batch_size 64 --num_slots 7 --hid_dim 1024 --dinosaur-emb --decoder_hid_dim 3072 --grad-clip 1 --dataset_path [coco path] --embed_path [coco embedding path] --dinosaur-downsample --num_workers 16 --std_target 1.0 --proj_weight 1.0 --var_weight 1.0 --cov_weight 1.0 --identity-proj --proj_dim 1024 --print_model`
```

## Evaluation (ARI calculation and mask visualization)

Use the same git hash and command above, only need to replace `train.py` to `eval.py`. Note that ARI calculation is not deterministic, so the numbers might not match exactly.
