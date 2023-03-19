## slot-attention-pytorch
 This repo is the Pytorch Implementation of paper "Object-Centric Learning with Slot Attention". It uses the slot attention module from [here](https://github.com/lucidrains/slot-attention) but augments it with encoder, decoder and train/test scripts.

 The official Tensorflow repository has been released <a href="https://github.com/google-research/google-research/tree/master/slot_attention">here</a>. This repo should be a direct translation from the official Tensorflow repo to a PyTorch version.

## Usage

Run Slot Attention with Project Head

```
python train.py --dataset_path=~/datasets/CLEVR_v1.0/images/ --dataset=clevr
```

Run base Slot Attention model (no projection head)

```
python train.py --dataset_path=~/datasets/CLEVR_v1.0/images/ --dataset=clevr --base
```

## Citation

```bibtex
@misc{locatello2020objectcentric,
    title = {Object-Centric Learning with Slot Attention},
    author = {Francesco Locatello and Dirk Weissenborn and Thomas Unterthiner and Aravindh Mahendran and Georg Heigold and Jakob Uszkoreit and Alexey Dosovitskiy and Thomas Kipf},
    year = {2020},
    eprint = {2006.15055},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```
