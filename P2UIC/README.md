<p align="center">
  <h1 align="center">P^2UIC: Plug-and-Play Physics-Aware Contrastive Mamba Framework for Underwater Image Captioning</h1>
  <p align="center">

<p align="center">
<br />
    <strong>Chunlei Wang</strong></a>
    ·
    <strong>Wenquan Feng</strong></a>
    ·
    <strong>Binghao Liu</strong></a>
    ·
    <strong>Xianyu Zhao</strong></a>    
    ·
    <strong>Kejun Zhao</strong></a>
    ·
    <strong>Qi Zhao</strong></a>
    <br />
 </p>

## Introduction

This repo is the implementation of "P^2UIC: Plug-and-Play Physics-Aware Contrastive Mamba Framework for Underwater Image Captioning"

The overall architecture of Multi-Head Criss-Cross Mamba:

<p align="center">
  <img src="images/P2UIC.png" width="720">
</p>

The heatmap visualization:

<p align="center">
  <img src="images/vis_1.png" width="540">
</p>
<p align="center">
  <img src="images/vis_2.png" width="540">
</p>

## Usage

### Install

Clone [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) repo and [CCMamba](https://github.com/BinghaoLiu/CCMamba), add the codes of `configs`, `mmseg/datasets/crack500.py` and `mmseg/models/decode_heads/ccmamba_head.py` into corresponding files of MMSegmentation.

Then, run
`pip install -v -e .`
to regist Crack500 dataset and CCMamba model.

### Train and Test

+ Use the following command for training
  
  ```
  python tools/train.py \
  config_path \
  --work-dir work_path
  ```

+ Use the following command for testing
  
  ```
  python tools/test.py \
  config_path \
  ckpt_path \
  --work-dir work_path
  ```

## Citation

If you have any question, please discuss with me by sending email to liubinghao@buaa.edu.cn

## References

The code is based on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation), [Mamba](https://github.com/state-spaces/mamba) and [mamba-minimal](https://github.com/johnma2006/mamba-minimal). Thanks for their great works!
