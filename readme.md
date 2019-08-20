# Delving Deep Into Hybrid Annotations for 3D Human Recovery in the Wild

This is the implementation of our ICCV 2019 accepted paper *Delving Deep Into Hybrid Annotations for 3D Human Recovery in the Wild* [paper on arxiv](https://arxiv.org/abs/1908.06442).


## Prerequisites
### Install 3rd-party package
Please refer to [install.md](https://github.com/penincillin/DCT_ICCV-2019/blob/master/insstall/install.md) to install the required packages and code.
### Installation
- Clone this repo
```bash
git clone git@github.com:penincillin/DCT_ICCV-2019.git
cd DCT_ICCV-2019
```
### Prepare Data and Models
Demo images and pretrained models can be downloaded from [Google Drive](https://drive.google.com/file/d/1oLMCfLTckvHzDq_UYc3p0pVd4CiFYjvR/view?usp=sharing). After downloading this zip file, uznip it and place it in the root directory of DCT_ICCV-2019.


## Run Inference
```
sh infer.sh
```
The predicted results will be stored in DCT_ICCV-2019/evaluate_results.
To run the model on your own images, just center crop the images according to each person and change the path of image list file to your own.


## Training code is comming soon.


## Citation
Please cite the paper in your publications if it helps your research:

    @inproceedings{Rong_2019_ICCV,
      author = {Rong, Yu and Liu, Ziwei and Li, Cheng and Cao, Kaidi and Change Loy, Chen},
      booktitle = {Proceedings of the IEEE international conference on computer vision},
      title = {Delving Deep into Hybrid Annotations for 3D Human Recovery in the Wild},
      year = {2019}
      }

