# Delving Deep Into Hybrid Annotations for 3D Human Recovery in the Wild
Yu Rong, Ziwei Liu, Cheng Li, Kaidi Cao, Chen Change Loy

- [paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Rong_Delving_Deep_Into_Hybrid_Annotations_for_3D_Human_Recovery_in_ICCV_2019_paper.pdf).  
- [Project Page](https://penincillin.github.io/dct_iccv2019)  
</br></br>
![Teaser Image](https://penincillin.github.io/project/dct_iccv2019/framework.png)  

Part of the code is inspired by [Pytorch-CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [HMR](https://github.com/akanazawa/hmr). Thanks the contributions of their authors.


## Prerequisites
### Install 3rd-party packages
Please refer to [install.md](install/install.md) to install the required packages and code.
### Prepare DCT code
- Clone this repo
```bash
git clone git@github.com:penincillin/DCT_ICCV-2019.git
cd DCT_ICCV-2019
```
### Prepare Data and Models
Download processed datasets, demo images and pretrained weights and models from [Google Drive](https://drive.google.com/file/d/1TsQXGyf4Cec1UtarYuzACo_7Wwx-b02r/view?usp=sharing), uznip it and place it in the root directory of DCT_ICCV-2019.


## Training
### Prepare Real-time Visualization
Before training starts, to visualize the training results and the loss curve in real-time, please run ```python -m visdom.server 8097``` and click the URL [http://localhost:8097](http://localhost:8097)

### Train All Dataset 
#### Use Image as input and use all annotations
```
sh script/train_all_img.sh
```
#### Use Image and IUV as input and use all annotations
```
sh script/train_all_img_iuv.sh
```

### Train UP-3D Dataset 
#### Use Images as input and use all annotations
```
sh script/train_up3d_img_3d_dp.sh
```
#### Use Images as input and not use 3D annotations
```
sh script/train_up3d_img_dp.sh
```
#### Use Images and IUV maps as input and use all annotations
```
sh script/train_up3d_img_iuv_3d_dp.sh
```
#### Use Images and IUV maps as input and not use 3D annotations
```
sh script/train_up3d_img_iuv_dp.sh
```

## Evaluation

### Evaluate on UP-3D Dataset
#### Evaluate models use images as input
```
sh script/test_up3d_img.sh
```
#### Evaluate models use images and IUV maps as input
```
sh script/test_up3d_img_iuv.sh
```
After run evaluation code, the results are stored in ```DCT_ICCV-2019/evaluate_results```. To visualize the results, run
```
sh script/visualize.sh
```
The generated images are stored in ```DCT_ICCV-2019/evaluate_results/images```.


## Run Inference on Other Images
To run the model on your own images, just center crop the images according to each person.  
Then update the content of image list file stored in ```DCT_ICCV-2019/dct_data/demo/img_list.txt```.  

To run inference for the pre-processed images, please checkout to the **```inference```** branch first.  
**```Remember to commit the current changes you made in master branch first.```**  

Run inference code:  
```
sh script/infer.sh
```
The results are stored in ```DCT_ICCV-2019/inference_results```. To further visualize the results, run visualization code:
```
sh script/visualize.sh
```
The generated images are stored in ```DCT_ICCV-2019/inference_results/images```.


## Citation
Please cite the paper in your publications if it helps your research:

    @inproceedings{Rong_2019_ICCV,
      author = {Rong, Yu and Liu, Ziwei and Li, Cheng and Cao, Kaidi and Loy, Chen Change},
      title = {Delving Deep Into Hybrid Annotations for 3D Human Recovery in the Wild},
      booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
      month = {October},
      year = {2019}
      }

