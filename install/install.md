## Prerequisites in general
- Linux
- Python 3 (I strong recommend to use [Anaconda](https://anaconda.org/) to set-up your Python environment and build a new conda virtual environment for this project.)
- CUDA==9.0 and CuDNN==6.0
- Pytorch=1.0.0 and latest version of torchvision (**Currently, higher version of Pytorch is not supported.**)
- Other required 3rd-party packages is listed in [requirements.txt.](requirements.txt)
- If you want to visualize the generated 3D human model, please refer to the following section.



## Prerequisites for visualization
To sucessfully visualize the generated 3D human model, you need to prepare another **Python 2** environment and install following packages. 
Please note that a Python 2 environment is required because the essential package for visualization - opendr only supports Python 2.  
- Install [opendr](https://github.com/mattloper/opendr/wiki) and make sure you can successfully import opendr in Python 2. If you have any problems in installing opendr, please feel free to contact me or raise an issue.    
- Install opencv-python.  
- Download code of [SMPL](http://smpl.is.tue.mpg.de/), unzip it and add it to your **Python 2** environment.
```bash
export PYTHONPATH="$PYTHONPATH:your_path_to_smpl"
```

## Setting of .bashrc or .zshrc
After installing all the prerequisties, I add the following lines in my .zshrc/.bashrc file
```bash
# cuda-9.0                                                                                                              
export PATH=/usr/local/cuda/bin:$PATH                                                                                   
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH                                                           

# Python 2.7.4, opendr installed and SMPL added to the environment                                                                                      
export PATH="/mnt/SSD/rongyu/software/anaconda2/bin:$PATH"                                                              
export PYTHONPATH="$PYTHONPATH:/mnt/SSD/rongyu/work/DCT/ICCV_release/smpl"                                              
                                                                                                                        
# Python 3.6, pytorch_1.0.0 is the conda virtual environment for this project.                                                                                                          
export PATH="/mnt/SSD/rongyu/software/anaconda3/bin:$PATH"                                                              
source activate pytorch_1.0.0 
```