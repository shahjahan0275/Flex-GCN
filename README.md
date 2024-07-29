# Flex-GCN
# Flexible Graph Convolutional Network for 3D Human Pose Estimation [The 35th British Machine Vision Conference 2024] 
This repository contains the official PyTorch implementation of the Iterative Graph Filtering Network for 3D Human Pose Estimation authored by Abu Taib Mohammed Shahjahan and A. Ben Hamza. If you discover our code to be valuable for your research, kindly consider including the following citation:
 
<div style="position: relative; display: inline-block; background-color: #f5f5f5; padding: 0px; border-radius: 0px;">
  <pre id="code-block" style="margin: 0; font-family: monospace; background-color: #f5f5f5; padding: 0;">
    
###### 
    @article{islam2023iterative,
      title={Flexible Graph Convolutional Network for 3D Human Pose Estimation},
      author={Shahjahan, Abu Taib Mohammed and Hamza, A Ben},
      conference={The 35th British Machine Vision Conference 2024},
      year={2024},      
    }
  </pre>
</div>

### Network Architecture

<div align="center">
  <img src="https://github.com/shahjahan0275/Flex-GCN/blob/main/demo/Network_Architechture.png" alt="Network_Architechture" width="800" height="300">
</div>


The PyTorch implementation for Flex-GCN

### Qualitative and quantitative results

![Greeting](https://github.com/shahjahan0275/Flex-GCN/blob/main/demo/Greeting.gif)

### Results on Human3.6M

SemGCN 	57.6mm 	-
High-order GCN 	55.6mm 	43.7mm
HOIF-Net 	54.8mm 	42.9mm
Weight Unsharing 	52.4mm 	41.2mm
MM-GCN 	51.7mm 	40.3mm
Modulated GCN 	49.4mm 	39.1mm
Ours 	47.1mm 	38.7mm
 	 	

|      Method       |  MPJPE (P1)   | PA-MPJPE (P2) |
| ------------------| ------------- | ------------- |
|      [SemGCN](https://github.com/garyzhao/SemGCN)      |    57.6mm     |      -        |
|    [High-order](https://github.com/ZhimingZo/HGCN)     |    55.6mm     |    43.7mm     |
|     [HOIF-Net](https://github.com/happyvictor008/Higher-Order-Implicit-Fairing-Networks-for-3D-Human-Pose-Estimation)      |    54.8mm     |    42.9mm     |
| [Weight Unsharing](https://github.com/tamasino52/Any-GCN)  |    52.4mm     |    41.2mm     |
|      [MM-GCN](https://github.com/JaeYungLee/MM_GCN)       |    51.7mm     |    40.3mm     |
|     [Modulated](https://github.com/ZhimingZo/Modulated-GCN)     |    49.4mm     |    39.1mm     |
|       Ours        |    46.9mm     |    38.6mm     |

## Quick Start
This repository is built upon Python v3.8 and Pytorch v1.8.0 on Ubuntu 20.04.4 LTS. All experiments are conducted on a single NVIDIA RTX 3070 GPU with 8GB of memory.

## Dependencies

Please make sure you have the following dependencies installed:

    -PyTorch >= 1.8.0
    -NumPy
    -Matplotlib

## Dataset
Our model is evaluated on [Human3.6M](http://vision.imar.ro/human3.6m/description.php) and [MPI-INF-3DHP](http://vision.imar.ro/human3.6m/description.php) datasets. Please put the datasets in <span style="background-color: #f0f0f0">./dataset</span> directory.

## Human3.6M
2D detections for Human3.6M dataset are provided by VideoPose3D Pavllo et al.

## MPI-INF-3DHP
We set up the MPI-INF-3DHP dataset in the same way as [PoseAug](https://github.com/jfzhang95/PoseAug). Please refer to [DATASETS.md](https://github.com/jfzhang95/PoseAug/blob/main/DATASETS.md) to prepare the dataset file.

# Training from Scratch
<div style="position: relative; display: inline-block; background-color: #f5f5f5; padding: 0px; border-radius: 0px;">
  <pre id="code-block" style="margin: 0; font-family: monospace; background-color: #f5f5f5; padding: 0;">
    
###### 
    python main_graph.py  --pro_train 1 --beta 0.2 --k hrn --batchSize 512 --hid_dim 384 --save_model 1  --save_dir './checkpoint/train_result/' --post_refine --save_out_type post --show_protocol2
  </pre>
</div>
