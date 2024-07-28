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
|      [SemGCN](https://github.com/garyzhao/SemGCN)       |    57.6mm     |      -        |
|    [High-order](https://github.com/ZhimingZo/HGCN)     |    55.6mm     |    43.7mm     |
|     [HOIF-Net](https://github.com/happyvictor008/Higher-Order-Implicit-Fairing-Networks-for-3D-Human-Pose-Estimation)      |    54.8mm     |    42.9mm     |
| [Weight Unsharing](https://github.com/tamasino52/Any-GCN)  |    52.4mm     |    41.2mm     |
|      [MM-GCN](https://github.com/JaeYungLee/MM_GCN)       |    51.7mm     |    40.3mm     |
|     [Modulated](https://github.com/ZhimingZo/Modulated-GCN)     |    49.4mm     |    39.1mm     |
|       Ours        |    46.9mm     |    38.6mm     |


