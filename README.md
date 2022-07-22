# BEVerse

The official implementation of the paper [BEVerse: Unified Perception and Prediction in Birds-Eye-View for Vision-Centric Autonomous Driving](https://arxiv.org/abs/2205.09743).
 
## News
* **2022.07.20:** We release the code and models of BEVerse.

## Model Zoo
|Method | mAP      | NDS     | IoU (Map) | IoU (Motion) | VPQ | Model |
|:--------:|:----------:|:---------:|:--------:|:-------------:|:-----:|:-------:|
| [**BEVerse-Tiny**](configs/bevdet/bevdet-sttiny.py)       | 32.1 | 46.6 | 48.7 | 38.7 | 33.3 | [Google Drive](https://drive.google.com/file/d/1S2o8v6YFkeHMuJIpw-SWNDGySacH1xCV/view?usp=sharing)
| [**BEVerse-Small**](configs/bevdet4d/bevdet4d-sttiny.py) | 35.2 | 49.5 | 51.7 | 40.9 | 36.1 | [Google Drive](https://drive.google.com/file/d/1n0teAat6Qy_EeJdDfWcwm0x8FZ2wsAo9/view?usp=sharing)

## Installation
Please check [installation](docs/installation.md) for installation and [data_preparation](docs/data_preparation.md) for preparing the nuScenes dataset.

## Getting Started
Please check [getting_started](docs/getting_started.md) for training, evaluation, and visualization of BEVerse.

## Visualization
![visualization](figs/vis.jpg "Results on nuScenes")

## Acknowledgement
This project is mainly based on the following open-sourced projects: [open-mmlab](https://github.com/open-mmlab), [BEVDet](https://github.com/HuangJunJie2017/BEVDet), [HDMapNet](https://github.com/Tsinghua-MARS-Lab/HDMapNet), [Fiery](https://github.com/wayveai/fiery).

## Bibtex

If this work is helpful for your research, please consider citing the following BibTeX entry.
```
@article{zhang2022beverse,
  title={BEVerse: Unified Perception and Prediction in Birds-Eye-View for Vision-Centric Autonomous Driving},
  author={Zhang, Yunpeng and Zhu, Zheng and Zheng, Wenzhao and Huang, Junjie and Huang, Guan and Zhou, Jie and Lu, Jiwen},
  journal={arXiv preprint arXiv:2205.09743},
  year={2022}
}

@article{huang2021bevdet,
  title={BEVDet: High-performance Multi-camera 3D Object Detection in Bird-Eye-View},
  author={Huang, Junjie and Huang, Guan and Zhu, Zheng and Yun, Ye and Du, Dalong},
  journal={arXiv preprint arXiv:2112.11790},
  year={2021}
}
```

