Modified from the official mmdet3d [getting_started.md](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/getting_started.md).

# Prerequisites
BEVerse is developed with the following version of modules.
- Linux or macOS (Windows is not currently officially supported)
- Python 3.7
- PyTorch 1.10.0
- CUDA 11.3.1
- GCC 7.3.0
- MMCV==1.3.14
- MMDetection==2.14.0
- MMSegmentation==0.14.1

Check [beverse.yaml](../beverse.yaml) for detailed dependencies.

# Installation

**a. Create a conda virtual environment and activate it.**

```shell
conda create -n beverse python=3.7 -y
conda activate beverse
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**

```shell
conda install pytorch==1.10.0 torchvision==0.11.1 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch
```

**c. Install [MMCV](https://mmcv.readthedocs.io/en/latest/), [MMDetection](https://github.com/open-mmlab/mmdetection), [MMSegmentation](https://github.com/open-mmlab/mmsegmentation), and other requirements.**

```shell
pip install -r requirements.txt
```

**f. Clone the BEVerse repository.**

```shell
git clone https://github.com/zhangyp15/BEVerse
cd BEVerse
```

**g.Install build requirements and then install BEVerse.**

```shell
python setup.py develop
```