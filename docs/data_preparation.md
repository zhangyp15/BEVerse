# Dataset Preparation

## Dataset structure

It is recommended to symlink the dataset root to `$BEVerse/data`.
If your folder structure is different from the following, you may need to change the corresponding paths in config files.

```
BEVerse
├── mmdet3d
├── tools
├── configs
├── projects
├── data
│   ├── nuscenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
```

## Download and prepare the nuScenes dataset

Download nuScenes V1.0 full dataset data [HERE](https://www.nuscenes.org/download), including the map extensions. Prepare nuscenes data by running

```bash
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
```
