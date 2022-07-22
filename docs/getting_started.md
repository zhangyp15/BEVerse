# Getting started with BEVerse
## Training

To train BEVerse with 8 GPUs, run:
```bash
bash tools/dist_train.sh $CONFIG 8
```

## Evaluation

To evaluate BEVerse with 8 GPU, run:
```bash
bash tools/dist_test.sh $YOUR_CONFIG $YOUR_CKPT 8 --eval=bbox --mtl
```

## Visualization

To visualize the predictions, run:
```bash
python tools/test.py $YOUR_CONFIG $YOUR_CKPT --eval=bbox --mtl --show
```

