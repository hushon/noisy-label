# noisy-label

## Setting up

```bash
docker pull hushon/pytorch:2.0
```

## Run training

```bash
CUDA_VISIBLE_DEVICES=0 python train.py ./configs/train_base.yaml
```