# Walker

# [system install ](README_ISAAC_install.md)
```bash
pip install rsl-rl-lib==2.2.4
```

# free style locomotion
![Style walker](docs/resources/gait_style.png)


## Stage 1
### Training
```bash
python scripts/rsl_rl/train_gait.py --task=Quadruped-Go2StyleLatent-v0 --headless
```

### Playing
```bash
python scripts/rsl_rl/play_gait.py --task=Quadruped-Go2StyleLatent-Play-v0
```

[Simple video](https://pan.baidu.com/s/1MuLAX2N4oXyPpaJAQ9cNTg&fc7e)
（Free switching between different gaits and speeds, capable of high-speed operation; There is still room for improvement）


### Training

```bash
python scripts/rsl_rl/train_gait.py --task=H1StyleLatent-v0 --headless
```

### Playing
```bash
python scripts/rsl_rl/play_gait.py --task=H1StyleLatent-Play-v0
```
At present, we only focus on the performance of bipedal operation, and there will be optimization in the future
(1) Better control of the relationship between speed and stride frequency.
(2) Course learning still needs to be implemented more reasonably.
(3) Can add the rewards of dual arm symmetry operation and constraint control.

## Stage 2

TODO

