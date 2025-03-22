# Walker

# [ststem install ](README_ISAAC_install.md)

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