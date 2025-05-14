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

[简单视频](https://pan.baidu.com/s/1MuLAX2N4oXyPpaJAQ9cNTg&fc7e)
（不同步态、不同速度自由切换、可高速运行；仍有改进空间）

