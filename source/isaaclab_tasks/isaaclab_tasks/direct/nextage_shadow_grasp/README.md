```
HYDRA_FULL_ERROR=1  DOCKER_ISAACLAB_PATH=. DISPLAY=:0 ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Nextage-Shadow-Grasp-Direct-v0 --num_envs 2 env.grasp_type=passive env.object_type=superquadric
```

```
HYDRA_FULL_ERROR=1  DOCKER_ISAACLAB_PATH=. ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Nextage-Shadow-Grasp-Direct-v0 --num_envs 1024 env.grasp_type=passive env.object_type=superquadric --headless --logger wandb
```

### converting urdf to usd
```
 DOCKER_ISAACLAB_PATH=. python scripts/tools/convert_urdf.py scripts/my_models/shadow/floating_shadow.urdf scripts/my_models/shadow/floating_shadow.usd --fix-base --headless
```
then
```
cd scripts/my_models/shadow/ && usdcat --flatten floating_shadow.usd -o floating_shadow.usd
```
