export HYDRA_FULL_ERROR=1
export DOCKER_ISAACLAB_PATH=.
export DISPLAY=:1

DEBUG=n
GRASP_TYPE=active
OBJ_TYPE=superquadric
ROBOT_NAME=shadow

LOG_PROJECT_NAME="${USER}_isaaclab"
EXPERIMENT_NAME="${ROBOT_NAME}-${GRASP_TYPE}-${OBJ_TYPE}"
RUN_NAME="$(hostname)-${EXPERIMENT_NAME}-n${NUM_ENVS}"

if [ "${DEBUG}" == "t" ]; then
    NUM_ENVS=5
    ARGS=""
else
    NUM_ENVS=1024
    ARGS="--headless --logger wandb --experiment_name ${EXPERIMENT_NAME} --run_name ${RUN_NAME} --log_project_name ${LOG_PROJECT_NAME}"
fi

echo "Running experiment with the following parameters:"
echo "GRASP_TYPE: ${GRASP_TYPE}"
echo "OBJ_TYPE: ${OBJ_TYPE}"
echo "ROBOT_NAME: ${ROBOT_NAME}"
echo "NUM_ENVS: ${NUM_ENVS}"
echo "EXPERIMENT_NAME: ${EXPERIMENT_NAME}"
echo "RUN_NAME: ${RUN_NAME}"
echo "LOG_PROJECT_NAME: ${LOG_PROJECT_NAME}"
# Ensure the script is run from the correct directory

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-Nextage-Shadow-Grasp-Direct-v0 \
    --num_envs ${NUM_ENVS} \
    env.grasp_type=${GRASP_TYPE} \
    env.object_type=${OBJ_TYPE} \
    env.robot_name=${ROBOT_NAME} ${ARGS} \
