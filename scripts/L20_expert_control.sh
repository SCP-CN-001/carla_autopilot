# setup environment variables
export SHELL_PATH=$(dirname $(readlink -f $0))
export WORKSPACE=${SHELL_PATH}/..
export CARLA_ROOT=${WORKSPACE}/CARLA_Leaderboard_20
export LEADERBOARD_ROOT=${WORKSPACE}/leaderboard_20/leaderboard
export SCENARIO_RUNNER_ROOT=${WORKSPACE}/leaderboard_20/scenario_runner
export AUTOPILOT_ROOT=${WORKSPACE}

export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.14-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:${LEADERBOARD_ROOT}
export PYTHONPATH=$PYTHONPATH:${SCENARIO_RUNNER_ROOT}
export PYTHONPATH=$PYTHONPATH:${AUTOPILOT_ROOT}

export PYTHONPATH=$PYTHONPATH:/home/rowena/Documents/RAMBLE/carla_autopilot

# general parameters
export PORT=2000
export TM_PORT=2500
export DEBUG_CHALLENGE=1

# simulation setup
export ROUTES=${WORKSPACE}/leaderboard_20/leaderboard/data/routes_training.xml
export ROUTES_SUBSET=0
export REPETITIONS=1

export CHALLENGE_TRACK_CODENAME=MAP
export TEAM_AGENT=${WORKSPACE}/src/expert_agent.py
export TEAM_CONFIG=${WORKSPACE}/src/configs/expert_agent.yaml
export TIME_STAMP=$(date +"%s")
export CHECKPOINT=${WORKSPACE}/logs/expert_control/route_${ROUTES_SUBSET}_${TIME_STAMP}.json

export RESUME=1
export TM_SEED=0

python -m cProfile -o ${WORKSPACE}/logs/program.prof ${WORKSPACE}/src/leaderboard_custom/leaderboard_evaluator.py \
    --port=${PORT} \
    --traffic-manager-port=${TM_PORT} \
    --routes=${ROUTES} \
    --routes-subset=${ROUTES_SUBSET} \
    --repetitions=${REPETITIONS} \
    --track=${CHALLENGE_TRACK_CODENAME} \
    --checkpoint=${CHECKPOINT} \
    --agent=${TEAM_AGENT} \
    --agent-config=${TEAM_CONFIG} \
    --debug=${DEBUG_CHALLENGE} \
    --resume=${RESUME} \
    --traffic-manager-seed=${TM_SEED}
