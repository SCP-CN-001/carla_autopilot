This directory is forked from branch `leaderboard_2` of the [autonomousvision/carla_garage](https://github.com/autonomousvision/carla_garage/tree/leaderboard_2), at commit ID 4d63da6.

I refactored the code to enhance readability and provide a more user-friendly interface for customizing sensors for data collection. You are welcome to raise issues or submit pull requests to improve the codebase.

## Changes

The main changes include:

1. Improve Efficiency and Stability

    I have enhanced the efficiency of the expert agent, doubling its performance compared to the original version. Additionally, the previous data collection method was prone to crashes because it stored all data in memory and wrote to the disk at the end of the run. The data collection process has been modified to write data in real-time, effectively preventing crashes caused by memory overflow.

    If you use the same sensor configuration as the original code, the agent is expected to run as fast as the original version while all the data are written to disk in real-time.

2. Readable Interface

    The settings for sensors, visualization, and data collection have been relocated to the config directory. This change simplifies customization of sensors and the data collection process, making it more user-friendly.

3. Dissolving the leaderboard and scenario_runner Modifications

    Many developers tend to pull the leaderboard and scenario_runner libraries and make direct modifications. While this practice can be convenient during development, it often leads to confusion about which parts of the libraries have been altered and raises concerns about the use of unauthorized information, particularly for the CARLA Leaderboard.

    To mitigate these issues, I have integrated the leaderboard and scenario_runner libraries as submodules, ensuring that they remain unmodified. Instead of altering these libraries directly, all necessary modifications are now made within the src/leaderboard_custom directory, maintaining the same file structure as the original libraries.

4. Refactoring the Folder Structure

    The original codebase mixed content from Roach and PDM-Lite, with common functionality codes interspersed with algorithm-specific codes. The codebase has been refactored to separate common functionality from algorithm-specific codes, resulting in a more readable and maintainable structure.

## Quick start

### Download this Repository

```shell
# git clone this repository and its submodules
git clone --recursive git@github.com:SCP-CN-001/carla_autopilot.git
```

### Download Carla

```shell
# download these under any directory you like, please avoid to download them under this repository
git clone https://leaderboard-public-contents.s3.us-west-2.amazonaws.com/CARLA_Leaderboard_2.0.tar.xz
tar -xf CARLA_Leaderboard_2.0.tar.xz

# build soft links
cd path/to/carla_autopilot
ln -s CARLA_Leaderboard_20/CARLA_Leaderboard_20 ./
```

### Setup Python Environment

```shell
conda create -n autopilot python=3.8
conda activate autopilot
pip install -r requirements.txt
```

### Run Expert Agent

```shell
# start a terminal and run
cd CARLA_Leaderboard_20
./CarlaUE4.sh --world-port=2000

# start another terminal and run
cd scripts
chmod +x L20_data_control.sh
./L20_data_control.sh
```

## Configurations

The `data_agent.py` inherits from `expert_agent.py` to provide vehicle commands.

1. Customizing the Expert Agent
    - Refer to `src/configs/expert_agent.yaml` and adjust the parameters as needed.
    - Modify `path_agent_configs` in `src/configs/data_agent.yaml` to point to the updated configuration file.

2. Customizing the Sensors
    - Refer to `src/configs/sensors.yaml` and add or remove sensor configurations as needed.
    - Modify `path_sensor_configs` in `src/configs/data_agent.yaml` to point to the updated configuration file.

3. Customizing the Visualization
    - Refer to `src/configs/visualize_binary.yaml` and adjust the parameters as needed.
    - Modify `path_data_collection_configs` in `src/configs/data_agent.yaml` to point to the updated configuration file.

4. Customizing Data Collection
    - Refer to `src/configs/data_agent.yaml`.
    - Adjust the parameters as needed to specify the desired data collection settings.

5. Change the semantic labels
    - Refer to `src/configs/binary_palette.yaml` and `src/configs/cityscapes_palette.yaml` to create custom semantic labels.
    - Modify `save_semantic_bev.path_palette` or customized sensor configurations in `src/configs/sensors.yaml` to use the custom semantic labels.
