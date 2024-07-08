This directory is forked from branch `leaderboard_2` of the [autonomousvision/carla_garage](https://github.com/autonomousvision/carla_garage/tree/leaderboard_2), at commit ID 4d63da6.

I refactored the code to enhance readability and provide a more user-friendly interface for customizing sensors for data collection.

The main changes include:

1. Dissolving the `leaderboard` and `scenario_runner`: Many developers tend to pull the leaderboard and scenario runners and directly modify them. While this can be convenient during development, it often leads to confusion about which parts of the libraries have been changed and whether any unauthorized information (for the CARLA Leaderboard) has been used. Therefore, this change aims to clarify and streamline the modification process.
2. Refactoring the folder structure:

## Quick start

### Download Carla

```shell
# download these under any directory you like
git clone https://leaderboard-public-contents.s3.us-west-2.amazonaws.com/CARLA_Leaderboard_2.0.tar.xz
tar -xf CARLA_Leaderboard_2.0.tar.xz

# build soft links
cd path/to/carla_autopilot
ln -s CARLA_Leaderboard_20/CARLA_Leaderboard_20 ./
```

### Setup Python Environment

### Run Expert Agent

```shell
# start a terminal and run
cd CARLA_Leaderboard_20
./CarlaUE4.sh --world-port=2000

# start another terminal and run
cd scripts
./
```
