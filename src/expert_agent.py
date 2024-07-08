##!/usr/bin/env python3
# @File: expert_agent.py
# @Description: An expert agent for data collection in CARLA Leaderboard 2.0.
# @CreatedTime: 2024/07/08
# @Author: Yueyuan Li


from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from src.leaderboard_custom.autoagents.autonomous_agent import AutonomousAgent


def get_entry_point():
    return "ExpertAgent"


class ExpertAgent(AutonomousAgent):
    """An expert agent for data collection in CARLA Leaderboard 2.0. This agent has access to the ground truth in the simulator directly.

    Adopted from:
    """

    def _init(self):
        self.ego_vehicle = CarlaDataProvider.get_hero_actor()
        self.world = CarlaDataProvider.get_world()

        # Check if the vehicle starts from a parking spot
