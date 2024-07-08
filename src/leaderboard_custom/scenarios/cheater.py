##!/usr/bin/env python3
# @File: cheater.py
# @Description: A snippet to mark specific scenarios for the expert agent.
# @CreatedTime: 2024/07/08
# @Author: Yueyuan Li


class Cheater:
    """In PDM-Lite, the `CarlaDataProvider` was modified in place to store the `active_scenarios` list as a globally available variable. To improve the readability of the code, we created a `Cheater` class to avoid mixing customized code with the `srunner` library."""

    active_scenarios = []

    @staticmethod
    def cleanup():
        Cheater.active_scenarios = []

    @staticmethod
    def update(scenario_type, scenario_instance):
        """Detect and mark specific scenarios for the expert agent."""

        if scenario_type in ["ConstructionObstacle", "ConstructionObstacleTwoWays"]:
            traffic_warning = scenario_instance.other_actors[1]
            last_cone = scenario_instance.other_actors[-2]
            Cheater.active_scenarios.append(
                (
                    scenario_type,
                    [
                        traffic_warning,
                        last_cone,
                        scenario_instance._direction,
                        False,
                        1e9,
                        1e9,
                        False,
                    ],
                )
            )
        elif scenario_type == "InvadingTurn":
            first_cone = scenario_instance.other_actors[-1]
            last_cone = scenario_instance.other_actors[0]
            Cheater.active_scenarios.append(
                (
                    scenario_type,
                    [first_cone, last_cone, scenario_instance._true_offset],
                )
            )
        elif scenario_type in ["Accident", "HazardAtSideLane"]:
            Cheater.active_scenarios.append(
                (
                    scenario_type,
                    [
                        scenario_instance.other_actors[-2],
                        scenario_instance.other_actors[-1],
                        scenario_instance._direction,
                        False,
                        1e9,
                        1e9,
                        False,
                    ],
                )
            )
        elif scenario_type in [
            "ParkedObstacle",
            "ParkedObstacleTwoWays",
            "VehicleOpensDoorTwoWays",
            "YieldToEmergencyVehicle",
        ]:
            Cheater.active_scenarios.append(
                (
                    scenario_type,
                    [
                        scenario_instance.other_actors[-1],
                        None,
                        scenario_instance._direction,
                        False,
                        1e9,
                        1e9,
                        False,
                    ],
                )
            )
