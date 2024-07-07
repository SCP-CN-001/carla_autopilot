class Cheater:
    active_scenarios = []

    @staticmethod
    def cleanup():
        Cheater.active_scenarios = []

    @staticmethod
    def update(scenario_type, scenario_instance):
        print(scenario_type)
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
