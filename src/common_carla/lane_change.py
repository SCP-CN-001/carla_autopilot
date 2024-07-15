from agents.navigation.local_planner import RoadOption


def is_near_lane_change(
    vehicle,
    route_points,
    route_index,
    commands,
    safety_distance=10.0,
    minimum_lookahead_points=200,
    previous_points=150,
    points_per_meter=10,
):
    """Check if the vehicle is near a lane change maneuver.

    Args:
        vehicle (_type_): _description_
        route_points (np.ndarray): An array of locations representing the planned route.
        safety_distance (float): Safety distance to be added to emergency braking distance.
        minimum_lookahead_points (int): Minimum number of points to look ahead when checking for lane change.
        previous_points (int): Number of previous points to consider when checking for lane change.
        points_per_meter (int): Points sampled per meter when interpolating route.

    Returns:
        bool: True if the vehicle is near a lane change maneuver, False otherwise.
    """
    vehicle_velocity = vehicle.get_velocity().length()

    # Calculate the braking distance based on the ego velocity
    braking_distance = (((vehicle_velocity * 3.6) / 10.0) ** 2 / 2.0) + safety_distance

    # Determine the number of waypoints to look ahead based on the braking distance
    look_ahead_points = max(
        minimum_lookahead_points,
        min(route_points.shape[0], points_per_meter * int(braking_distance)),
    )

    from_index = max(0, route_index - previous_points)
    to_index = min(len(commands) - 1, route_index + look_ahead_points)
    # Iterate over the points around the current position, checking for lane change commands
    for i in range(from_index, to_index, 1):
        if commands[i] in (
            RoadOption.CHANGELANELEFT,
            RoadOption.CHANGELANERIGHT,
        ):
            return True

    return False
