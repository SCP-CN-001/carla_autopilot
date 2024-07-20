import carla


def get_ego_waypoint(world, ego_vehicle):
    ego_waypoint = world.get_map().get_waypoint(
        ego_vehicle.get_location(),
        project_to_road=True,
        lane_type=carla.libcarla.LaneType.Any,
    )
    return ego_waypoint


def get_next_waypoints(world, ego_vehicle):
    """Get the waypoints from the current waypoint the ego vehicle is at to the end of the lane."""
    ego_waypoint = get_ego_waypoint(world, ego_vehicle)
    try:
        current_road_id = ego_waypoint.road_id
        current_lane_id = ego_waypoint.lane_id
        next_road_id = current_road_id
        next_lane_id = current_lane_id
        current_waypoint = [ego_waypoint]
        next_waypoints = []

        # https://github.com/carla-simulator/carla/issues/2511#issuecomment-597230746
        while current_road_id == next_road_id and current_lane_id == next_lane_id:
            # Get a list of waypoints at a certain approximate distance.
            list_next_waypoints = current_waypoint[0].next(distance=1)
            if len(list_next_waypoints) == 0:
                break
            current_waypoint = list_next_waypoints
            next_waypoint = list_next_waypoints[0]
            next_waypoints.append(next_waypoint)
            next_road_id = next_waypoint.road_id
            next_lane_id = next_waypoint.lane_id
    except:
        next_waypoints = []

    return next_waypoints