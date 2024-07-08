import carla
from shapely.geometry import Polygon
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


def get_route_polygon(planner, max_distance) -> Polygon:
    list_points = []

    ego_vehicle = CarlaDataProvider.get_hero_actor()
    ego_transform = ego_vehicle.get_transform()
    ego_location = ego_transform.location

    extent_y = ego_vehicle.bounding_box.extent.y
    right_extent = extent_y
    left_extent = -extent_y
    right_vector = ego_transform.get_right_vector()

    right_location = carla.Location(
        right_extent * right_vector.x, right_extent * right_vector.y
    )
    left_location = carla.Location(
        left_extent * right_vector.x, left_extent * right_vector.y
    )
    pt1 = ego_location + right_location
    pt2 = ego_location + left_location

    list_points.append([[pt1.x, pt1.y, pt1.z], [pt2.x, pt2.y, pt2.z]])

    for waypoint, _ in planner.get_plan():
        if ego_location.distance(waypoint.transform.location) > max_distance:
            break

        right_vector = waypoint.transform.get_right_vector()
        pt1 = waypoint.transform.location + right_location
        pt2 = waypoint.transform.location + left_location
        list_points.append([[pt1.x, pt1.y, pt1.z], [pt2.x, pt2.y, pt2.z]])

    # Two points don't create a polygon, nothing to check
    if len(list_points) < 2:
        return None

    return Polygon(list_points)


def get_ego_waypoint():
    world = CarlaDataProvider.get_world()
    ego_vehicle = CarlaDataProvider.get_hero_actor()
    ego_waypoint = world.get_map().get_waypoint(
        ego_vehicle.get_location(),
        project_to_road=True,
        lane_type=carla.libcarla.LaneType.Any,
    )
    return ego_waypoint


def get_next_waypoints():
    """Get the waypoints from the current waypoint the ego vehicle is at to the end of the lane."""
    ego_waypoint = get_ego_waypoint()
    try:
        current_road_id = ego_waypoint.road_id
        current_lane_id = ego_waypoint.lane_id
        next_road_id = current_road_id
        next_lane_id = current_lane_id
        current_waypoint = [ego_waypoint]
        next_waypoints = []

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
