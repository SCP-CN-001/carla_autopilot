import carla
from shapely.geometry import Polygon
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


def get_route_polygon(planner, max_distance, offset=0.0) -> Polygon:
    list_points = []

    ego_vehicle = CarlaDataProvider.get_hero_actor()
    ego_transform = ego_vehicle.get_transform()
    ego_location = ego_transform.location

    extent_y = ego_vehicle.bounding_box.extent.y
    right_extent = extent_y + offset
    left_extent = -extent_y + offset
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
