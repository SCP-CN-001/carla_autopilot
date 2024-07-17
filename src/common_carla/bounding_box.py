import carla
import numpy as np
from agents.tools.misc import get_trafficlight_trigger_location

from src.common_carla.actor import get_forward_speed
from src.common_carla.geometry import cross_product, dot_product
from src.common_carla.waypoint import get_next_waypoints
from src.utils.geometry import get_relative_position, get_transform_3D, normalize_angle


def get_close_actor_bbox(actor_info, actor_class, ego_vehicle):
    ego_yaw = np.deg2rad(ego_vehicle.get_transform().rotation.yaw)
    ego_matrix = np.array(ego_vehicle.get_transform().get_matrix())

    actor_transform = carla.Transform(actor_info[0].location, actor_info[0].rotation)
    actor_rotation = actor_transform.rotation
    actor_matrix = np.array(actor_transform.get_matrix())
    actor_extent = [
        actor_info[0].extent.x,
        actor_info[0].extent.y,
        actor_info[0].extent.z,
    ]
    actor_yaw = np.deg2rad(actor_rotation.yaw)

    relative_yaw = normalize_angle(actor_yaw - ego_yaw)
    relative_location = get_relative_position(ego_matrix, actor_matrix)

    distance = np.linalg.norm(relative_location)

    result = {
        "id": int(actor_info[1]),
        "class": actor_class,
        "extent": actor_extent,
        "location": [relative_location[0], relative_location[1], relative_location[2]],
        "yaw": relative_yaw,
        "distance": distance,
        "affects_ego": actor_info[2],
        "matrix": actor_transform.get_matrix(),
    }

    if actor_class == "traffic_light":
        result["state"] = actor_info[3]

    return result


def get_actor_bbox(
    world_map,
    actor,
    actor_class,
    ego_vehicle,
    ego_wp,
    ego_direction,
    lane_id_left_most_lane_same_direction,
    lane_id_right_most_lane_opposite_direction,
    ego_lane_number,
    lidar_points=None,
):
    """

    Can handle walker, static, static_car (static.prop.mesh), static.prop.trafficwarning, traffic_light_vqa, stop_sign_vqa

    Args:
        world_map (_type_): _description_
        actor (_type_): _description_
        actor_class (_type_): _description_
        ego_vehicle (_type_): _description_
        ego_wp (_type_): _description_
        ego_direction (_type_): _description_
        lane_id_left_most_lane_same_direction (_type_): _description_
        lane_id_right_most_lane_opposite_direction (_type_): _description_
        ego_lane_number (_type_): _description_
        lidar_points (_type_, optional): _description_. Defaults to None.
    """
    ego_yaw = np.deg2rad(ego_vehicle.get_transform().rotation.yaw)
    ego_matrix = np.array(ego_vehicle.get_transform().get_matrix())

    actor_transform = actor.get_transform()
    actor_rotation = actor_transform.rotation
    actor_matrix = np.array(actor_transform.get_matrix())
    actor_extent = actor.bounding_box.extent
    actor_extent = [actor_extent.x, actor_extent.y, actor_extent.z]
    yaw = np.deg2rad(actor_rotation.yaw)

    relative_yaw = normalize_angle(yaw - ego_yaw)
    relative_location = get_relative_position(ego_matrix, actor_matrix)
    # TODO: Handle the passed traffic lights

    if not lidar_points is None:
        num_points = count_points_in_bbox(
            lidar_points, relative_location, relative_yaw, actor_extent
        )
    else:
        num_points = -1

    distance = np.linalg.norm(relative_location)

    if actor_class != "landmark":
        if actor_class in ["traffic_light_vqa", "stop_sign_vqa"]:
            trigger = get_trafficlight_trigger_location(actor)
            actor_wp = world_map.get_waypoint(
                trigger, project_to_road=False, lane_type=carla.LaneType.Any
            )
        else:
            actor_wp = world_map.get_waypoint(
                actor.get_location(), lane_type=carla.LaneType.Any
            )
        same_road_as_ego = False
        same_direction_as_ego = False
        lane_relative_to_ego = None

        if actor_wp.road_id == ego_wp.road_id:
            same_road_as_ego = True
            direction = actor_wp.lane_id / abs(actor_wp.lane_id)
            if direction == ego_direction:
                same_direction_as_ego = True
            if same_direction_as_ego:
                lane_relative_to_ego = abs(
                    actor_wp.lane_id - lane_id_left_most_lane_same_direction
                )
            else:
                lane_relative_to_ego = (
                    -1
                    * abs(actor_wp.lane_id - lane_id_right_most_lane_opposite_direction)
                    - 1
                )
            lane_relative_to_ego = lane_relative_to_ego - ego_lane_number

    result = {
        "class": actor_class,
        "extent": actor_extent,
        "location": [
            relative_location[0],
            relative_location[1],
            relative_location[2],
        ],
        "yaw": relative_yaw,
        "distance": distance,
        "num_points": num_points,
    }

    if actor_class == "walker":
        result["id"] = actor.id
        actor_speed = get_forward_speed(actor)
        result["speed"] = actor_speed
        result["role_name"] = actor.role_name
        result["gender"] = actor.gender
        result["age"] = actor.age
    elif actor_class == "static_car":
        if "Car" in actor.attributes["mesh_path"]:
            result["road_id"] = actor_wp.road_id
            result["junction_id"] = actor_wp.junction_id
    elif actor_class == "traffic_light_vqa":
        result["road_id"] = actor_wp.road_id
        result["junction_id"] = actor_wp.junction_id
        result["state"] = int(actor.state)
    elif actor_class == "stop_sign_vqa":
        result["road_id"] = actor_wp.road_id
        result["junction_id"] = actor_wp.junction_id
    elif actor_class == "landmark":
        result["id"] = actor.id
        result["name"] = actor.name
        result["text"] = actor.text
        result["value"] = actor.value

    if actor_class != "landmark":
        result["lane_id"] = actor_wp.lane_id
        result["lane_type"] = actor_wp.lane_type
        result["same_road_as_ego"] = same_road_as_ego
        result["same_direction_as_ego"] = same_direction_as_ego
        result["lane_relative_to_ego"] = (lane_relative_to_ego,)
        result["matrix"] = actor_transform.get_matrix()

    return result


def get_separating_plane(
    relative_position: carla.Vector3D,
    plane_normal: carla.Vector3D,
    obb1: carla.BoundingBox,
    obb2: carla.BoundingBox,
):
    """Check if there is a separating plane between two oriented bounding boxes (OBBs).

    Args:
        relative_position (carla.Vector3D): The relative position between the two OBBs.
        plane_normal (carla.Vector3D): The normal vector of the plane.
        obb1 (carla.BoundingBox): The first oriented bounding box.
        obb2 (carla.BoundingBox): The second oriented bounding box.

    Returns:
        bool: True if there is a separating plane, False otherwise.
    """
    # Calculate the projection of the relative position onto the plane normal
    projection_distance = abs(dot_product(relative_position, plane_normal))

    # Calculate the sum of the projections of the OBB extents onto the plane normal
    obb1_projection = (
        abs(
            dot_product(
                obb1.rotation.get_forward_vector() * obb1.extent.x, plane_normal
            )
        )
        + abs(
            dot_product(obb1.rotation.get_right_vector() * obb1.extent.y, plane_normal)
        )
        + abs(dot_product(obb1.rotation.get_up_vector() * obb1.extent.z, plane_normal))
    )

    obb2_projection = (
        abs(
            dot_product(
                obb2.rotation.get_forward_vector() * obb2.extent.x, plane_normal
            )
        )
        + abs(
            dot_product(obb2.rotation.get_right_vector() * obb2.extent.y, plane_normal)
        )
        + abs(dot_product(obb2.rotation.get_up_vector() * obb2.extent.z, plane_normal))
    )

    # Check if the projection distance is greater than the sum of the OBB projections
    return projection_distance > obb1_projection + obb2_projection


def check_obb_intersection(obb1: carla.BoundingBox, obb2: carla.BoundingBox):
    """
    Check if two 3D oriented bounding boxes (OBBs) intersect.

    Args:
        obb1 (carla.BoundingBox): The first oriented bounding box.
        obb2 (carla.BoundingBox): The second oriented bounding box.

    Returns:
        bool: True if the two OBBs intersect, False otherwise.
    """
    relative_position = obb2.location - obb1.location
    rot1 = obb1.rotation
    rot2 = obb2.rotation

    # Check for separating planes along the axes of both OBBs
    if (
        get_separating_plane(relative_position, rot1.get_forward_vector(), obb1, obb2)
        or get_separating_plane(relative_position, rot1.get_right_vector(), obb1, obb2)
        or get_separating_plane(relative_position, rot1.get_up_vector(), obb1, obb2)
        or get_separating_plane(
            relative_position, rot2.get_forward_vector(), obb1, obb2
        )
        or get_separating_plane(relative_position, rot2.get_right_vector(), obb1, obb2)
        or get_separating_plane(relative_position, rot2.get_up_vector(), obb1, obb2)
    ):
        return False

    # Check for separating planes along the cross products of the axes of both OBBs
    if (
        get_separating_plane(
            relative_position,
            cross_product(rot1.get_forward_vector(), rot2.get_forward_vector()),
            obb1,
            obb2,
        )
        or get_separating_plane(
            relative_position,
            cross_product(rot1.get_forward_vector(), rot2.get_right_vector()),
            obb1,
            obb2,
        )
        or get_separating_plane(
            relative_position,
            cross_product(rot1.get_forward_vector(), rot2.get_up_vector()),
            obb1,
            obb2,
        )
        or get_separating_plane(
            relative_position,
            cross_product(rot1.get_right_vector(), rot2.get_forward_vector()),
            obb1,
            obb2,
        )
        or get_separating_plane(
            relative_position,
            cross_product(rot1.get_right_vector(), rot2.get_right_vector()),
            obb1,
            obb2,
        )
        or get_separating_plane(
            relative_position,
            cross_product(rot1.get_right_vector(), rot2.get_up_vector()),
            obb1,
            obb2,
        )
        or get_separating_plane(
            relative_position,
            cross_product(rot1.get_up_vector(), rot2.get_forward_vector()),
            obb1,
            obb2,
        )
        or get_separating_plane(
            relative_position,
            cross_product(rot1.get_up_vector(), rot2.get_right_vector()),
            obb1,
            obb2,
        )
        or get_separating_plane(
            relative_position,
            cross_product(rot1.get_up_vector(), rot2.get_up_vector()),
            obb1,
            obb2,
        )
    ):
        return False

    # If no separating plane is found, the OBBs intersect
    return True


def is_point_in_bbox(point, bbox_center, bbox_extent):
    """Check if a point is inside a 2D bounding box.

    Args:
        point (_type_): _description_
        bbox_center (_type_): _description_
        bbox_extent (_type_): _description_
    """
    # bugfix slim bbox
    bbox_extent.x = max(bbox_extent.x, bbox_extent.y)
    bbox_extent.y = max(bbox_extent.x, bbox_extent.y)

    A = carla.Vector2D(bbox_center.x - bbox_extent.x, bbox_center.y - bbox_extent.y)
    B = carla.Vector2D(bbox_center.x + bbox_extent.x, bbox_center.y - bbox_extent.y)
    D = carla.Vector2D(bbox_center.x - bbox_extent.x, bbox_center.y + bbox_extent.y)
    M = carla.Vector2D(point.x, point.y)

    AB = B - A
    AD = D - A
    AM = M - A
    am_ab = dot_product(AM, AB, carla.Vector2D)
    ab_ab = dot_product(AB, AB, carla.Vector2D)
    am_ad = dot_product(AM, AD, carla.Vector2D)
    ad_ad = dot_product(AD, AD, carla.Vector2D)

    return 0 <= am_ab <= ab_ab and 0 <= am_ad <= ad_ad


def count_points_in_bbox(points, location, direction, extent):
    """Count the number of points inside a 3D bounding box.

    Args:
        points (np.ndarray): 3D points (N,3)
        location (np.ndarray): Location of the center of the bounding box.
        direction (float): Direction of the bounding box in radians.
        extent (list): Extent of the bounding box in meters.

    Returns:
        int: The number of points inside the bounding box.
    """
    points = get_transform_3D(points, location, direction)

    # check points in bbox
    x, y, z = extent[0], extent[1], extent[2]
    num_points = (
        (points[:, 0] < x)
        & (points[:, 0] > -x)
        & (points[:, 1] < y)
        & (points[:, 1] > -y)
        & (points[:, 2] < z)
        & (points[:, 2] > -z)
    ).sum()
    return num_points
