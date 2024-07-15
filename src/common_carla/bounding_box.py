import carla
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from src.common_carla.geometry import cross_product, dot_product


def get_bboxes(lidar=None):
    ego_vehicle = CarlaDataProvider.get_hero_actor()
    world = CarlaDataProvider.get_world()

    ego_transform = ego_vehicle.get_transform()
    ego_rotation = ego_transform.rotation

    ego_control = ego_vehicle.get_control()
    ego_brake = ego_control.brake

    actors = world.get_actors()
    actor_list = actors.filter("*vehicle*")


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
    """_summary_

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
