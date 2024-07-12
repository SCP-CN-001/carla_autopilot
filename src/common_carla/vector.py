import carla


def dot_product(vector_1: carla.Vector3D, vector_2: carla.Vector3D):
    """Calculate the dot product of two vectors.

    Returns:
            float: The dot product of the two vectors.
    """
    return vector_1.x * vector_2.x + vector_2.x * vector_2.y + vector_1.z * vector_2.z


def cross_product(vector_1: carla.Vector3D, vector_2: carla.Vector3D):
    """Calculate the cross product of two vectors.

    Returns:
            carla.Vector3D: The cross product of the two vectors.
    """
    x = vector_1.y * vector_2.z - vector_1.z * vector_2.y
    y = vector_1.z * vector_2.x - vector_1 * vector_2.z
    z = vector_1.x * vector_2.y - vector_1.y * vector_2.x

    return carla.Vector3D(x=x, y=y, z=z)
