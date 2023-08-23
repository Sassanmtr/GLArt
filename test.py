import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d


def fibonacci_sphere(samples, radius, center):
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius_y = np.sqrt(1 - y*y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius_y
        z = np.sin(theta) * radius_y

        points.append([radius * x + center[0], radius * y + center[1], radius * z + center[2]])

    return points


def bounding_sphere(mesh_points):
    """
    Calculate the bounding sphere of a mesh.

    Parameters:
    mesh_points: A numpy array of points in the mesh.

    Returns:
    The center and radius of the bounding sphere.
    """
    center = np.mean(mesh_points, axis=0)
    radius = np.max(np.sqrt(np.sum((mesh_points - center)**2, axis=1)))
    return center, radius




# Usage:
# Assume that 'mesh' is a numpy array of your mesh points

bunny = o3d.data.BunnyMesh()
mesh = o3d.io.read_triangle_mesh(bunny.path)
mesh_points = np.asarray(mesh.vertices)

center, radius = bounding_sphere(mesh_points)

# Now you can generate your viewpoints on the bounding sphere
viewpoints = fibonacci_sphere(samples=100, radius=radius+0.1, center=center)





# Visualize
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter([p[0] for p in viewpoints], [p[1] for p in viewpoints], [p[2] for p in viewpoints])
ax.scatter(mesh_points[:, 0], mesh_points[:, 1], mesh_points[:, 2], color='b', alpha=0.1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
