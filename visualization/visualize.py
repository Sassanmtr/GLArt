import mesh_to_sdf
import urdfpy
from typing import Optional, Tuple, List
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import trimesh
import rerun as rr
import open3d as o3d
import torch
from torch.autograd import Variable

def color_pcd(pcd_np: np.ndarray, color: Optional[Tuple] = None):
    """
    Args:
        pcd: numpy array of shape (N, 3)
    """
    if not color:
        min_z = pcd_np[:, 2].min()
        max_z = pcd_np[:, 2].max()
        cmap_norm = mpl.colors.Normalize(vmin=min_z, vmax=max_z)
        #'hsv' is changeable to any name as stated here: https://matplotlib.org/stable/tutorials/colors/colormaps.html
        point_colors = plt.get_cmap("jet")(cmap_norm(pcd_np[:, 2]))[
            :, :3
        ]  # We don't need alpha
    else:
        assert len(color) == 3
        N, _ = pcd_np.shape
        point_colors = np.tile(color, (N, 1))
    return point_colors


class Grid3D:
    def __init__(
        self, density=30, grid_dim: float = 1.0, device="cpu", precision=torch.float32
    ):
        self.points = Variable(
            self.generate_point_grid(density, grid_dim).to(device, precision),
            requires_grad=True,
        )

    def generate_point_grid(self, grid_density, grid_dim):
        """
        Initial 3D point grid generation

        Args:
            grid_density (int): grid point density

        Returns: 3D point grid

        """
        # Set up the grid
        grid_density_complex = grid_density * 1j
        X, Y, Z = np.mgrid[
            -grid_dim:grid_dim:grid_density_complex,
            -grid_dim:grid_dim:grid_density_complex,
            -grid_dim:grid_dim:grid_density_complex,
        ]
        grid_np = np.concatenate(
            (X[..., None], Y[..., None], Z[..., None]), axis=-1
        ).reshape((-1, 3))

        # Make an offset for every second z grid plane
        grid_np[1::2, :2] += (X.max() - X.min()) / grid_density / 2
        grid = torch.from_numpy(grid_np.astype(np.float32))
        return grid

class RerunViewer:
    def __init__(self, addr: str = None):
        rr.init("GraspSampler")
        if addr is None:
            rr.spawn()
        else:
            rr.connect(addr)
        RerunViewer.clear()
        return

    @staticmethod
    def add_pointcloud(name, points, colors=None, radii=None):
        rr_points = rr.Points3D(positions=points, colors=colors, radii=radii)
        rr.log(name, rr_points)

        return

    @staticmethod
    def add_mesh_trimesh(name: str, mesh: trimesh.Trimesh):
        rr_mesh = rr.Mesh3D(
            vertex_positions=mesh.vertices,
            vertex_colors=mesh.visual.vertex_colors,
            vertex_normals=mesh.vertex_normals,
            indices=mesh.faces,
        )
        rr.log(name, rr_mesh)
        return
    
    @staticmethod
    def add_trimeshes(name: str, meshes: List[trimesh.Trimesh]):
        for i, mesh in enumerate(meshes):
            RerunViewer.add_mesh_trimesh(name + f"/{i}", mesh)
        return


    @staticmethod
    def add_grasps(name, grasp_poses, color=[0.0, 1.0, 0.0]):
        grasps_trimesh = create_markers_multiple(
            grasp_poses, color, axis_frame=True, highlight_first=True
        )
        RerunViewer.add_trimeshes(name, grasps_trimesh)
        return


    @staticmethod
    def add_axis(name, pose, size=0.04):
        mesh = trimesh.creation.axis(origin_size=size, transform=pose)
        RerunViewer.add_trimeshes(name, [mesh])
        return

    @staticmethod
    def add_grid_bounding_box(name, grid_dim):
        half_size = [
            grid_dim,
            grid_dim,
            grid_dim,
        ]  # This version of rerun has a bug with half_size
        rr.log_obb(
            name, half_size=half_size, position=[0, 0, 0], rotation_q=[0, 0, 0, 1]
        )
        return

    @staticmethod
    def add_sphere():
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=100.0)
        sphere.paint_uniform_color([1.0, 1.0, 1.0])
        rr.log(
            "vis/background",
            rr.Mesh3D(
                vertex_positions=sphere.vertices,
                indices=sphere.triangles,
                vertex_colors=sphere.vertex_colors,
            )
        )

    @staticmethod
    def add_cube():
        sphere = o3d.geometry.TriangleMesh.create_box()
        sphere.paint_uniform_color([0.0, 0.0, 0.0])
        rr.log(
            "vis/cube",
            rr.Mesh3D(
                vertex_positions=sphere.vertices,
                indices=sphere.triangles,
                vertex_colors=sphere.vertex_colors,
            )
        )

    @staticmethod
    def add_grid_box():
        grid_size = 10
        positions = []
        colors = []
        radii = []
        from numpy.random import default_rng
        rng = default_rng(12345)
        # Generate points in a grid
        for x in range(grid_size):
            for y in range(grid_size):
                for z in range(grid_size):
                    # Calculate position for each grid point
                    position = [(x - grid_size/2) * 0.1, (y - grid_size/2) * 0.1, (z - grid_size/2) * 0.1]
                    positions.append(position)

                    # Assign a random color
                    colors.append(rng.uniform(0, 255, size=[1, 3]))  # For example, white color

                    # Set a fixed radius for each point
                    radii.append(0.01)
        rr.log("vis/grid_box", rr.Points3D(positions, colors=colors, radii=radii))

    @staticmethod
    def clear():
        rr.log("vis", rr.Clear(recursive=True))
        return

    

def create_gripper_marker(
    color=[0, 0, 255], gripper_width=0.08, tube_radius=0.002, sections=6
):
    """Create a 3D mesh visualizing a parallel yaw gripper. It consists of four cylinders.

    Args:
        color (list, optional): RGB values of marker. Defaults to [0, 0, 255].
        tube_radius (float, optional): Radius of cylinders. Defaults to 0.001.
        sections (int, optional): Number of sections of each cylinder. Defaults to 6.

    Returns:
        trimesh.Trimesh: A mesh that represents a simple parallel yaw gripper.
    """
    left_y = -0.5 * (gripper_width + tube_radius)
    right_y = 0.5 * (gripper_width + tube_radius)
    mid_z = 0.066
    top_z = 0.112
    cfr = trimesh.creation.cylinder(
        radius=tube_radius,
        sections=sections,
        segment=[
            [0.0, right_y, mid_z],
            [0.0, right_y, top_z],
        ],
    )
    cfl = trimesh.creation.cylinder(
        radius=tube_radius,
        sections=sections,
        segment=[
            [0.0, left_y, mid_z],
            [0.0, left_y, top_z],
        ],
    )
    cb1 = trimesh.creation.cylinder(
        radius=tube_radius, sections=sections, segment=[[0, 0, 0], [0, 0, mid_z]]
    )
    cb2 = trimesh.creation.cylinder(
        radius=tube_radius,
        sections=sections,
        segment=[[0.0, left_y, mid_z], [0.0, right_y, mid_z]],
    )

    tmp = trimesh.util.concatenate([cb1, cb2, cfr, cfl])
    tmp.visual.face_colors = color

    return tmp

def create_markers(transform, color, tube_radius=0.002, axis_frame: bool = True):
    original_transform = transform.copy()
    # original_transform[:3, 3] -= 0.075 * transform[:3, 2]
    original_transform[:3, 3] -= 0.15 * transform[:3, 2]
    # original_transform[:3, 3] -= 0.9034* transform[:3, 2]
    position_marker = trimesh.creation.axis(transform=transform, origin_size=0.002)  # type: ignore
    stick_marker = create_gripper_marker(color, tube_radius=tube_radius)  # type: ignore
    stick_marker.apply_transform(original_transform)  # type: ignore
    return (
        np.array([stick_marker] + ([position_marker] if axis_frame else []))
        .flatten()
        .tolist()
    )


def create_markers_multiple(
    transforms, color, axis_frame: bool = True, highlight_first: bool = False
):
    res = (
        np.array([create_markers(t, color, axis_frame=axis_frame) for t in transforms])
        .flatten()
        .tolist()
    )
    if highlight_first and len(transforms) > 0:
        first_marker = create_markers(
            transforms[0], color, tube_radius=0.006, axis_frame=axis_frame
        )
        res[0] = first_marker[0]
    return res


def object_to_trimesh(urdf_file_path, scale, joint_state):
    # Load the URDF robot model
    arti_obj = urdfpy.URDF.load(urdf_file_path)
    joint_name = arti_obj.actuated_joints[0].name
    fk = arti_obj.visual_trimesh_fk({joint_name: joint_state})
    trimesh_scene = trimesh.Scene()
    for mesh, pose in fk.items():
        scale_mat = np.eye(4) * scale
        scale_mat[3, 3] = 1.0
        try:
            trimesh_scene.add_geometry(mesh, transform=scale_mat @ pose)
        except:
            trimesh_scene.add_geometry(mesh, transform=scale_mat @ pose["pose"])
        # trimesh_scene.add_geometry(mesh, transform=scale_mat @ internal_transformed @ pose)
    ## Uncomment for visualization
    # trimesh_scene.show()

    return trimesh_scene


def object_to_point_cloud(
    object_trimesh, number_samples: int = 200000
) -> Tuple[np.ndarray, np.ndarray]:
    mesh, transform = mesh_to_sdf.utils.scale_to_unit_sphere(
        object_trimesh, get_transform=True
    )
    surface_point_cloud = mesh_to_sdf.get_surface_point_cloud(
        mesh,
        sample_point_count=number_samples,
        surface_point_method="sample",  # [scan, sample] To allow inside of the mesh?
        calculate_normals=True,
    )
    points = surface_point_cloud.points * transform["scale"] + transform["translation"]
    normals = surface_point_cloud.normals
    return points, normals