from typing import Optional, Tuple, List
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import trimesh
import rerun as rr


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
    
    # @staticmethod
    # def add_np_pointcloud(
    #     name: str, points: np.ndarray, colors_uint8: np.ndarray = None, radii: float = None
    # ):
    #     rr_points = rr.Points3D(positions=points, colors=colors_uint8, radii=radii)
    #     rr.log(name, rr_points)
    #     return


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