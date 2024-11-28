import os
import trimesh as tri
from utils.file_parser import FileParser
import numpy as np
import open3d as o3d
import random as rand
import urdfpy

def load_urdf_meshes(obj, urdf_file_path, scale, joint_state, only_articulated=False):
    if only_articulated == True:
        mesh_dict, _ = obj.get_articulated_meshes()
    if only_articulated == False:
        mesh_dict, _ = obj.get_all_meshes()
    # Load the URDF robot model
    arti_obj = urdfpy.URDF.load(urdf_file_path)
    joint_name = arti_obj.actuated_joints[0].name
    transformed_meshes = []
    corres_meshes = []
    for _, value in mesh_dict.items():
        for filename in value:
            corres_meshes.append(filename)

    rx = np.deg2rad(-90)  # Rotate 90 degrees around X-axis
    ry = np.deg2rad(90)  # Rotate 90 degrees around Y-axis

    # Create the corrective rotation matrices
    rotation_matrix_x = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])
    rotation_matrix_y = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])

    # Combine the corrective rotations
    combined_rotation_matrix = rotation_matrix_y @ rotation_matrix_x
    internal_transformed = np.eye(4)
    internal_transformed[:3, :3] = combined_rotation_matrix
    for mesh, pose in arti_obj.visual_trimesh_fk({joint_name: joint_state}).items():
        try: 
            mesh_name = 'textured_objs/'+mesh.metadata['file_name']
        except:
            mesh_name = 'textured_objs/'+mesh.metadata['name']
        if mesh_name in corres_meshes:
            scale_mat = np.eye(4) * scale
            scale_mat[3, 3] = 1.0
            mesh.apply_transform(scale_mat @ internal_transformed @ pose['pose'])  
            transformed_meshes.append(mesh)
    return transformed_meshes


def merge_meshes(meshes):
    return tri.util.concatenate(meshes)


def get_bounding_boxes(meshes):
    """
    Get bounding boxes for given meshes.

    Parameters:
    meshes: List of meshes for which to get bounding boxes.

    Returns:
    bounding_boxes: List of bounding boxes.
    """
    bounding_boxes = []
    for mesh in meshes:
        mesh = trimesh_to_o3d(mesh)
        bounding_box = mesh.get_axis_aligned_bounding_box()
        bounding_boxes.append(bounding_box)
    return bounding_boxes


def trimesh_to_o3d(mesh):
    """
    Convert a trimesh object to an open3d object.

    Parameters:
    mesh: Trimesh object to convert.

    Returns:
    mesh_o3d: Converted Open3D object.
    """

    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh.faces)
    return mesh_o3d


def fibonacci_sphere(radius, center, samples = 100):
    """
    Generate points on a sphere using the Fibonacci lattice method.

    Parameters:
    radius: Radius of the sphere.
    center: Center point of the sphere.
    samples: Number of points to generate.

    Returns:
    points: List of points on the sphere.
    """
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples)) * 2  # y goes from 1 to -1
        radius_y = np.sqrt(1 - y*y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius_y
        z = np.sin(theta) * radius_y

        points.append([radius * x + center[0], radius * y + center[1], radius * z + center[2]])

    return points

def subdivide_mesh(mesh, max_triangle_area):
    """
    Subdivide a mesh until all its triangles have an area smaller than max_triangle_area.

    Parameters:
    mesh: Mesh to subdivide.
    max_triangle_area: The maximum area a triangle is allowed to have.

    Returns:
    subdivided_mesh: Subdivided mesh.
    """
    while np.max(mesh.area_faces) > max_triangle_area:
        faces_to_subdivide = [i for i in range(len(mesh.faces)) if mesh.area_faces[i] > max_triangle_area]
        if not faces_to_subdivide:
            break  # Exit the loop if no faces left to subdivide
        mesh = mesh.subdivide(faces_to_subdivide)
    return mesh


def get_balanced_pointcloud(data_dir, obj_id, scale, joint_state, number_of_points):
    """
    Generate a balanced point cloud for an object.

    Parameters:
    obj_id: Object ID.
    number_of_points: Number of points in the cloud.

    Returns:
    pcd: Point cloud with balanced distribution of points.
    """
    samp = Sampling(data_dir, obj_id, scale, joint_state, 100000)
    pc = samp.create_balanced_cloud()
    combined_points = pc.points
    combined_normals = pc.normals
    for _ in range((number_of_points // 100000)-1): #this is done because the sampling time is exponential so we just always run it for 100k and then add the points
        samp = Sampling(data_dir, obj_id, scale, joint_state, 100000)
        pc = samp.create_balanced_cloud()
        combined_points = np.vstack((combined_points, pc.points))
        combined_normals = np.vstack((combined_normals, pc.normals))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(combined_points)
    pcd.normals = o3d.utility.Vector3dVector(combined_normals)
    return pcd
    

class Sampling:
    def __init__(self, data_dir, object_number, scale, joint_state, number_of_points = 100000):
        """
        Initialize the Sampling object.
        
        Parameters:
        object_number: Identifier for the object to be sampled.
        number_of_points: Number of points for the point cloud.
        """
        self.number_of_points = number_of_points
        self.object_number = object_number
        self.data_dir = data_dir
        self.scale = scale
        self.joint_state = joint_state
        self.obj = FileParser(self.data_dir, self.object_number)
        self.urdf_file_path = os.path.join(self.data_dir, str(self.object_number), "mobility.urdf")
        self.mesh_tri = merge_meshes(load_urdf_meshes(self.obj, self.urdf_file_path, self.scale, self.joint_state, only_articulated = False))
        self.mesh_tri = subdivide_mesh(self.mesh_tri, 0.01)

        self.mesh = trimesh_to_o3d(self.mesh_tri)
        viewpoints = self.get_viewpoints(500)

        triangles = self.get_visible_triangles(viewpoints)
        self.visable_mesh = self.mesh_from_triangle_index(triangles)

        self.visable_mesh.normalize_normals()
        self.pcd_cuda = self.visable_mesh.sample_points_uniformly(number_of_points, use_triangle_normal=True)
        self.pcd = self.pcd_cuda
        self.correct_normals()
        self.pcd_tree = o3d.geometry.KDTreeFlann(self.pcd_cuda)

    def get_nearest_points(self, number_of_nearest_points,radius, point):
        """
        Get the nearest points within a given radius around a point.
        
        Parameters:
        number_of_nearest_points: The number of closest points to find.
        radius: Search radius.
        point: Reference point.
        
        Returns:
        List of indices of nearest points.
        """
        print("point", point)
        [k, idx, _] = self.pcd_tree.search_radius_vector_3d(self.pcd_cuda.points[point], radius)
        if len(idx) < number_of_nearest_points:
            return list(idx)
        return rand.sample(list(idx), number_of_nearest_points-1)
    
    def paint(self, points, color):
        """
        Paint specified points in the point cloud with a given color.
        
        Parameters:
        points: List of point indices to paint.
        color: The color to paint the points.
        """
        np.asarray(self.pcd_legacy.colors)[list(points), :] = color
    
    def is_potentially_grippable(self, point, points):
        """
        Check if a point is potentially grippable based on its neighboring points.
        
        Parameters:
        point: The index of the point to check.
        points: List of indices of neighboring points.
        
        Returns:
        Boolean indicating if the point is potentially grippable.
        """

        # Get the normals for all points
        normals = [np.asarray(self.pcd_cuda.normals)[idx] for idx in points]
        normal = np.asarray(self.pcd_cuda.normals)[point]
        # Normalize each normal
        for i in range(len(normals)):
            normals[i] /= np.linalg.norm(normals[i])

        # Check each pair of normals
        for i in range(len(normals)):
            # Compute the angle between the pair of normals
            angle = np.degrees(np.arccos(np.clip(np.dot(normals[i], normal), -1.0, 1.0)))

            # If the angle is 90 degrees or more, the point is on an edge
            if angle >= 60 and angle <= 120:  # between 60 and 120 degrees
                for j in range(len(normals)):
                    angle = np.degrees(np.arccos(np.clip(np.dot(normals[i], normals[j]), -1.0, 1.0)))
                    if angle >= 150: # more than 150 degrees
                        return True


        return False
            
    def get_edges(self, point_index):
        """
        Get the edges of the mesh based on the point cloud.
        
        Parameters:
        point_index: List of point indices to consider.
        
        Returns:
        Unique set of edge points.
        """
        edges = []
        edges_unique = set()
        for index in point_index:
            if index not in edges:
                neighbors = self.get_nearest_points(6, 0.04, index)
                
                if self.is_potentially_grippable(index, neighbors):

                    edge = [index] 
                    edges.append(edge)
                    edges_unique.update(edge)
        return edges_unique
    
    def get_points_in_boxes(self, bounding_boxes: list, points: list):
        """
        Get the points that are within specified bounding boxes.
        
        Parameters:
        bounding_boxes: List of bounding boxes.
        points: List of points to consider.
        
        Returns:
        Set of points within the bounding boxes.
        """
        points = set()
        for bounding_box in bounding_boxes:

            points.update(bounding_box.get_point_indices_within_bounding_box(self.pcd_cuda.points))
        return points

    def get_articulated_points(self):
        """
        Get the points that are part of articulated components.
        
        Returns:
        Set of articulated points.
        """
        boxes = self.get_articulated_bounding_boxes()
        # print("boxes", boxes)
        return self.get_points_in_boxes(boxes, self.pcd_cuda.points)
    
    def get_articulated_bounding_boxes(self):
        """
        Get the bounding boxes of articulated components.
        
        Returns:
        List of bounding boxes for articulated components.
        """
        meshes = load_urdf_meshes(self.obj, self.urdf_file_path, self.scale, self.joint_state, only_articulated = True)
        return get_bounding_boxes(meshes)

    def bounding_sphere(self):
        """
        Calculate the bounding sphere of a mesh.

        Returns:
        The center and radius of the bounding sphere.
        """

        center = self.mesh_tri.bounding_sphere.primitive.center
        radius = self.mesh_tri.bounding_sphere.primitive.radius
        return center, radius*1.3

    def get_visible_triangles(self, viewpoints, batchsize=10000):
        """
        Get the visible triangles from multiple viewpoints.
        
        Parameters:
        viewpoints: A list of points representing the viewpoints.
        batchsize: The number of triangles to process in each batch.
        
        Returns:
        A set of visible triangle indices.
        """
        # Convert the viewpoints to a numpy array if they're not already
        viewpoints = np.array(viewpoints)
        # Create a scene and add the triangle mesh
        scene = o3d.t.geometry.RaycastingScene()
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh)
        scene.add_triangles(mesh)
        # Calculate the centroids of the triangles
        vertices = np.asarray(self.mesh.vertices)
        triangles = np.asarray(self.mesh.triangles)
        centroids = vertices[triangles].mean(axis=1)
        remaining_triangles = np.arange(len(centroids))  # Start with all triangles remaining
        visible_triangles = set()
        for viewpoint in viewpoints:
            hit_triangles = set()
            # Process remaining triangles in batches
            for i in range(0, len(remaining_triangles), batchsize):
                batch_indices = remaining_triangles[i:i+batchsize]  # Indices of the triangles in this batch
                batch_centroids = centroids[batch_indices]  # Centroids of the triangles in this batch
                # Get the directions from the viewpoint to each centroid in the batch
                directions = batch_centroids - viewpoint
                # Normalize the directions
                directions /= np.linalg.norm(directions, axis=1)[:, np.newaxis]
                # Concatenate origins and directions to create rays
                origins = np.tile(viewpoint, (len(batch_centroids), 1))
                rays = np.hstack((origins, directions)).astype(np.float32)
                rays = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
                # Perform ray casting
                ans = scene.cast_rays(rays)
                # Check the results
                indices_tri = ans['primitive_ids'].cpu().numpy()
                indices_tri = indices_tri[indices_tri != -1] 
                #indices_tri_np = indices_tri.cpu().numpy()
                hit_triangles.update(indices_tri)
            # Add hit triangles from all batches to the visible set
            visible_triangles.update(hit_triangles)
            # Update remaining triangles after all batches have been processed
            remaining_triangles = [tri for tri in remaining_triangles if tri not in hit_triangles]
        return visible_triangles


    def mesh_from_triangle_index(self, triangle_index):
        """
        Create a new mesh containing only the triangles specified by their indices.
        
        Parameters:
        triangle_index: A set or list of triangle indices.
        
        Returns:
        A new mesh containing only the specified triangles.
        """
        indices = np.array(list(triangle_index))
        indices = indices[indices != 4294967295] # invalid id tag

        new_mesh = o3d.geometry.TriangleMesh()
        new_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(self.mesh.vertices))
        new_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(self.mesh.triangles)[indices])
        return new_mesh
    

    def get_viewpoints(self, number_of_viewpoints):
        """
        Generate a set of viewpoints around the object.
        
        Parameters:
        number_of_viewpoints: The number of viewpoints to generate.
        
        Returns:
        A list of generated viewpoints.
        """
        center, radius = self.bounding_sphere()
        viewpoints = fibonacci_sphere(radius=radius, center=center, samples = number_of_viewpoints)
        return viewpoints


    def create_balanced_cloud(self): 
        """
        Create a balanced point cloud based on the ratio of articulated to unarticulated points.
        Returns:
        A new point cloud containing the balanced set of points.
        """
        articulated_points = self.get_articulated_points()
        antipodal_points = self.get_edges(articulated_points)
        selected_unarticulated = []
        selected_articulated = []
        # Combine the selected points into a new point cloud
        new_points = np.concatenate([selected_unarticulated, selected_articulated, list(antipodal_points)])
        new_points = list(map(int, new_points))
        new_point_cloud = o3d.geometry.PointCloud()
        new_points_positions = np.asarray(self.pcd_cuda.points)[new_points]
        new_points_normals = np.asarray(self.pcd_cuda.normals)[new_points]
        new_point_cloud.points = o3d.utility.Vector3dVector(new_points_positions)
        new_point_cloud.normals = o3d.utility.Vector3dVector(new_points_normals)
        return new_point_cloud


    def correct_normals(self):
        """
        Correcting by shooting a ray from the current point and normal and if there are an even number of intersections the normal is flipped.
        """
        # Get the points and normals from the pointcloud
        points = np.asarray(self.pcd_cuda.points)
        normals = np.asarray(self.pcd_cuda.normals)

        # Create a Trimesh point cloud from the points
        mesh = self.mesh_tri

        # Compute the intersections
        _, index_ray, _ = mesh.ray.intersects_location(
            ray_origins=points,
            ray_directions=normals,
            multiple_hits=True
        )
        unique, counts = np.unique(index_ray, return_counts=True)
        odd_indices = unique[counts % 2 == 0]
        normals[odd_indices] = -normals[odd_indices]
        self.pcd_cuda.normals = o3d.utility.Vector3dVector(normals)
