import os
import trimesh as tri
from file_parser import FileParser
import numpy as np
import pyglet
import open3d as o3d
from helper import calculate_normal, get_angle_between
import random as rand
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import urdfpy
import trimesh

def test_urdf_meshes(obj, urdf_file_path, scale, joint_state, only_articulated=False):
    arti_obj = urdfpy.URDF.load(urdf_file_path)
    joint_name = arti_obj.actuated_joints[0].name
    transformed_meshes = []
    mesh_names = {}
    mesh_poses = {}
    # for mesh, pose in arti_obj.visual_geometry_fk({joint_name: joint_state}).items():
    #     mesh_names[mesh] = mesh.mesh.filename
    for mesh, pose in arti_obj.visual_trimesh_fk({joint_name: joint_state}).items():
        mesh_name = 'textured_objs/'+mesh.metadata['file_name']
        mesh_poses[mesh_name] = pose['pose']

    if only_articulated == True:
        mesh_dict, origin = obj.get_articulated_meshes()
    if only_articulated == False:
        mesh_dict, origin = obj.get_all_meshes()

    import pyrender
    scene = pyrender.Scene()
    for key, value in mesh_dict.items():
        for filename in value:
            # Load the mesh
            loaded = tri.load_mesh(obj.get_full_path(filename))
            
            # If the loaded object is a Scene, dump it to get a list of Trimesh objects
            meshes = loaded.dump() if isinstance(loaded, tri.Scene) else [loaded]
            
            for mesh in meshes:
                mesh.apply_scale(scale)
                mat = np.eye(4)
                mat[:3, 3] = origin
                mesh.apply_transform(np.dot(mat, mesh_poses[filename]))
                # mesh.apply_transform(mesh_poses[filename])
                # mesh.face_angles[:, 1] += joint_state
                
                transformed_meshes.append(mesh)
                mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
                scene.add(mesh)#, pose=pose['pose'])
    pyrender.Viewer(scene, use_raymond_lighting=True)
    return transformed_meshes



def load_urdf_meshes(obj, urdf_file_path, scale, joint_state, only_articulated=False):
    if only_articulated == True:
        mesh_dict, origin = obj.get_articulated_meshes()
    if only_articulated == False:
        mesh_dict, origin = obj.get_all_meshes()
    # Load the URDF robot model
    arti_obj = urdfpy.URDF.load(urdf_file_path)
    # Open the articulated joints to the desired joint state
    articulated_link = [joint.child for joint in arti_obj.actuated_joints]
    joint_name = arti_obj.actuated_joints[0].name
    # fk = arti_obj.link_fk({joint_name: joint_state})
    transformed_meshes = []
    actuated_meshes = arti_obj.visual_trimesh_fk(links=articulated_link).keys()
    # new added code
    corres_meshes = []
    for key, value in mesh_dict.items():
        for filename in value:
            corres_meshes.append(filename)

    # # Visualization
    # import pyrender
    # scene = pyrender.Scene()
    # # Define the length of the axes
    # axis_length = 1

    # # Define the colors for the axes (R, G, B)
    # x_color = [1.0, 0.0, 0.0]  # Red for X-axis
    # y_color = [0.0, 1.0, 0.0]  # Green for Y-axis
    # z_color = [0.0, 0.0, 1.0]  # Blue for Z-axis

    # num_points = 100
    # # Create points for lines along each axis
    # x_points = np.array([[i * axis_length / num_points, 0, 0] for i in range(num_points)])
    # y_points = np.array([[0, i * axis_length / num_points, 0] for i in range(num_points)])
    # z_points = np.array([[0, 0, i * axis_length / num_points] for i in range(num_points)])

    # # Create mesh objects for the lines
    # for i in range(num_points - 1):
    #     x_segment = pyrender.Mesh.from_points(x_points[i:i+2], colors=np.tile(x_color, (2, 1)))
    #     y_segment = pyrender.Mesh.from_points(y_points[i:i+2], colors=np.tile(y_color, (2, 1)))
    #     z_segment = pyrender.Mesh.from_points(z_points[i:i+2], colors=np.tile(z_color, (2, 1)))
        
    #     # Add the line segments to the scene
    #     scene.add(x_segment)
    #     scene.add(y_segment)
    #     scene.add(z_segment)

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
            # # mesh.apply_scale(scale)
            scale_mat = np.eye(4) * scale
            scale_mat[3, 3] = 1.0
            mesh.apply_transform(scale_mat @ internal_transformed @ pose['pose'])  
            transformed_meshes.append(mesh)
        # mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        # scene.add(mesh)#, pose=pose['pose'])
 
    # pyrender.Viewer(scene, use_raymond_lighting=True)

    return transformed_meshes

# urdf_file_path = '/home/mokhtars/Documents/Thesis/datasets/others/CatData/Microwave/7310/mobility.urdf'
# obj = FileParser("/home/mokhtars/Documents/articulatedobjectsgraspsampling/", "7221")
# urdf_file_path = '/home/mokhtars/Documents/articulatedobjectsgraspsampling/7221/mobility.urdf'
# load_urdf_meshes(obj, urdf_file_path=urdf_file_path, scale=0.5, joint_state=0.3, only_articulated=False)

def get_meshes(obj, scale, only_articulated = False):
    """
    Get meshes of an object.

    Parameters:
    obj: Object for which to get meshes.
    only_articulated: Boolean, if True returns only articulated meshes, otherwise returns all meshes.

    Returns:
    transformed_meshes: List of transformed meshes.
    """

    if only_articulated == True:
        mesh_dict, origin = obj.get_articulated_meshes()
    if only_articulated == False:
        mesh_dict, origin = obj.get_all_meshes()
    
    # articulated_links = [i for i in obj.get_articulated_meshes()[0].values()][0]

    transformed_meshes = []

    for key, value in mesh_dict.items():
        for filename in value:
            # Load the mesh
            loaded = tri.load_mesh(obj.get_full_path(filename))
            
            # If the loaded object is a Scene, dump it to get a list of Trimesh objects
            meshes = loaded.dump() if isinstance(loaded, tri.Scene) else [loaded]
            
            for mesh in meshes:
                mesh.apply_scale(scale)
                transformed_meshes.append(mesh)

    return transformed_meshes

def merge_meshes(meshes):
    """
    42
    """
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
    scaled_vertices = mesh.vertices * 0.3
    mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    # mesh_o3d.vertices = o3d.utility.Vector3dVector(scaled_vertices)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh.faces)
    # Create coordinate axes as lines with respective colors
    # axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])

    # Add the Open3D mesh and coordinate axes to a geometry list
    # geometries = [mesh_o3d, axes]
    # o3d.visualization.draw_geometries([mesh_o3d])
    # o3d.visualization.draw_geometries(geometries)
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

def get_inverse(points, selection):
    """
    Get the inverse set of points not in the selection.

    Parameters:
    points: Total points.
    selection: Selected points.

    Returns:
    List of points not in selection.
    """
    # Suppose `pcd` is your point cloud and `selected_points` is your list of points
    all_indices = set(np.arange(len(points)))
    return list(all_indices - set(selection))

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

def get_balanced_pointcloud(data_dir, obj_id, scale, joint_state, number_of_points, ratio):
    """
    Generate a balanced point cloud for an object.

    Parameters:
    obj_id: Object ID.
    number_of_points: Number of points in the cloud.
    ratio: Ratio for how many uniformally, movable points you want compared to the initial graspable points etc 500 graspable points detected and ratio 0.5,0.4 so the resulting pointcloud is 250 uniform points 200 movable points and it is assumed you always want all graspable so 500 graspable 

    Returns:
    pcd: Point cloud with balanced distribution of points.
    """
    samp = Sampling(data_dir, obj_id, scale, joint_state, 100000)
    pc = samp.create_balanced_cloud(ratio)
    combined_points = pc.points
    combined_normals = pc.normals
    for _ in range((number_of_points // 100000)-1): #this is done because the sampling time is exponential so we just always run it for 100k and then add the points

        samp = Sampling(data_dir, obj_id, scale, joint_state, 100000)
        pc = samp.create_balanced_cloud(ratio)
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
        # self.mesh_tri = merge_meshes(test_urdf_meshes(self.obj, self.urdf_file_path, self.scale, self.joint_state, only_articulated = False))
        # self.mesh_tri = merge_meshes(get_meshes(self.obj, self.scale, only_articulated = False))
        self.mesh_tri = subdivide_mesh(self.mesh_tri, 0.01)

        # print(len(self.mesh_tri.faces))

        self.mesh = trimesh_to_o3d(self.mesh_tri)
        viewpoints = self.get_viewpoints(500)

        triangles = self.get_visible_triangles(viewpoints)
        self.visable_mesh = self.mesh_from_triangle_index(triangles)
        # print("Number of triangles:", len(self.visable_mesh.triangles))
        #self.make_mesh_double_sided()

        self.visable_mesh.normalize_normals()

        self.pcd_cuda = self.visable_mesh.sample_points_uniformly(number_of_points, use_triangle_normal=True)
        self.pcd = self.pcd_cuda
        self.correct_normals()

        self.pcd_tree = o3d.geometry.KDTreeFlann(self.pcd_cuda)
        #self.pcd_legacy = o3d.geometry.PointCloud(self.pcd_cuda)
        #self.pcd = o3d.t.geometry.PointCloud.from_legacy(self.pcd_legacy)
        #self.pcd_legacy.paint_uniform_color([0, 1, 1])


# Usage:


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
        # print("point", point)
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
        # print(points)
        # print(len(points))
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
        # Assume initially that the surface is not an edge
        accessible = False
        antipodal = False

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
        # meshes = test_urdf_meshes(self.obj, self.urdf_file_path, self.scale, self.joint_state, only_articulated = True)
        # print("meshes", len(meshes))
        return get_bounding_boxes(meshes)
    
    def make_mesh_double_sided(self):
        """
        Make the mesh double-sided by reversing the order of vertices for each triangle, therefor making sure there is a triangle normal pointing in each direction.
        This was only for visualization since some triangles were not visualized due to this issue
        """
        # Create a new mesh and set its vertices and triangles to be the same as the original mesh
        mesh_double_sided = o3d.cuda.pybind.geometry.TriangleMesh()
        mesh_double_sided.vertices = self.mesh.vertices

        # Convert the triangles to a NumPy array, reverse the order of vertices, and convert it back to a Vector3iVector
        triangles_np = np.asarray(self.mesh.triangles)
        triangles_np_reversed = triangles_np[:, ::-1]
        mesh_double_sided.triangles = o3d.cuda.pybind.utility.Vector3iVector(triangles_np_reversed)

        # Combine the original mesh and the new mesh
        self.mesh.triangles.extend(mesh_double_sided.triangles)

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

        # Convert mesh to open3d format


        # Create a scene and add the triangle mesh
        scene = o3d.t.geometry.RaycastingScene()
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh)
        mesh_id = scene.add_triangles(mesh)

        # Calculate the centroids of the triangles
        #centroids = self.mesh_tri.triangles_center
        vertices = np.asarray(self.mesh.vertices)
        triangles = np.asarray(self.mesh.triangles)
        centroids = vertices[triangles].mean(axis=1)
        remaining_triangles = np.arange(len(centroids))  # Start with all triangles remaining
        visible_triangles = set()

        for viewpoint in viewpoints:
            hit_triangles = set()
            start = time.time()
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
                start_ray = time.time()
                ans = scene.cast_rays(rays)
                end_ray = time.time()

                # Check the results
                indices_tri = ans['primitive_ids'].cpu().numpy()
                indices_tri = indices_tri[indices_tri != -1] 



                #indices_tri_np = indices_tri.cpu().numpy()
                hit_triangles.update(indices_tri)
                
            # Add hit triangles from all batches to the visible set

            visible_triangles.update(hit_triangles)

            # Update remaining triangles after all batches have been processed
            remaining_triangles = [tri for tri in remaining_triangles if tri not in hit_triangles]

            end = time.time()
            viewpoint_time = end - start

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

    def mesh_from_triangle_index_tri(self, triangle_index):
        """
        Create a new Trimesh containing only the triangles specified by their indices.
        
        Parameters:
        triangle_index: A set or list of triangle indices.
        
        Returns:
        A new Trimesh containing only the specified triangles.
        """
        indices = list(triangle_index)
        new_mesh = tri.Trimesh(vertices=self.mesh_tri.vertices, faces=self.mesh_tri.faces[indices])
        return new_mesh

    def sample_visable_points(self):
        """
        Sample points that are visible from a set of viewpoints.
        
        Returns:
        A PointCloud object containing the sampled points.
        """
        center, radius = self.bounding_sphere()
        boxes = self.get_articulated_bounding_boxes()
        number_of_boxes = len(boxes)
        number_of_viewpoints = self.number_of_points // number_of_boxes
        viewpoints = fibonacci_sphere(radius=radius, center=center, samples = number_of_viewpoints)
        # print("the number of viewpoints are:", number_of_viewpoints)
        points = self.sample_points_from_viewpoints(viewpoints, boxes)
        points = np.array(points)
        pcd_cuda = o3d.geometry.PointCloud()
        pcd_cuda.points = o3d.utility.Vector3dVector(points)
        return pcd_cuda
       
    def visualize_mesh_and_bounding_boxes(self):
        """
        Visualize the mesh along with its articulated bounding boxes.
        """
        geometries = [self.mesh]
        bounding_boxes = self.get_articulated_bounding_boxes()
        for bbox in bounding_boxes:
            obb = o3d.geometry.OrientedBoundingBox.create_from_axis_aligned_bounding_box(bbox)
            geometries.append(obb)

        # o3d.visualization.draw_geometries(geometries)

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

    def create_balanced_cloud(self, ratio):             #the ratio stems from the antipodal points. It is assumed you want all antipodal points and then it samples based on the number of antipodal points
        #ratio relative to antipodal points
        # Get the points in each category
        """
        Create a balanced point cloud based on the ratio of articulated to unarticulated points.
        
        Parameters:
        ratio: Tuple containing the ratio of unarticulated to articulated points.
        
        Returns:
        A new point cloud containing the balanced set of points.
        """
        articulated_points = self.get_articulated_points()

        unarticulated_points = get_inverse(self.pcd_cuda.points, articulated_points)
        antipodal_points = self.get_edges(articulated_points)

        # Calculate the number of points to select from each category
        total_points = len(unarticulated_points) + len(articulated_points)
        num_antipodal = len(antipodal_points)

        # Calculate remaining points and ratios
        num_unarticulated = round(ratio[0] * num_antipodal)
        num_articulated = round(ratio[1] * num_antipodal)
        # Randomly select points from each category
        selected_unarticulated = random.sample(list(unarticulated_points), num_unarticulated)
        # if num_unarticulated > len(list(unarticulated_points)):
        #     selected_unarticulated = random.sample(list(unarticulated_points), len(list(unarticulated_points)))
        #     selected_articulated = random.sample(list(articulated_points), len(list(articulated_points)))
        # else:
        #     selected_unarticulated = random.sample(list(unarticulated_points), num_unarticulated)
        #     selected_articulated = random.sample(list(articulated_points), num_articulated)
        selected_articulated = random.sample(list(articulated_points), num_articulated)
        # print("selected articulated", len(selected_articulated))

        # Combine the selected points into a new point cloud
        new_points = np.concatenate([selected_unarticulated, selected_articulated, list(antipodal_points)])
        # print("new points", len(new_points), new_points)

        new_points = list(map(int, new_points))
        new_point_cloud = o3d.geometry.PointCloud()
        new_points_positions = np.asarray(self.pcd_cuda.points)[new_points]
        new_points_normals = np.asarray(self.pcd_cuda.normals)[new_points]
        new_point_cloud.points = o3d.utility.Vector3dVector(new_points_positions)
        new_point_cloud.normals = o3d.utility.Vector3dVector(new_points_normals)


        return new_point_cloud

    def correct_normals(self):
        """
        Correct the normals of the point cloud based on the underlying mesh.
        Correcting by shooting a ray from the current point and normal and if there are an even number of intersections the normal is flipped.
        """
        # Get the points and normals from the pointcloud
        start_time = time.time()
        points = np.asarray(self.pcd_cuda.points)
        normals = np.asarray(self.pcd_cuda.normals)

        # Create a Trimesh point cloud from the points
        mesh = self.mesh_tri

        # Compute the intersections
        start_time = time.time()
        locations, index_ray, index_tri = mesh.ray.intersects_location(
            ray_origins=points,
            ray_directions=normals,
            multiple_hits=True
        )
        unique, counts = np.unique(index_ray, return_counts=True)
        odd_indices = unique[counts % 2 == 0]
        normals[odd_indices] = -normals[odd_indices]
        self.pcd_cuda.normals = o3d.utility.Vector3dVector(normals)
        end_time = time.time()
        # print(f"Ray tracing took {end_time - start_time} seconds.")

    def check_collision(self, points):
        """
        Check for collisions of rays emanating from specified points.
        
        Parameters:
        points: List of points from which rays will be cast.
        """
        scene = o3d.t.geometry.RaycastingScene()
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh)
        position = np.asarray(self.pcd_cuda.points)[points]
        directions = np.asarray(self.pcd_cuda.normals)[points]
        origins = np.tile(position, (len(directions), 1))
        rays = np.hstack((origins, directions)).astype(np.float32)
        rays = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)

        # Perform ray casting

        ans = scene.cast_rays(rays)

        # print(ans.keys())
"""
start_time = time.time()





start_time = time.time()
samp = Sampling(7167, 100000)
#pc = samp.create_balanced_cloud([0, 0])
#o3d.visualization.draw_geometries([samp.pcd_legacy, samp.mesh], point_show_normal=False)
articulated = samp.get_articulated_points()
print("articulated", len(articulated))
#pc_articulated = o3d.geometry.PointCloud()
#pc_articulated = samp.pcd_cuda.select_by_index(list(articulated))
#pc_articulated.paint_uniform_color([0, 1, 1])
#o3d.visualization.draw_geometries([pc_articulated, samp.mesh], point_show_normal=False)
edges = samp.get_edges(articulated)
print("edges", len(edges))
end_time = time.time()
print(f"graspable sampling took {end_time - start_time} seconds.")
#pc_edges = o3d.geometry.PointCloud()
pc_edges = samp.pcd_cuda.select_by_index(list(edges))
pc_edges.paint_uniform_color([0, 1, 1])
o3d.visualization.draw_geometries([pc_edges, samp.mesh], point_show_normal=False)
#o3d.visualization.draw_geometries([samp.pcd_legacy, samp.mesh], point_show_normal=False)

#o3d.visualization.draw_geometries([samp.pcd_legacy, samp.mesh] ,point_show_normal=False)

#o3d.io.write_point_cloud("44781_articulated_edges_example.pcd", samp.pcd_legacy)
"""

"""             
start_time = time.time()
pc = Sampling(44781, number_of_points=100000)
pc.get

articulated = pc.get_articulated_points()
edges = pc.get_edges(articulated)
pc.paint(articulated, [0, 0, 1])
pc.paint(edges, [0, 1, 0])

pc.visualize_mesh_and_bounding_boxes()
edges = pc.get_edges(pc.get_articulated_points())

#pc.check_collision(edges)
print("number of articulated", len(edges))
pc.paint(edges, [0, 1, 0])
end_time = time.time()
execution_time = end_time - start_time
print(f"The function executed in {execution_time} seconds.")
balanced = pc.create_balanced_cloud([0.1, 0.3])
o3d.visualization.draw_geometries([balanced], point_show_normal=True)"""





"""


center, radius = pc.bounding_sphere()

# Now you can generate your viewpoints on the bounding sphere
viewpoints = fibonacci_sphere(samples=100, radius=radius*1.25, center=center)


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
mesh_points = np.asarray(pc.mesh_tri.vertices)

hits = pc.sample_points_from_viewpoints(viewpoints)
hits = np.array(hits)
# Visualize
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.scatter([p[0] for p in viewpoints], [p[1] for p in viewpoints], [p[2] for p in viewpoints])
#ax.scatter(mesh_points[:, 0], mesh_points[:, 1], mesh_points[:, 2], color='b', alpha=0.1, s=3)
ax.scatter(hits[:, 0], hits[:, 1], hits[:, 2], color='r', s = 5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
"""

"""

pc = Sampling(7263, number_of_points=100000)

viewpoints = pc.sample_points_on_sphere(radius=1.0, degree_interval=10)
# Define the viewpoint
articulated_points = pc.get_articulated_points()
pc.paint(articulated_points, [0, 0, 1])
edges = pc.get_edges(articulated_points)
pc.paint(edges, [0, 1, 0])
pc.pcd_legacy.normals = o3d.utility.Vector3dVector(np.asarray(pc.pcd_cuda.normals) * 1)
bounding_boxes = pc.get_articulated_bounding_boxes()
geometries = [pc.pcd_legacy] + bounding_boxes

o3d.visualization.draw_geometries(geometries,  point_show_normal=True)


def sample_points_from_viewpoints(self, viewpoints, boxes,batch_size=10):

    bbox_centers = [bbox.get_center() for bbox in boxes]
    index_all_triangles = []  # List to store all hit locations across all batches

    # Split the viewpoints into batches
    for i in range(0, len(viewpoints), batch_size):
        batch_viewpoints = viewpoints[i: i + batch_size]
        ray_origins = []  # List to store the ray origins for this batch
        ray_directions = []  # List to store the ray directions for this batch

        for viewpoint in batch_viewpoints:
            for center in bbox_centers:
                direction = center - viewpoint
                direction /= np.linalg.norm(direction)  # Normalize the direction
                ray_origins.append(viewpoint)  # Add the viewpoint as the origin for this ray
                ray_directions.append(direction)  # Add the direction for this ray

        # Convert the lists to numpy arrays with matching shapes
        ray_origins = np.array(ray_origins)
        ray_directions = np.array(ray_directions)

        # Perform the ray tracing for this batch
        locations, index_ray, index_triangles = self.mesh_tri.ray.intersects_location(ray_origins, ray_directions, multiple_hits=False)
        
        # Store the hit locations for this batch
        index_all_triangles.extend(index_triangles)

    return index_all_triangles


def old_get_visible_triangles(self, viewpoints, batchsize=1000):
   
        # Convert the viewpoints to a numpy array if they're not already
        viewpoints = np.array(viewpoints)
        
        # Calculate the centroids of the triangles
        centroids = self.mesh_tri.triangles_center
        remaining_triangles = np.arange(len(centroids))  # Start with all triangles remaining
        print("remaining_triangles", max(remaining_triangles))
        visible_triangles = set()
        # Create a scene and add the triangle mesh
        scene = o3d.t.geometry.RaycastingScene()
        mesh_id = scene.add_triangles(self.mesh)

        for viewpoint in viewpoints:
            print("viewpoint", viewpoint)
            hit_triangles = set()
            start = time.time()
            # Process remaining triangles in batches
            for i in range(0, len(remaining_triangles), batchsize):
                batch_indices = remaining_triangles[i:i+batchsize]  # Indices of the triangles in this batch
                batch_centroids = centroids[batch_indices]  # Centroids of the triangles in this batch

                # Get the directions from the viewpoint to each centroid in the batch
                directions = batch_centroids - viewpoint
                # Normalize the directions
                directions /= np.linalg.norm(directions, axis=1)[:, np.newaxis]

                # Perform ray casting
                start_ray = time.time()
                #locations, index_ray, indices_tri = self.mesh_tri.ray.intersects_location(ray_origins=[viewpoint]*len(batch_centroids), ray_directions=directions, multiple_hits=False)
                rays = o3d.geometry.RayMeshIntersectionCaster(ray_origins=[viewpoint]*len(batch_centroids), ray_directions=directions)
                intersection = self.mesh.cast_rays(rays)
                end_ray = time.time()
                # Since we are interested in the intersections with the batch triangles
                # Let's filter the indices based on the size of the batch
                print("batch, raytime", i, end_ray - start_ray)
                
                # Find the visible triangles in this batch
                indices_tri = [info.triangle_index for info in intersection]
                hit_triangles.update(indices_tri)
                
            
            # Add hit triangles from all batches to the visible set
            visible_triangles.update(hit_triangles)
            
            # Update remaining triangles after all batches have been processed
            remaining_triangles = [tri for tri in remaining_triangles if tri not in hit_triangles]
            print("remaining_triangles", len(remaining_triangles))
            end = time.time()
            viewpoint_time = end - start
            print(f"Viewpoint took {viewpoint_time} seconds.")
        return visible_triangles

    
"""