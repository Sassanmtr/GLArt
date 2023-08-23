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

def get_meshes(obj, only_articulated = False):
        if only_articulated == True:
            mesh_dict, origin = obj.get_articulated_meshes()
        if only_articulated == False:
            mesh_dict, origin = obj.get_all_meshes()


        transformed_meshes = []

        for key, value in mesh_dict.items():
            for filename in value:
                # Load the mesh
                path = "/home/freyhe/anaconda3/lib/python3.10/site-packages/pybullet_data/"
                
                loaded = tri.load_mesh(obj.get_full_path(filename))
                
                # If the loaded object is a Scene, dump it to get a list of Trimesh objects
                meshes = loaded.dump() if isinstance(loaded, tri.Scene) else [loaded]
                
                for mesh in meshes:
                    # Create the transformation matrix
                    transform_matrix = tri.transformations.translation_matrix(origin)
                    # Apply the transformationreturn self.obj.get_articulated_bounding_boxes(
                    #transformed_mesh = mesh.apply_transform(transform_matrix)
                    # Add the transformed mesh to the list
                    transformed_meshes.append(mesh)

        return transformed_meshes

def merge_meshes(meshes):
    return tri.util.concatenate(meshes)

def get_bounding_boxes(meshes):
        bounding_boxes = []
        for mesh in meshes:
            mesh = trimesh_to_o3d(mesh)
            bounding_box = mesh.get_axis_aligned_bounding_box()
            bounding_boxes.append(bounding_box)
        return bounding_boxes

def trimesh_to_o3d(mesh):
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh.faces)
        return mesh_o3d

def fibonacci_sphere(radius, center, samples = 100):
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
    # Suppose `pcd` is your point cloud and `selected_points` is your list of points
    all_indices = set(np.arange(len(points)))
    return list(all_indices - set(selection))

def subdivide_mesh(mesh, max_triangle_area):
    while np.max(mesh.area_faces) > max_triangle_area:
        faces_to_subdivide = [i for i in range(len(mesh.faces)) if mesh.area_faces[i] > max_triangle_area]
        if not faces_to_subdivide:
            break  # Exit the loop if no faces left to subdivide
        subdivided_mesh = mesh.subdivide(faces_to_subdivide)
        return subdivided_mesh

class Sampling:
    def __init__(self, object_number, number_of_points = 10000):
        self.number_of_points = number_of_points
        self.object_number = object_number
        self.obj = FileParser(self.object_number)
        self.mesh_tri = merge_meshes(get_meshes(self.obj, only_articulated = False))
        self.mesh_tri = subdivide_mesh(self.mesh_tri, 0.01)
        print("articulated triangles", len(self.mesh_tri.faces))

        print(len(self.mesh_tri.faces))

        self.mesh = trimesh_to_o3d(self.mesh_tri)
        viewpoints = self.get_viewpoints(1000)

        triangles = self.get_visible_triangles(viewpoints)
        self.visable_mesh = self.mesh_from_triangle_index(triangles)
        print("Number of triangles:", len(self.visable_mesh.triangles))
        #self.make_mesh_double_sided()

        self.visable_mesh.normalize_normals()

        self.pcd_cuda = self.visable_mesh.sample_points_uniformly(number_of_points, use_triangle_normal=True)
        self.correct_normals()

        self.pcd_tree = o3d.geometry.KDTreeFlann(self.pcd_cuda)
        self.pcd_legacy = o3d.geometry.PointCloud(self.pcd_cuda)
        self.pcd = o3d.t.geometry.PointCloud.from_legacy(self.pcd_legacy)
        self.pcd_legacy.paint_uniform_color([1, 0, 0])


# Usage:


    def get_nearest_points(self, number_of_nearest_points,radius, point):
        print("point", point)
        [k, idx, _] = self.pcd_tree.search_radius_vector_3d(self.pcd_cuda.points[point], radius)
        if len(idx) < number_of_nearest_points:
            return list(idx)
        return rand.sample(list(idx), number_of_nearest_points-1)
    
    def paint(self, points, color):
        print(points)
        print(len(points))
        np.asarray(self.pcd_legacy.colors)[list(points), :] = color
    
    def is_potentially_grippable(self, point, points):
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
            if angle >= 45 and angle <= 135:  # between 45 and 135 degrees
                for j in range(len(normals)):
                    angle = np.degrees(np.arccos(np.clip(np.dot(normals[i], normals[j]), -1.0, 1.0)))
                    if angle >= 150: # more than 150 degrees
                        return True


        return False
            
    def get_edges(self, point_index):
        edges = []
        edges_unique = set()
        for index in point_index:
            if index not in edges:
                neighbors = self.get_nearest_points(9, 0.04, index)
                
                if self.is_potentially_grippable(index, neighbors):

                    edge = [index] 
                    edges.append(edge)
                    edges_unique.update(edge)
        return edges_unique
    
    def get_points_in_boxes(self, bounding_boxes: list, points: list):

        points = set()
        for bounding_box in bounding_boxes:

            points.update(bounding_box.get_point_indices_within_bounding_box(self.pcd_cuda.points))
        return points

    def get_articulated_points(self):
        boxes = self.get_articulated_bounding_boxes()
        print("boxes", boxes)
        return self.get_points_in_boxes(boxes, self.pcd_cuda.points)
    
    def get_articulated_bounding_boxes(self):
        meshes = get_meshes(self.obj, only_articulated = True)
        print("meshes", len(meshes))
        return get_bounding_boxes(meshes)
    
    def make_mesh_double_sided(self):
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

        indices = np.array(list(triangle_index))
        indices = indices[indices != 4294967295] # invalid id tag

        new_mesh = o3d.geometry.TriangleMesh()
        new_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(self.mesh.vertices))
        new_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(self.mesh.triangles)[indices])
        return new_mesh

    def mesh_from_triangle_index_tri(self, triangle_index):
        indices = list(triangle_index)
        new_mesh = tri.Trimesh(vertices=self.mesh_tri.vertices, faces=self.mesh_tri.faces[indices])
        return new_mesh

    def sample_visable_points(self):
        center, radius = self.bounding_sphere()
        boxes = self.get_articulated_bounding_boxes()
        number_of_boxes = len(boxes)
        number_of_viewpoints = self.number_of_points // number_of_boxes
        viewpoints = fibonacci_sphere(radius=radius, center=center, samples = number_of_viewpoints)
        print("the number of viewpoints are:", number_of_viewpoints)
        points = self.sample_points_from_viewpoints(viewpoints, boxes)
        points = np.array(points)
        pcd_cuda = o3d.geometry.PointCloud()
        pcd_cuda.points = o3d.utility.Vector3dVector(points)
        return pcd_cuda
       
    def visualize_mesh_and_bounding_boxes(self):
        geometries = [self.mesh]
        bounding_boxes = self.get_articulated_bounding_boxes()
        for bbox in bounding_boxes:
            obb = o3d.geometry.OrientedBoundingBox.create_from_axis_aligned_bounding_box(bbox)
            geometries.append(obb)

        o3d.visualization.draw_geometries(geometries)

    def get_viewpoints(self, number_of_viewpoints):
        center, radius = self.bounding_sphere()
        viewpoints = fibonacci_sphere(radius=radius, center=center, samples = number_of_viewpoints)
        return viewpoints

    def create_balenced_cloud(self, ratio):             #the ratio stems from the antipodal points. It is assumed you want all antipodal points and then it samples based on the number of antipodal points
        #ratio relative to antipodal points
        # Get the points in each category
        articulated_points = self.get_articulated_points()

        unarticulated_points = get_inverse(self.pcd_cuda.points, articulated_points)
        antipodal_points = self.get_edges(articulated_points)

        # Calculate the number of points to select from each category
        total_points = len(unarticulated_points) + len(articulated_points)
        num_antipodal = len(antipodal_points)

        # Calculate remaining points and ratios
        num_unarticulated = round(ratio[0] * num_antipodal)
        num_articulated = round(ratio[1] * num_antipodal)
        print("num articulated", num_unarticulated)
        # Randomly select points from each category
        selected_unarticulated = random.sample(list(unarticulated_points), num_unarticulated)
        selected_articulated = random.sample(list(articulated_points), num_articulated)
        print("selected articulated", len(selected_articulated))

        # Combine the selected points into a new point cloud
        new_points = np.concatenate([selected_unarticulated, selected_articulated, list(antipodal_points)])
        print("new points", len(new_points), new_points)

        new_points = list(map(int, new_points))
        new_point_cloud = o3d.geometry.PointCloud()
        new_points_positions = np.asarray(self.pcd_cuda.points)[new_points]
        new_points_normals = np.asarray(self.pcd_cuda.normals)[new_points]
        new_point_cloud.points = o3d.utility.Vector3dVector(new_points_positions)
        new_point_cloud.normals = o3d.utility.Vector3dVector(new_points_normals)


        return new_point_cloud

    def correct_normals(self):
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
        print(f"Ray tracing took {end_time - start_time} seconds.")

    def check_collision(self, points):
        scene = o3d.t.geometry.RaycastingScene()
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh)
        position = np.asarray(self.pcd_cuda.points)[points]
        directions = np.asarray(self.pcd_cuda.normals)[points]
        origins = np.tile(position, (len(directions), 1))
        rays = np.hstack((origins, directions)).astype(np.float32)
        rays = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)

        # Perform ray casting

        ans = scene.cast_rays(rays)

        print(ans.keys())

"""samp = Sampling(11661, 10000)


pc_1 = samp.create_balenced_cloud([0.1, 0.2])
o3d.visualization.draw_geometries([pc_1], point_show_normal=True)
"""



"""             
start_time = time.time()
pc = Sampling(11661, number_of_points=100000)

pc.visualize_mesh_and_bounding_boxes()
edges = pc.get_edges(pc.get_articulated_points())
#pc.check_collision(edges)
print("number of articulated", len(edges))
pc.paint(edges, [0, 1, 0])
end_time = time.time()
execution_time = end_time - start_time
print(f"The function executed in {execution_time} seconds.")
balenced = pc.create_balenced_cloud([0.1, 0.3])
o3d.visualization.draw_geometries([balenced], point_show_normal=True)"""





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