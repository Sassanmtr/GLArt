import xml.etree.ElementTree as ETree
import open3d
import numpy as np

class FileParser:
    """
    A class for parsing URDF files to extract information about articulated and non-articulated objects.
    """
    def __init__(self, object_number):
        """
        Initialize the FileParser.

        Parameters:
        - object_number (str): The identifier for the object whose URDF file is to be parsed.
        """
        self.object_number = object_number
        self.tree = self.get_tree()
        self.path = "/home/freyhe/anaconda3/lib/python3.10/site-packages/pybullet_data/"


    def get_tree(self):
        """
        Parse the URDF file and return the ElementTree object.

        Returns:
        - ElementTree object representing the URDF file.
        """
        path = "/home/freyhe/anaconda3/lib/python3.10/site-packages/pybullet_data/"
        urdf_file = path + str(self.object_number)+"/mobility.urdf"

        tree = ETree.parse(urdf_file)
        return tree

    def get_non_fixed_links(self):
        """
        Get all non-fixed links from the URDF file.

        Returns:
        - List of non-fixed link names.
        """
        root = self.tree.getroot()
        non_fixed_links = []
        for joint in root.iter('joint'):
            if joint.get('type') != 'fixed':
                for child in joint.iter('child'):
                    non_fixed_links.append(child.get('link'))

        return non_fixed_links

    def get_fixed_links(self):
        """
        Get all fixed links from the URDF file.

        Returns:
        - List of fixed link names.
        """
        root = self.tree.getroot()
        fixed_links = []
        for joint in root.iter('joint'):
            if joint.get('type') == 'fixed':
                for child in joint.iter('child'):
                    fixed_links.append(child.get('link'))

        return fixed_links

    def get_all_links(self):
        """
        Get all links from the URDF file.

        Returns:
        - List of all link names.
        """
        root = self.tree.getroot()
        non_fixed_links = []
        for joint in root.iter('joint'):
            for child in joint.iter('child'):
                non_fixed_links.append(child.get('link'))

        return non_fixed_links

    def get_mesh_dict(self, links):
        """
        Create a dictionary of meshes for the given links.

        Parameters:
        - links (list): List of link names to get meshes for.

        Returns:
        - Dictionary where keys are link names and values are list of mesh filenames for each link.
        - Origin (list of floats): The origin coordinates.
        """
        root = self.tree.getroot()
        mesh_dict = {}
        origin = [0,0,0]
        for link in root.iter('link'):
            if link.get('name') in links:
                for visual in link.iter('visual'):
                    if origin == [0,0,0]:

                        origin_loc = visual.find('origin')
                        origin = origin_loc.get('xyz')  
                        origin = list(map(float, origin.split()))    
                        saved_origin = True

                    for geometry in visual.iter('geometry'):
                        for mesh in geometry.iter('mesh'):

                            filename = mesh.get('filename')
                            mesh_dict.setdefault(link.get('name'), []).append(filename)
        return mesh_dict, origin

    def dict_to_mesh(mesh_dict):
        """
        Convert a dictionary of meshes into a single concatenated mesh.

        Parameters:
        - mesh_dict (dict): Dictionary of mesh filenames.

        Returns:
        - A single concatenated mesh object.
        """
        meshes = []
        for mesh_list in mesh_dict.values():
            for mesh_file in mesh_list:
                mesh = open3d.io.read_triangle_mesh(mesh_file)
                meshes.append(mesh)

        merged_mesh = open3d.geometry.TriangleMesh.concatenate(meshes)
        return merged_mesh

    def get_full_path(self,filename):
        """
        Get the full file path for a given filename.

        Parameters:
        - filename (str): The filename.

        Returns:
        - Full file path string.
        """
        return self.path + str(self.object_number) + "/" + filename

    def get_center_of_object(self):
        """
        Calculate the center of mass of the object based on its meshes.
        This is done by taking the center of mass of every link seperatly 
        since all links are given mass 1 by default, and then using the center
        of mass from all the links we can calculate the center of mass of the object

        Returns:
        - Numpy array representing the center of mass coordinates.
        """
        root = self.tree.getroot()
        centers_of_mass = []
        for link in root.iter('link'):
            mesh_concat = None
            for visual in link.iter('visual'):
                
                origin = visual.find('origin').get('xyz')
                origin = list(map(float, origin.split())) 

                for geometry in visual.iter('geometry'):
                        for mesh in geometry.iter('mesh'):
                            filename = mesh.get('filename')

                            mesh = open3d.io.read_triangle_mesh(self.get_full_path(filename))
                            if mesh_concat is None:
                                mesh_concat = mesh
                            else: 
                                mesh_concat += mesh
            if mesh_concat is None:
                continue
            vertices = np.asarray(mesh_concat.vertices)
            print(vertices.shape)

            mean = np.array(vertices.mean(axis=0))

            origin = np.array(origin)

            centers_of_mass.append(mean) #center of links


        return np.mean(centers_of_mass, axis=0) #np.mean(centers_of_mass, axis=0) #center of object


    def get_articulated_bounding_boxes(self):
        """
        Get bounding boxes for all articulated links.

        Returns:
        - List of bounding boxes.
        """
        non_fixed_links = self.get_non_fixed_links()
        mesh_dict, origin = self.get_mesh_dict(non_fixed_links)
        bounding_boxes = self.get_bounding_boxes(mesh_dict)
        return bounding_boxes

    def get_articulated_meshes(self): #ouputs a dictionary of links with their meshes 
        """
        Get meshes for all articulated links.

        Returns:
        - Dictionary of meshes for articulated links.
        - Origin (list of floats): The origin coordinates.
        """
        non_fixed_links = self.get_non_fixed_links()
        mesh_dict, origin = self.get_mesh_dict(non_fixed_links)
        return mesh_dict, origin

    def get_non_articulated_meshes(self): #ouputs a dictionary of links with their meshes
        """
        Get meshes for all non-articulated links.

        Returns:
        - Dictionary of meshes for non-articulated links.
        - Origin (list of floats): The origin coordinates.
        """
        fixed_links = self.get_fixed_links()
        mesh_dict, origin = self.get_mesh_dict(fixed_links)
        return mesh_dict, origin

    def get_all_meshes(self):
        """
        Get meshes for all links.

        Returns:
        - Dictionary of meshes for all links.
        - Origin (list of floats): The origin coordinates.
        """
        all_links = self.get_all_links()
        mesh_dict, origin = self.get_mesh_dict(all_links)
        return mesh_dict, origin
