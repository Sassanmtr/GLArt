import pybullet as p
import pybullet_data


def setup_physics_client():
    """
    Set up the physics client using PyBullet's Direct mode.
    Returns:
    The ID of the physics client.
    """
    physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    return physicsClient


def setGravity_global(gripclass,x,y,z):
    """
    Set the gravity vector for the simulation environment.
    Parameters:
    gripclass: An instance of the Gripper class.
    x, y, z: The components of the gravity vector.
    """
    gravity = (x,y,z)
    gripclass.gravity = gravity
    p.setGravity(*gravity)


def setup_env(start_position_camera) -> tuple:
    """
    Set up the simulation environment.
    
    Parameters:
    start_position_camera: The starting position for the debug visualizer camera.
    """
    p.setTimeStep(1/240)   
    p.resetDebugVisualizerCamera(
        cameraDistance=0.5, 
        cameraYaw=200, 
        cameraPitch=-30, 
        cameraTargetPosition=start_position_camera
        )
