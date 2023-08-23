import pybullet as p
import pybullet_data
import object_info
import loader
from gripper import Gripper


def setup_physics_client():
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    return physicsClient


def setGravity_global(gripclass,x,y,z):
    gravity = (x,y,z)
    gripclass.gravity = gravity
    p.setGravity(*gravity)

def setup_env(start_position_camera) -> tuple:


    physicsClient = setup_physics_client()
    p.setTimeStep(1/240)   #1/240 is basic  

    startOrientation = p.getQuaternionFromEuler([1.5708, 0, 1.5708])  #the input here is radien and 1.5708rad is 90°
    p.resetDebugVisualizerCamera(
        cameraDistance=0.5, 
        cameraYaw=200, 
        cameraPitch=-30, 
        cameraTargetPosition=start_position_camera
        )

def setup_top_down_camera() -> None:
    camera_distance = 7  # Distance from the camera to the target
    camera_pitch = -89  # Top-down angle in degrees
    camera_yaw = 0  # Yaw angle in degrees
    camera_target = (0, -0.1, 1.5)  # Target position (x, y, z)

    # Set the camera position and orientation
    p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, camera_target)