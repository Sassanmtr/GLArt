from pathlib import Path
import urdfpy
import os
from tqdm import tqdm
import json

success_counter = 0
failure_counter = 0
failure_objects = []
current_dir = Path(__file__).resolve().parent
data_dir = str(current_dir.parent / "PartNetMobilityDataset")
test_data_dir = str(current_dir / "test_data")
object_name = "7128"
test_obj_path = os.path.join(test_data_dir, object_name, "mobility.urdf")
a = urdfpy.URDF.load(test_obj_path)
a.show()

# items = os.listdir(data_dir)
# for item in tqdm(items):
#     urdf_path = os.path.join(data_dir, item, "mobility.urdf")
#     try:
#         urdf_obj = urdfpy.URDF.load(urdf_path)
#         success_counter += 1
#     except:
#         failure_objects.append(item)
#         failure_counter += 1

# print("Success: ", success_counter)
# print("Failure: ", failure_counter)
# # Save the success_objects list as a JSON file
# output_file_path = '/home/mokhtars/Documents/GraspSampler/failure_urdfpy.json'
# with open(output_file_path, 'w') as json_file:
#     json.dump(failure_objects, json_file)

print()

