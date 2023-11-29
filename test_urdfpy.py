import urdfpy
import os
from tqdm import tqdm
import json

success_counter = 0
failure_counter = 0
failure_objects = []
data_dir = "/home/mokhtars/Documents/PartNetMobilityDataset"
a = urdfpy.URDF.load("/home/mokhtars/Documents/articulatedobjectsgraspsampling/7320/mobility.urdf")
a.show()
items = os.listdir(data_dir)
for item in tqdm(items):
    urdf_path = os.path.join(data_dir, item, "mobility.urdf")
    try:
        urdf_obj = urdfpy.URDF.load(urdf_path)
        success_counter += 1
    except:
        failure_objects.append(item)
        failure_counter += 1

print("Success: ", success_counter)
print("Failure: ", failure_counter)
# Save the success_objects list as a JSON file
output_file_path = '/home/mokhtars/Documents/articulatedobjectsgraspsampling/failure_urdfpy.json'
with open(output_file_path, 'w') as json_file:
    json.dump(failure_objects, json_file)

print()

