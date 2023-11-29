import os
import numpy as np

# Specify the directory you want to loop through
data_dir = "/home/mokhtars/Documents/articulatedobjectsgraspsampling/grasps"

# Check if the directory exists
if os.path.exists(data_dir) and os.path.isdir(data_dir):
    # Loop through the files in the directory
    for filename in os.listdir(data_dir):
        # Check if the item is a file (not a subdirectory)
        if os.path.isfile(os.path.join(data_dir, filename)):
            print("File:", filename)
            data = np.load(os.path.join(data_dir, filename))
            print()
else:
    print(f"The directory '{data_dir}' does not exist.")