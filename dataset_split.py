import os
import shutil
import configs as cfg
import glob
import numpy as np

# image path
path = cfg.image_folder

output_path = cfg.image_folder

train_val_split = cfg.train_val_split

classes = os.listdir(path)

# remove output path if exist
if os.path.exists(output_path + '/train/'):
    shutil.rmtree(output_path + '/train/')

if os.path.exists(output_path + '/val/'):
    shutil.rmtree(output_path + '/val/')

for cls in classes:
    names = np.array(glob.glob(path + cls + '/*.jpg'))
    indices = range(len(names))
    train_size = int(len(names) * train_val_split[0])

    # random sample from indices
    train_data_id = np.random.choice(indices, train_size)
    train_data_names = names[train_data_id]
    val_data_names = names[~train_data_id]

    if not os.path.exists(output_path + '/train/' + cls) or \
            not os.path.exists(output_path + '/val/' + cls):
        os.makedirs(output_path + '/train/' + cls)
        os.makedirs(output_path + '/val/' + cls)

    for data_name in train_data_names:
        name = os.path.split(data_name)[1]
        output_name = output_path + 'train/' + cls + os.sep + name
        shutil.copy(data_name, output_name)
        print(f'copied {data_name} to {output_name}')

    for data_name in val_data_names:
        name = os.path.split(data_name)[1]
        output_name = output_path + 'val/' + cls + os.sep + name
        shutil.copy(data_name, output_name)
        print(f'copied {data_name} to {output_name}')
