from PIL import Image

import numpy as np

import sys

test_data = './data/smoke_data/val/smoke/6723.jpg'

image = np.array(Image.open(test_data))
