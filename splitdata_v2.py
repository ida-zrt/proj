import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt
import os
import shutil

cls_names = ['normal', 'phone', 'smoke']

sizes = []
heights = []
widths = []
hw_ratios = []
for name in cls_names:
    pics = glob.glob('./data/smoke_data/{}/*'.format(name))
    for pic in pics:
        img = cv.imread(pic)
        height, width = img.shape[0:2]
        sizes.append(height * width)
        heights.append(height)
        widths.append(width)
        hw_ratios.append(height / width)
        print(f'{pic} complete!')

min_sizes = 3744
max_sizes = 31961088

plt.figure(0)
plt.hist(np.sqrt(np.array(sizes)), bins=50, normed=True)
plt.xlabel('size')
plt.ylabel('frequency')
plt.title('Image size distribution')
plt.savefig('size distribution.png', dpi=100)
plt.show()

plt.figure(1)
plt.hist(heights, bins=50, normed=True)
plt.xlabel('height')
plt.ylabel('frequency')
plt.title('Image height distribution')
plt.savefig('height distribution.png', dpi=100)
# plt.show()

plt.figure(2)
plt.hist(widths, bins=50, normed=True)
plt.xlabel('width')
plt.ylabel('frequency')
plt.title('Image width distribution')
plt.savefig('width distribution.png', dpi=100)
# plt.show()

plt.figure(3)
plt.hist(hw_ratios, bins=50, normed=True)
plt.xlabel('hw_ratio')
plt.ylabel('frequency')
plt.title('Image hw_ratio distribution')
plt.savefig('hw_ratio distribution.png', dpi=100)
# plt.show()

plt.figure(4)
plt.plot(widths, heights, '*r')
plt.xlabel('widths')
plt.ylabel('heights')
plt.title('width-height distribution')
plt.savefig('width-height scatter')


small_cnt = 0
medium_cnt = 0
big_cnt = 0
for name in cls_names:
    pics = glob.glob('./data/smoke_data/val/{}/*'.format(name))
    for pic in pics:
        img = cv.imread(pic)
        height, width = img.shape[0:2]
        size = height * width
        if size < 224 * 224:
            path = './data/smoke_data/small/{}/'.format(name)
            if not os.path.exists(path):
                os.makedirs(path)
            picname = os.path.split(pic)[1]
            shutil.copy(pic, path + picname)
            small_cnt = small_cnt + 1
        elif size >= 224 * 224 and size < 1792 * 1792:
            path = './data/smoke_data/medium/{}/'.format(name)
            if not os.path.exists(path):
                os.makedirs(path)
            picname = os.path.split(pic)[1]
            shutil.copy(pic, path + picname)
            medium_cnt = medium_cnt + 1
        else:
            path = './data/smoke_data/big/{}/'.format(name)
            if not os.path.exists(path):
                os.makedirs(path)
            picname = os.path.split(pic)[1]
            shutil.copy(pic, path + picname)
            big_cnt = big_cnt + 1
