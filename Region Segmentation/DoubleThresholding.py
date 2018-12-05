import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def double_thresholding(img, t1, t2):
	region_arr = np.zeros((img.shape[0], img.shape[1]))
	
	idx = img<t1
	region_arr[idx] = -1
	idx = img>t2
	region_arr[idx] = 1
	print region_arr
	for i in range(region_arr.shape[0]):
		for j in range(region_arr.shape[1]):
			if region_arr[i][j]==0:
				if region_arr[i-1][j] == 1 or region_arr[i+1][j] ==1 or region_arr[i][j-1] ==1 or region_arr[i][j+1] ==1:
					region_arr[i][j] = 1
	idx = region_arr == 0
	region_arr[idx] = -1
	print region_arr
	
img =  [[20,30,35,99,89,90,55,99],
		[66,67,87,88,99,87,86,85],
		[77,162,163,189,98,99,93,89],
		[75,180,188,97,120,78,130,98],
		[70,165,170,65,110,70,140,45],
		[98,200,65,75,85,95,130,75],
		[70,100,130,89,160,159,140,99],
		[33,43,54,66,77,86,96,99]]

double_thresholding(np.array(img), 100, 160)
