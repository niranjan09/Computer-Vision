def calculate_threshold_using_histogram(histo_arr, b = 0.47):
	total_b = 0
	total_arr = sum(histo_arr)
	threshold = -1
	for i, gray_level in enumerate(histo_arr):
		total_b+=gray_level
		print i, gray_level
		print total_b/float(total_arr)
		if total_b/float(total_arr)>=b:
			threshold = i
			break
	print threshold, total_b
	return threshold

#arr = [2, 4, 2, 9, 5, 4, 2, 5, 7, 8, 6, 5, 7, 12, 9, 6]
#print calculate_threshold_using_histogram(arr)
#print sum(arr)

