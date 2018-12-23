"""
                                ###Project 1###
                                Name: Niranjan Vinayak Ketkar
                                University ID: 
                                Email ID: 
                                Title: Canny Edge Detector
"""
#Importing numpy for matrix related operations
import numpy as np

#import pyplot for Reading Writing Image, showing graphs
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#python function to copy arrays/matrix efficiently
import copy
#importing sys to get argv passed to program
import sys


#Function to read image from file
def readImage(filepath, w = -1, h = -1):
    #special handling for raw file as matplotlib cannot read raw file
    if(filepath[-4:] == '.raw'):
        if w==-1 or h == -1:
            print("Height and width of raw image not provided...exiting")
            exit()
        arr = np.fromfile(filepath, dtype = "uint8", sep="")
        if len(arr)!= w*h:
            print("Wrong dimensions provided!")
            exit()
        arr = np.reshape(arr, (w, h))
        return arr
    else:
        return np.array(plt.imread(filepath))    

#Function to display Image
def displayImage(image, text):
    #Image is quantized to integer values just before displaying
    #Before displaying, image might be or might not be integers
    #plt.margins(0, 0)
    #plt.subplots_adjust(top=1, bottom = 0, right = 1, left = 0, wspace = 0, hspace = 0)
    plt.imshow(image.astype(int), cmap =cm.gray ,vmin=0, vmax=255)
    plt.title(text)
    
    #Save figure before displaying too
    plt.savefig(text, box_inches = 'tight', pad_inches = 0)
    plt.show()

#Function to perform convolution operation
def convolve2d(img, filt, normalizing_factor):
    
    #row and col counts of original image
    row_counts = len(img)
    col_counts = len(img[0])
    
    #row and col counts of convolved image
    filter_size = len(filt)
    output_row_count = row_counts-filter_size+1
    output_col_count = col_counts - filter_size+1
    
    #convolved image will have size less than original(depends on size of filter)
    out_img = np.zeros((output_row_count, output_col_count))
        
    #Have avoided use of library functions for convolutional operation
    #and have used just simple for loops for it
    for row in range(output_row_count):
        for col in range(output_col_count):
            temp_val = 0.0
            for r in range(filter_size):
                for c in range(filter_size):
                    temp_val += (float(img[row+r][col+c]) * filt[r][c])            
            out_img[row][col] = temp_val/normalizing_factor
    
    return out_img



#Function to apply guassian filter(7x7) on image
def gaussianFilter(image):
    Gaussian_mask_7x7 = [[1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0],
                         [1.0, 2.0, 2.0, 4.0, 2.0, 2.0, 1.0],
                         [2.0, 2.0, 4.0, 8.0, 4.0, 2.0, 2.0],
                         [2.0, 4.0, 8.0, 16.0, 8.0, 4.0, 2.0], 
                         [2.0, 2.0, 4.0, 8.0, 4.0, 2.0, 2.0], 
                         [1.0, 2.0, 2.0, 4.0, 2.0, 2.0, 1.0], 
                         [1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0]]

    #min = 0 and max = 255*140/140 = 255 so its properly normalized after dividing by 140    
    g_smoothed_image = convolve2d(image, Gaussian_mask_7x7, 140.0)
    return g_smoothed_image

#Function to find gradient using prewitt operator
def prewittOperator(image):
    Px = [[-1.0, 0.0, 1.0], 
          [-1.0, 0.0, 1.0], 
          [-1.0, 0.0, 1.0]]
    
    Py = [[1.0, 1.0, 1.0], 
          [0.0, 0.0, 0.0],
          [-1.0, -1.0, -1.0]]
    
    horizontal_gradients = convolve2d(image, Px, 1.0)    
    vertical_gradients = convolve2d(image, Py, 1.0)
    
    #Lets calculate edge magnitude:
    edge_magnitude = np.zeros((len(horizontal_gradients), len(horizontal_gradients[0])))
    
    for row in range(len(horizontal_gradients)):
        for col in range(len(horizontal_gradients[0])):
            #print horizontal_gradients[row][col]**2, vertical_gradients[row][col]**2
            edge_magnitude[row][col] = (horizontal_gradients[row][col]**2 + vertical_gradients[row][col]**2)**0.5
    return edge_magnitude, horizontal_gradients, vertical_gradients

#Function to apply NMS of Canny edge detector
def nonMaximaSuppression((edge_magnitude, horizontal_gradients, vertical_gradients)):
    #initialize all values with zeros, will be replaced with actual values 
    #whereever applicable, later
    nms_output = np.zeros((len(edge_magnitude), len(edge_magnitude[0])))
    
    #It will not be possible to do NMS on boundary pixels, 
    #so we are keeping them as zeros only
    for row in range(1, len(horizontal_gradients) - 1):
        for col in range(1, len(horizontal_gradients[0]) - 1):
            if horizontal_gradients[row][col]!=0.0:
                gradient_angle =                 \
                (np.arctan(vertical_gradients[row][col]/horizontal_gradients[row][col]))        \
                *180.0/np.pi
            
            #gradient angle will be 90 when denominator is 0
            elif vertical_gradients[row][col]!=0.0:
                gradient_angle = 90.0
            else:
                gradient_angle = 0.0
            
            #find appropriate sector for the angle and compare magnitudes
            if (((gradient_angle >67.5 or gradient_angle<-67.5) and (edge_magnitude[row][col] ==             \
            max([edge_magnitude[row-1][col], edge_magnitude[row][col], edge_magnitude[row+1][col]])))         \
            or ((gradient_angle >=22.5 and gradient_angle<=67.5) and (edge_magnitude[row][col] ==             \
            max([edge_magnitude[row-1][col+1], edge_magnitude[row][col], edge_magnitude[row+1][col-1]])))     \
            or ((gradient_angle >-22.5 and gradient_angle<22.5) and (edge_magnitude[row][col] ==             \
            max([edge_magnitude[row][col+1], edge_magnitude[row][col], edge_magnitude[row][col-1]])))         \
            or ((gradient_angle >=-67.5 and gradient_angle<=-22.5) and (edge_magnitude[row][col] ==         \
            max([edge_magnitude[row+1][col+1], edge_magnitude[row][col], edge_magnitude[row-1][col-1]])))):
                nms_output[row][col] = edge_magnitude[row][col]
    
    return nms_output

#Function to get thresholded image using p-tile method
def thresholdPtile(img, ptile):
    print("P-Tile is:"+ str(ptile*100) + "%")
    
    #initialize output image with zeros
    thresholded_img = np.zeros(img.shape)


    #converting image into 1-d array
    img_array = img.reshape(-1)
    
    #we will remove zeros and sort the image array
    img_array = np.sort(img_array[img_array>0.0])
    
    #finding p% th pixel value from end of array
    #which will be threshold
    threshold = img_array[int(round((1-ptile)*len(img_array)))]
    print("Threshold value is:" + str(threshold))
    print("Total number of edges found:" + str(int(len(img_array)*ptile)))
    
    #Apply thresholding based on the value found above
    thresholded_img[img<threshold] = 0
    thresholded_img[img>=threshold] = 255
    
    return thresholded_img

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        #NxM image is input
        if(len(sys.argv) == 4):
            input_image = readImage(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
        else:
            input_image = readImage(sys.argv[1])
        #display the input image just read
        displayImage(input_image, "Original Image")
        
        #gaussian image will be (N-6)x(M-6)
        gaussian_smoothed_image = gaussianFilter(input_image)
        
        #display gaussian smoothed image
        displayImage(gaussian_smoothed_image, "Output of Gaussian Smoothing")
        
        #calculation of edges using prewitt operator
        #output images after this operation will be (N-8)x(M-8)
        edge_magnitude, horizontal_gradients, vertical_gradients = prewittOperator(gaussian_smoothed_image)
        
        #possible range of gradients is -765 to 765, since -ve value just means opposite direction here,
        #we will divide by 3 and take absolute value of the result to show it as image
        displayImage(np.absolute(horizontal_gradients/3.0), "Horizontal - Gradients")
        displayImage(np.absolute(vertical_gradients/3.0), "Vertical - Gradients")
        
        #maximum value of edge magnitude can reach upto (765^2 + 765^2)^(0.5), 
        #so we will divide by (3*(2^0.5)) to normalize it for display purpose
        displayImage(edge_magnitude/(3.0*(2.0**0.5)), "Edge-Magnitude")

        #Apply NMS to edge magnitudes based on gradient angles
        nms_Image = nonMaximaSuppression((edge_magnitude, horizontal_gradients, vertical_gradients))
        #maximum possible value of NMS output will be same as that of edge magnitude value
        #so for display purpose we will normalize it by same factor
        #Normalization is only needed for display purpose, but since we also need to print
        #threshold value, It 'LOOKS' better to have threshold between 0 and 255
        #So I am normalizing the image before thresholding operation
        nms_Image = nms_Image/(3.0*(2.0**0.5))
        
        #image is already normalized so no need to process further before displaying image
        displayImage(nms_Image, "NMS Output")
        
        #since image is normalized in above step, no need to normalize it again
        #Thresholding is done by keeping image as float only, to increase accuracy!
        
        #Threshold image using P-tile = 0.1
        thresholded_image_10 = thresholdPtile(nms_Image, 0.1)
        displayImage(thresholded_image_10, "P-Tile Threshold = 10%")

        #Threshold image using P-tile = 0.3
        thresholded_image_20 = thresholdPtile(nms_Image, 0.3)
        displayImage(thresholded_image_20, "P-Tile Threshold = 30%")
        
        #Threshold image using P-tile = 0.5
        thresholded_image_50 = thresholdPtile(nms_Image, 0.5)
        displayImage(thresholded_image_50, "P-Tile Threshold = 50%")
    else:
        print("Input Image path not provided...Exiting.")
