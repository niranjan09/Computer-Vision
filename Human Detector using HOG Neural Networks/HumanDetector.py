"""
Project 2: Human Detection Using HOG and Neural Networks


"""
#imporing numpy for matrix related operations
import numpy as np
#importing matplotlib for image read/write and graph plotting
import matplotlib.pyplot as plt
import matplotlib.image as mpl_img
import matplotlib.cm as cm

#importing sys to get command line arguments passed to program
import sys
#importing os to traverse through directories and get images
import os

#This class contains all functions related to image processing
class ImageProcessor:
        
    def __init__(self):
        #Defining HOG related parameters
        self.HOG_WINDOW_SIZE = (128, 128)
        self.HOG_CELL_SIZE = (8, 8)
        self.HOG_BLOCK_SIZE = (16, 16)
        self.HOG_STEP_SIZE = (8,8)
    
    #This function takes image data folders and returns concatenated list of
    #images and another list which contains expected output for that image
    def segregateImages(self, data_dict, file_path_delimiter):
        image_path_list = []
        data_output = []
        
        for data_folder in data_dict.keys():
            for dirname, subdirlist, filelist in os.walk(data_folder):
                for image_file_name in filelist:
                    image_path = data_folder + file_path_delimiter + image_file_name
                    image_path_list.append(image_path)
            
                    data_output.append([data_dict[data_folder]])
        return image_path_list, data_output
    
    #Function to convert RGB image into grayscale image
    def rgbToGray(self, img):
        return img[...,:3].dot([0.299, 0.587, 0.114])

    #Function to display Image
    def displayImage(self, image, text):
        #Image is quantized to integer values just before displaying
        #Before displaying, image might be or might not be integers
        #plt.margins(0, 0)
        #plt.subplots_adjust(top=1, bottom = 0, right = 1, left = 0, wspace = 0, hspace = 0)
        plt.imshow(image.astype(np.int), cmap = cm.gray ,vmin=0, vmax=255)
        plt.title(text)
        
        #Save figure before displaying too
        plt.savefig(text.split("#")[-1], box_inches = 'tight', pad_inches = 0)
        plt.show()


    #Function to read image from file
    def readImage(self, filepath):
        return np.array(plt.imread(filepath))  

    #Function to perform convolution operation
    def convolve2d(self, img, filt, normalizing_factor):
        
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
        
        #we will assign zero value to where this operation is undefined
        out_img = np.append(out_img, np.zeros((len(out_img), filter_size//2)), 1)
        out_img = np.append(out_img, np.zeros((filter_size//2, len(out_img[0]))), 0)
        out_img = np.append(np.zeros((len(out_img), (filter_size-1)//2)), out_img, 1)
        out_img = np.append(np.zeros(((filter_size-1)//2, len(out_img[0]))), out_img, 0)
        
        return out_img

    #Function to find gradient using prewitt operator
    def prewittOperator(self, image):
        Px = [[-1.0, 0.0, 1.0], 
              [-1.0, 0.0, 1.0], 
              [-1.0, 0.0, 1.0]]
        
        Py = [[1.0, 1.0, 1.0], 
              [0.0, 0.0, 0.0],
              [-1.0, -1.0, -1.0]]
        
        horizontal_gradients = self.convolve2d(image, Px, 1.0)    
        vertical_gradients = self.convolve2d(image, Py, 1.0)
        
        edge_magnitude = (horizontal_gradients**2 + vertical_gradients**2)**0.5
        
        return edge_magnitude, horizontal_gradients, vertical_gradients
    
    #Function to get Feature vector calculated using technique of HOG, 
    #when values of gradient angles and edge magnitudes are provided 
    def getFeatVecWithHOG(self, edge_magnitudes, horizontal_gradients, vertical_gradients,    \
                saveToFile = False, fileName = None, cell_size = None, block_size = None, step_size = None):
        #If cell, block or step size are not provided to function,
        #then global default values are taken from class definitions
        if(cell_size == None):
            cell_size = self.HOG_CELL_SIZE
        if(block_size == None):
            block_size = self.HOG_BLOCK_SIZE
        if(step_size == None):
            step_size = self.HOG_STEP_SIZE
        
        image_height, image_width = horizontal_gradients.shape
        v_cell_size, h_cell_size = cell_size[0], cell_size[1]
        feature_vector = []
        
        #we are counting total no of horizontal and vertical 
        #cells and blocks for the given image
        v_cells_count, h_cells_count = image_height//cell_size[0], image_width//cell_size[1]
        v_cells_per_block, h_cells_per_block = np.array(block_size)/np.array(cell_size)
        v_cells_per_step, h_cells_per_step = np.array(step_size)/np.array(step_size)
        v_blocks_count = (v_cells_count - v_cells_per_block)/v_cells_per_step + 1
        h_blocks_count = (h_cells_count - h_cells_per_block)/h_cells_per_step + 1
        
        #we calculate gradient angle for the given image using tan inverse function
        gradient_angles = np.arctan2(vertical_gradients, horizontal_gradients)        \
                        *180.0/np.pi
        
        #we are keeping the range of our angles between [-10, 170)
        gradient_angles[gradient_angles<-10.0] += 180.0
        gradient_angles[gradient_angles>=170.0] -= 180.0
        
        #dividing by 20 to know in which bins that angle is falling
        gradient_angles = gradient_angles/20.0
        
        #creating an array to store histograms of all cells
        cell_histograms = np.zeros((v_cells_count, h_cells_count, 9))
        
        #lets fill the histogram array by traversing gradient angles
        for vp in range(image_height):
            for hp in range(image_width):
                
                #edge magnitude and gradient_angle at current pixel
                current_angle = gradient_angles[vp][hp]
                current_magnitude = edge_magnitudes[vp][hp]
                
                #calculate nearby bins
                lower_bin = np.floor(current_angle)
                higher_bin = np.ceil(current_angle)
                
                #calculate angles distance from nearby bins
                lower_bin_dist = current_angle - lower_bin
                higher_bin_dist = higher_bin - current_angle
                
                #calculate values based on distance from centers of two bins
                #nearer bin will have higher weightage, so cross relation
                lower_bin_val = current_magnitude*higher_bin_dist
                higher_bin_val = current_magnitude*lower_bin_dist
                
                #update histogram array based on these values
                cell_histograms[int(vp//v_cell_size)][int(hp//h_cell_size)][int(lower_bin)]+=lower_bin_val
                cell_histograms[int(vp//v_cell_size)][int(hp//h_cell_size)][int(higher_bin%9)]+=higher_bin_val
                
        #block level normalization
        for vwc in range(0, v_blocks_count, v_cells_per_step):
            for hwc in range(0, h_blocks_count, h_cells_per_step):
                
                #getting the current block into one flat array
                current_block = cell_histograms[vwc:vwc+v_cells_per_block, hwc:hwc+h_cells_per_block].reshape(-1)
                #normalizing over this scope of block
                current_block_L2_norm = sum(current_block**2)**0.5
                
                #if all elements are zero then no need to normalze!
                if current_block_L2_norm!=0:
                    current_block/=current_block_L2_norm
                
                #appending this to feature vector
                feature_vector.append(current_block)
    
        feature_vector = np.array(feature_vector).reshape(-1, 1)
    
        #Save feature vector to file
        if saveToFile:
            np.savetxt(fileName+".txt", feature_vector)
        return feature_vector
    
    #This function takes input of list of images and returns list of feature vectors
    #for each image calculated using HOG. edge magnitude and gradient angles are calculated
    #using prewitt operator which is prerequisite for HOG feature vector calculation
    def getPrewittHOGofImageList(self, image_path_list, display = False):
        feat_vec_list = []
        for image_path in image_path_list:
            input_image = self.readImage(image_path)
            if('png' in sys.argv[1]):
                input_image*=255.0
        
            grayscale_input_image = self.rgbToGray(input_image)
            #comment out this code if you 
            #displayImage(grayscale_input_image, "Lena Test Image")
        
            edge_magnitudes, horizontal_gradients, vertical_gradients,  =             \
                        self.prewittOperator(grayscale_input_image)
            
            #comment out this code if you want to view gradients of images
            #displayImage(np.abs(horizontal_gradients/3.0), "Horizontal Gradients")
            #displayImage(np.abs(vertical_gradients/3.0), "Vertical Gradients")
            if display:
                self.displayImage(edge_magnitudes/(3*(2**0.5)), "Edge Magnitude Image for#"+(image_path.split(os.sep)[-1])[:-4])
            if display:
                feat_vector = self.getFeatVecWithHOG(edge_magnitudes, horizontal_gradients, vertical_gradients, saveToFile = True, fileName = image_path.split(os.sep)[-1][:-4])
            else:
                feat_vector = self.getFeatVecWithHOG(edge_magnitudes, horizontal_gradients, vertical_gradients, False, None)
            feat_vec_list.append(feat_vector)
        
        feat_vec_list = np.array(feat_vec_list)
        
        return feat_vec_list

#class to create perceptron
#architecture of neural network is passed as an argument
#(7524, 250, 1): 7524 input layer size, 250 is hidden layer size and 1 is output layer size
class Perceptron:
    def __init__(self, architecture = (7524, 250, 1), epoch_count = 1000, learning_rate = 0.01):
        np.random.seed(9)
        
        self.architecture = architecture
        
        #weights are initialized with random values
        self.w1 = np.random.randn(architecture[1], architecture[0])*0.01
        self.w2 = np.random.randn(architecture[2], architecture[1])*0.01
        
        #bias is initialized with zeros
        self.b1 = np.zeros((architecture[1], 1))
        self.b2 = np.zeros((architecture[2], 1))
        
        #All the other values are initially None
        self.z1 = self.z2 = self.a1 = self.a2 = None
        self.dz1 = self.dz2 = self.da1 = self.da2 = None
        self.db1 = self.db2 = self.dw1 = self.dw2 = None
        
        self.learning_rate = learning_rate
        self.epoch_count = epoch_count
    
    #This function performs forward propagation on neural network
    def forwardPass(self, train_data):
        self.z1 = self.w1.dot(train_data) + self.b1
        #ReLU operation is used as activation function for hidden layer
        self.a1 = np.maximum(self.z1, 0)
        
        self.z2 = self.w2.dot(self.a1) + self.b2
        #Sigmoid activation function is used in output layer
        self.a2 = 1/(1+np.exp(-self.z2))
    
    #Function to calculate squared error
    def calcSquareError(self, expected_output):
        return 0.5*np.square(self.a2 - expected_output).sum()
    
    #Function to calculate gradient of error with respect to weights and biases of 
    #network
    def backwardPass(self, train_data, expected_output):
        self.da2 = self.a2 - expected_output
        self.dz2 = self.da2*self.a2*(1 - self.a2)
        
        self.dw2 = self.dz2.dot(self.a1.T)
        #bias is just single value, we can take sum of all current batch size
        #or average of it, but since in our case we are training image by image
        #it doesnt matter
        self.db2 = np.sum(self.dz2, axis = 1, keepdims = True)
        
        self.da1 = self.w2.T.dot(self.dz2)
        self.dz1 = np.array(self.da1)
        #calculating derivative of relu and multiplyig by it
        self.dz1[self.z1 < 0] = 0
        
        self.dw1 = self.dz1.dot(train_data.T)
        self.db1 = np.sum(self.dz1, axis = 1, keepdims = True)
    
    #This function updates values of weights and biases using learning rate and
    #calculated gradients
    def updateParameters(self):
        self.w1 = self.w1 - self.learning_rate * self.dw1
        self.b1 = self.b1 - self.learning_rate * self.db1
        
        self.w2 = self.w2 - self.learning_rate * self.dw2
        self.b2 = self.b2 - self.learning_rate * self.db2
    
    #This Function is used to train perceptron, it follows path of forward propagation
    #backward propagation and weights updation respectively for each training image in
    #the data set. This is 1 epoch and "epoch_count" times this is repeated
    def trainPerceptron(self, train_image_path_list, train_data_input, train_data_output):
        print("Training started ...hold your breath!!!")
        epoch_error_list = []
        
        train_data_length = len(train_data_input)
        #NOTE: epoch count is not only the stopping condition
        for epoch in range(self.epoch_count):        
            
            epoch_error = 0.0
            
            for data_counter, train_data in enumerate(train_data_input):
                self.forwardPass(train_data)
                error = self.calcSquareError(train_data_output[data_counter])
                epoch_error += error
                self.backwardPass(train_data, train_data_output[data_counter])
                self.updateParameters()
                #break
            #break
            #comment out this code to implement learning rate decay, learning rate
            #decay is useful to minimize overall error
            #if epoch%500 == 0:
            #    self.learning_rate/=2
            
            epoch_error_list.append(epoch_error)
            #After each epoch Mean error for that epoch is printed on console
            print("epoch No: " + str(epoch) + "/" + str(self.epoch_count),                     \
                "Error Average for this epoch", epoch_error/train_data_length)
            #Lets put some stopping condition other than number of epochs
            #lets say if average error is not decreasing by 1% after 500 epochs then we willl stop
            if epoch>500 and epoch_error_list[-1]/epoch_error_list[-2]>0.999:
                break
        plt.plot(epoch_error_list)
        plt.show()
    #Function to test our code on test data images
    #It predicts probability of human in the given image
    #If the probability is greater than 0.5 then it is considered as Human and
    #we are calculating prediction accuracy based on this quantized prediction
    # e.g. 2 out of 3 images are predicted correctly: accuracy = 66.67%
    def testPerceptron(self, test_image_path_list, test_data_input, test_data_output):
        misclassified_count = 0
        
        pos_list = []
        neg_list = []
        
        for data_counter, test_data in enumerate(test_data_input):
            self.forwardPass(test_data)
            print(test_image_path_list[data_counter] + "Predicted Human Probability: " + str(self.a2),                \
                "Actual Probability: " + str(test_data_output[data_counter]))
            
            #error by considering values>0.5 as prediction as Human
            current_pred = np.round(self.a2.sum())
            if current_pred:
                pos_list.append([test_image_path_list[data_counter], str(self.a2.sum())])
            else:
                neg_list.append([test_image_path_list[data_counter], str(self.a2.sum())])
            misclassified_count +=                                                 \
                (int(current_pred - test_data_output[data_counter]) == 0)
        
        #we can show the two separate sets of images generate in pos_list and neg_list
        #as positive predictions and negative predictions
        #fig = plt.figure()
        
        
        print(str(float(misclassified_count)/float(len(test_data_output))*100) + "% prediction accuracy of the model on test data.")


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        datafilepath = sys.argv[1]
    else:
        datafilepath = raw_input("data filepath was not provided, please enter a filepath for data now:")
    
    #The expected folder structure is like given below:
    #the input folder path should contain these 4 folders
    TRAIN_POSITIVE_FOLDER = 'train_positive'
    TRAIN_NEGATIVE_FOLDER = 'train_negative'
    TEST_POSITIVE_FOLDER = 'test_positive'
    TEST_NEGATIVE_FOLDER = 'test_negative'
    FILE_PATH_DELIMITER = os.sep
    
    TRAIN_POSITIVE_PATH = datafilepath + FILE_PATH_DELIMITER + TRAIN_POSITIVE_FOLDER
    TRAIN_NEGATIVE_PATH = datafilepath + FILE_PATH_DELIMITER + TRAIN_NEGATIVE_FOLDER
    TEST_POSITIVE_PATH = datafilepath + FILE_PATH_DELIMITER + TEST_POSITIVE_FOLDER
    TEST_NEGATIVE_PATH = datafilepath + FILE_PATH_DELIMITER + TEST_NEGATIVE_FOLDER
    
    train_data_dict = {TRAIN_POSITIVE_PATH:1, TRAIN_NEGATIVE_PATH:0}
    test_data_dict = {TEST_POSITIVE_PATH:1, TEST_NEGATIVE_PATH:0}
        
    imageProcessor = ImageProcessor()
    
    train_image_path_list, train_data_output =                                            \
                imageProcessor.segregateImages(train_data_dict, FILE_PATH_DELIMITER)
    test_image_path_list, test_data_output =                                            \
                imageProcessor.segregateImages(test_data_dict, FILE_PATH_DELIMITER)
    
    train_data_input = np.array(imageProcessor.getPrewittHOGofImageList(train_image_path_list))
    test_data_input = np.array(imageProcessor.getPrewittHOGofImageList(test_image_path_list, display = False))
    
    perceptron = Perceptron()
    perceptron.trainPerceptron(train_image_path_list, train_data_input, train_data_output)
    perceptron.testPerceptron(test_image_path_list, test_data_input, test_data_output)

