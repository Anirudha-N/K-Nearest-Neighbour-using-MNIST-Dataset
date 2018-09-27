################################## sliding_window.py ##########################
################################## Created By Anirudha Nilegaonkar ##########################
import csv
import numpy as np
import operator 

#Function for finding Euclidean Distance
def CalculateEuclideanDistance(input_1, input_2, length):
    distance = 0
    for i in range(length-1):
        distance +=(input_1[i] - input_2[i])**2
    Euclidean_distance = distance**(1/2)
    return Euclidean_distance 

#Function for finding 9 distances by using sliding window of 28X28 on 30X30 matirx 
def slidingWindow(final_train_matrix, testInstance):
	nine_distance=[]
	for i in range(3):
		for j in range(3):
			#creating window of 28X28 from 30X30 train matrix
			#You will get 9 "28X28" windows  
			window=final_train_matrix[i:i+28,j:j+28]
			temp=np.ravel(window)
			#Find distance between testInstance and each window
			sliding_window_distance = CalculateEuclideanDistance(testInstance,temp,len(testInstance))
			nine_distance.append(sliding_window_distance)
	#select min distance among 9 distances
	dist=min(nine_distance)
	return dist

#Function for selecting best neighbour
def findBestNeighbour(find_neighbours):
    neighbour_count = {}
    #Finding neighbour with maximum occurance
    for x in range(len(find_neighbours)):
        occurrence =find_neighbours[x]
        if occurrence in neighbour_count:
            neighbour_count[occurrence] += 1
        else:
            neighbour_count[occurrence] = 1 
    #Select neighbour with maximum occurance in "find_neighbours" list       
    BestNeighbour = sorted(neighbour_count.items(), key=operator.itemgetter(1), reverse=True)
    return BestNeighbour[0][0]

#Making 28X28 train matrix image into 30X30 using padding
def imagePadding(Input_Matrix):
	x=np.reshape(Input_Matrix, (28, 28))
	#Appending two columns of 0's at the end of 28X28 column matrix now matrix is of 28X30 
	Desired_Matrix = np.zeros((28,30))
	Desired_Matrix[:,:-2] = x
	No_Column_Input_Matrix=np.size(Desired_Matrix,1)
	if(No_Column_Input_Matrix==30):
		Extra_row=np.zeros(30)
	#appending row's with zeros at the end of Matrix
	for i in range(28):
		Desired_Matrix = np.vstack([Desired_Matrix, Extra_row])
	index=[]

	#Shifting rows according to index
	for i in range(29):
		index.append(i)
	temp=[29]
	final_index=temp+index
	Column_Exchange=Desired_Matrix[:,final_index]
	padding=Column_Exchange[final_index,:]
	return padding

def FindAccuracy(test_matrix, predictions):
    true_positive = 0
    for i in range(len(test_matrix)):
    	#finding pairs of numbers which satisfies condition Predicted Number=Actual Number
        if test_matrix[i][0] == predictions[i]:
            true_positive += 1
    #accuracy= (true_positives/total Number of test examples)*100
    return (true_positive/float(len(test_matrix))) * 100.0 

def main():
	#prediction list will contain predicted values
	predictions=[]
	k=3 #Optimal K

	#For train data
	with open('mnist_train.csv',newline='') as csv_file1:
		train_data_lines=csv.reader(csv_file1)
		train_dataset=list(train_data_lines)
		train_matrix=np.array(train_dataset).astype("int")
	
	#for test data
	with open('mnist_test.csv',newline='') as csv_file2:
		test_data_lines=csv.reader(csv_file2)
		test_dataset=list(test_data_lines)
		test_matrix=np.array(test_dataset).astype("int")
	
	#28x28 test matrix
	#finding distance of each test image with 60000 training images using sliding window
	
	for j in range(len(test_matrix)):
		First_Digit_test_matrix=test_matrix[j,0]
		Input_test_Matrix=test_matrix[j,1:785]
		flag=0

		Distance=[]
		
		for i in range(len(train_matrix)):
			First_Digit_train_matrix=train_matrix[i,0]
			Input_Matrix=train_matrix[i,1:785]
			re_matrix=imagePadding(Input_Matrix)
			#finding distances of a test image with 1 training images using sliding window
			dist=slidingWindow(re_matrix,Input_test_Matrix)#,flag)
			#Contains distance values of test Instance w.r.t all train matrix rows
			Distance.append((First_Digit_train_matrix,dist))

		#Sorting Neighbours_distances with ascending order 
		Distance.sort(key=operator.itemgetter(1))
		Final_neighbors = []

		#Choose first "K" distances
		for s in range(k):
			Final_neighbors.append(Distance[s][0])
		result = findBestNeighbour(Final_neighbors)
		predictions.append(result)
		print('Actual Number is:' + repr(First_Digit_test_matrix) + 'Predicted Number is:' + repr(result))
	accuracy = getAccuracy(test_matrix, predictions)
	print('Accuracy: ' + repr(accuracy) + '%')
main()