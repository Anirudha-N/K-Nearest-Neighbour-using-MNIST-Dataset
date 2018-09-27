################################## kNN.py ##########################
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

#Function for finding K neighbours 
def findNeighbours(final_train_matrix, testInstance, k):
    Neighbours_distances = []
    for i in range(len(final_train_matrix)):
        respective_distance = CalculateEuclideanDistance(testInstance, final_train_matrix[i,1:785],len(testInstance))
        
        #Contains distance values of test Instance w.r.t all train matrix rows
        Neighbours_distances.append((final_train_matrix[i],respective_distance))
   
    #Sorting Neighbours_distances with ascending order 
    Neighbours_distances.sort(key=operator.itemgetter(1))
   
    #Choose first "K" distances
    Final_neighbors = []
    for i in range(k):
        Final_neighbors.append(Neighbours_distances[i][0])
    return Final_neighbors

#Function for selecting best neighbour
def findBestNeighbour(find_neighbours):
    neighbour_count = {}
    #Finding neighbour with maximum occurance
    for x in range(len(find_neighbours)):
        occurrence =find_neighbours[x][0]
        if occurrence in neighbour_count:
            neighbour_count[occurrence] += 1
        else:
            neighbour_count[occurrence] = 1 
    #Select neighbour with maximum occurance in "find_neighbours" list       
    BestNeighbour = sorted(neighbour_count.items(), key=operator.itemgetter(1), reverse=True)
    return BestNeighbour[0][0]

def main():
    
    #opening csv file
    with open('mnist_train.csv', newline='') as csv_file1:
    
        train_data_lines = csv.reader(csv_file1)
        train_dataset=list(train_data_lines) 
        
        #Converting list into matrix and changing Datatype into int
        train_matrix=np.array(train_dataset).astype("int")
    
    with open('mnist_test.csv', newline='') as csv_file2:
    
        test_data_lines = csv.reader(csv_file2)
        test_dataset=list(test_data_lines)
        
        #Converting list into matrix and changing Datatype into int
        test_matrix=np.array(test_dataset).astype("int")

    #prediction list will contain predicted values
    predictions=[]     
    k=1
    for i in range(len(test_dataset)):
        #finding "k" neighbours
        find_neighbours=findNeighbours(train_matrix,test_matrix[i],k)
        
        #choosing best neighbour among K neighbours
        result = findBestNeighbour(find_neighbours)
        
        predictions.append(result)
        print('Actual Number is:' + repr(test_matrix[i,0])+' Predicted Number is:' + repr(result))
    
    #Finding the accuracy
    true_postives=0
    for i in range(len(test_matrix)):
  		#finding pairs of numbers which satisfies condition Predicted Number=Actual Number 	
    	if test_matrix[i][0]==predictions[i]:
    		true_postives+=1
    
    #accuracy= (true_positives/total Number of test examples)*100
    accuracy=(true_postives/float(len(test_matrix))) * 100.0
    print('Accuracy: ' + repr(accuracy) + '%')
           
main()


    