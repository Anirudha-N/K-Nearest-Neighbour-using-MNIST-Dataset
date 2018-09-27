################################## cross_validation.py ##########################
################################## Created By Anirudha Nilegaonkar ##########################
import csv
import numpy as np
import operator
import random

#Function for finding Euclidean Distance
def CalculateEuclideanDistance(input_1, input_2, length):
    distance = 0
    for i in range(length-1):
        distance += (input_1[i] - input_2[i]) **2
    Euclidean_distance=distance**(1/2)
    return Euclidean_distance

#Function for K finding neighbours  
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
        response =find_neighbours[x][0]
        if response in neighbour_count:
            neighbour_count[response] += 1
        else:
            neighbour_count[response] = 1 
    #Select neighbour with maximum occurance in "find_neighbours" list       
    BestNeighbour = sorted(neighbour_count.items(), key=operator.itemgetter(1), reverse=True)
    return BestNeighbour[0][0]

def FindAccuracy(test_matrix, predictions):
    true_positive = 0
    for i in range(len(test_matrix)):
        #finding pairs of numbers which satisfies condition Predicted Number=Actual Number
        if test_matrix[i][0] == predictions[i]:
            true_positive += 1
    #accuracy= (true_positives/total Number of test examples)*100
    return (true_positive/float(len(test_matrix))) * 100.0

#Used for folding purpose
def Left_Shift_By_One(a,length_of_array):
    temp=a[0]
    for i in range(length_of_array-1):
        a[i]=a[i+1]
    a[length_of_array-1]=temp
    return a

def main():
    
    with open('mnist_train.csv', newline='') as csv_file1:
    
        train_data_lines = csv.reader(csv_file1)
        train_dataset=list(train_data_lines) 
        train_matrix=np.array(train_dataset).astype("int")
        #prforming only on 20% of training examples
    
        x=train_matrix[0:59999,0:785]
                                        #10 fold cross validation
        #--STEPS:
        #1.shuffle training set
        #2.split it into 10 arrays
        #3.perform cross validation by using "Left_Shift_By_One" function
        #4.At each fold calculate optimal K and its associated accuracy and append it in Final_K array and max_accuracy array. 
        #5.Find Final optimal k from final optimal array with max(max_accuracy)

        np.random.shuffle(x)
        split_matrix=np.array_split(x,10)#OR (train_matrix,10)
        
        flag=0
        Final_K=[]
        Final_Accuracy=[]
         
        for i in range(len(split_matrix)):
            max_accuracy=0
            optimal_K=0
            #performing cross validation by using "Left_Shift_By_One" function
            if(flag==0):
                test_array=np.array(split_matrix[0])
                train_array=split_matrix[1:10]   
                train_final_array=np.concatenate(train_array)
                for k in range(1,11):
                    find_neighbours=[]
                    result=[]
                    predictions=[]
                    for i in range(len(test_array)):
                        find_neighbours=findNeighbours(train_final_array,test_array[i],k)
                        result = findBestNeighbour(find_neighbours)
                        predictions.append(result)
                        print('Actual Number is:' + repr(test_array[i,0])+' Predicted Number is:' + repr(result))
                    accuracy = FindAccuracy(test_array, predictions)
                    print('Accuracy: ' + repr(accuracy) + '%')
                    if(max_accuracy < accuracy):
                        max_accuracy=accuracy
                        optimal_K=k
                    else:
                        continue
                #At each fold calculating optimal K and its associated accuracy and append it in Final_K array and max_accuracy array        
                Final_K.append(optimal_K)
                Final_Accuracy.append(max_accuracy)           
            
            #performing cross validation by using "Left_Shift_By_One" function
            if(flag==1):
                test_array=[]
                train_array=[]
                a_array=[]
                Shifted_array=Left_Shift_By_One(split_matrix,len(split_matrix))
                a_array=np.array(Shifted_array).astype("int")
                test_array=np.array(a_array[0])
                train_array=a_array[1:10]
                train_final_array=np.concatenate(train_array)
                max_accuracy=0
                optimal_K=0

                for k in range(1,11):
                    find_neighbours=[]
                    result=[]
                    predictions=[]
                    accuracy=0
                    for i in range(len(test_array)):
                        find_neighbours=findNeighbours(train_final_array,test_array[i],k)
                        result = findBestNeighbour(find_neighbours)
                        predictions.append(result)
                        print('Actual Number is:' + repr(test_array[i,0])+' Predicted Number is:' + repr(result))
                    accuracy = FindAccuracy(test_array, predictions)
                    print('Accuracy: ' + repr(accuracy) + '%')
                    if(max_accuracy < accuracy):
                        max_accuracy=accuracy
                        optimal_K=k
                    else:
                        continue
                #At each fold calculating optimal K and its associated accuracy and append it in Final_K array and max_accuracy array
                Final_K.append(optimal_K)
                Final_Accuracy.append(max_accuracy)
                
            flag=1  

        #Find Final optimal k from final optimal array with max(max_accuracy)
        Accuracy_search=np.amax(Final_Accuracy)
        print("Maximum Accuracy"+repr(np.amax(Final_Accuracy)))
        Index=Final_Accuracy.index(Accuracy_search)
        k_final=Final_K[Index]
        print("Optimal K is:"+repr(k_final))

main()






























    