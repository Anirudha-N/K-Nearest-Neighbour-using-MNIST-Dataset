# K-Nearest-Neighbour-using-MNIST-Dataset:

This repository consists: 

1.KNN code using python.

2.Finding optimal K using 10-fold cross validation.

3.Sliding Window approach to increase accuracy.

Prerequisite:You will need MNSIT training data and MNSIT testing data in .csv format.In these codes I used "mnist_training.csv" and "mnist_test.csv".

-------------------------------------------------------------------------------------------------------------------------

# Information about KNN and MNIST Dataset:

# K-Nearest Neighbour:

1.K-nearest neighbours is a non-parametric algorithm. It is used for classification and
regression problems.

2.To categories test instance into a particular type, algorithm finds K- nearest neighbours
of a test instance from training examples.

3.It uses various cost functions like Euclidean Distance, Sliding window, etc. to calculate
minimum distance between test instance and training examples.

4. Type of training example is assigned to test instance if distance between them is
minimum among all examples.

# MNIST Dataset:

1.MNIST database is a database of handwritten digits.

2.It consists 60000 training examples and 10000 test examples.

3.Each image has 785 labels. The first label represents Digit and 784 labels represent pixel
intensities of Digit’s image.

-------------------------------------------------------------------------------------------------------------------------

# K-NN Pseudo Code:

CalculateEuclideanDistance(testInstance, final_train_matrix, len(testInstance)):

{

    #distance += (testInstance[i]-final_train_matrix[i])^2
    
    #CalculateEuclideanDistance=square_root_of(distance)
}

findNeighbours(train dataset , one row of test dataset , k):

{
  
    #Finding Euclidean Distances of testInstance w.r.t all train matrix rows
       
       CalculateEuclideanDistance(testInstance, final_train_matrix, len(testInstance))
    
    #Select K minimum distances as K neighbors
}

findBestNeighbour(neighbors):

{

    #Finding neighbor with maximum occurrence in a list given by findNeighbours() function.

    #Select neighbour with maximum occurance as best neighbour.

}

main()
{

    #Load train data and test data from “mnist_train.csv” and “mnist_test.csv” respectively.

    for (int i=0 to i=9999)
    
    {
    
      #Finding "k" neighbors :
      
            find_neighbours= findNeighbours(train dataset , one row of test
            
            dataset(testInstance),k)
      
      #Choosing best neighbor among K neighbors:
            
            findBestNeighbour(neighbors)
    
    }
    
    #Finding the accuracy, where accuracy= (true_positives/total Number of test examples)*100

}

-------------------------------------------------------------------------------------------------------------------------

# K-NN output:

Accuracy: 96.91%

-------------------------------------------------------------------------------------------------------------------------

# Cross-Validation Pseudo Code:

CalculateEuclideanDistance(testInstance, final_train_matrix, len(testInstance)):

{

    #distance += (testInstance[i]-final_train_matrix[i])^2
    
    #CalculateEuclideanDistance=square_root_of(distance)

}

findNeighbours(train dataset , one row of test dataset , k):

{

    #Finding Euclidean Distances of testInstance w.r.t all train matrix rows
      
          CalculateEuclideanDistance(testInstance, final_train_matrix, len(testInstance))

    #Select K minimum distances as K neighbors

}

findBestNeighbour(neighbors):

{

    #Finding neighbor with maximum occurrence in a list given by findNeighbours() function.
    
    #Select neighbour with maximum occurance as best neighbour.

}

Left_Shift_By_One(array, length_of_array):

{

    temp=first element of array
    
    for i in range(length_of_array-1):
    
          array[i]=array[i+1]

    array[length_of_array-1]=temp

}

main()

{

    #Load train data from “mnist_train.csv”.

    #10 fold cross validation
   
    #--STEPS:
    
        #1.shuffle training set.
        
        #2.Split it into 10 arrays.
        
        #3.Perform cross validation by using "Left_Shift_By_One" function.
        
        #4.At each fold calculate optimal K and its associated accuracy and append it in
         
              Final_K array and max_accuracy array.

        #5.Find Final optimal k from final optimal array with max(max_accuracy).

}

-------------------------------------------------------------------------------------------------------------------------

# Cross Validation Output:

Optimal K=3

Applying optimal K to classify all the images in MNIST test set on original MNIST.

Accuracy=97.38


-------------------------------------------------------------------------------------------------------------------------

# Sliding Window Pseudo Code

CalculateEuclideanDistance(testInstance, final_train_matrix, len(testInstance)):

{

    #distance += (testInstance[i]-final_train_matrix[i])^2

    #CalculateEuclideanDistance=square_root_of(distance)

}

slidingWindow(final_train_matrix, testInstance):

{

    #creating window of 28X28 from 30X30 train matrix.(You will get 9 "28X28" windows)
    
    #Find distance between testInstance and each window
    
    #select min distance among 9 distances
}

findBestNeighbour(neighbors):

{

    #Finding neighbor with maximum occurrence in a list given by findNeighbours() function.

    #Select neighbour with maximum occurance as best neighbour.

}

imagePadding(Input_Matrix):

{

    #Appending two columns of 0's at the end of 28X28 column matrix now matrix is of 28X30

    #appending row's with zeros at the end of Matrix

    #Shifting rows according to index

}

main()

{

    #Load train data and test data from “mnist_train.csv” and “mnist_test.csv” respectively.

    #Making 28X28 train matrix image into 30X30 using padding
  
              imagePadding(Input_Matrix)
              
    #Function for finding 9 distances by using sliding window of 28X28 on 30X30 matrix.Take minimum of nine distances   

              dist=slidingWindow(final_train_matrix, testInstance)
    
    #Sorting Neighbours_distances with ascending order
    
    #Choose first "K" distances
    
    #Choosing best neighbor among K neighbors:
            
            findBestNeighbour(neighbors):
    
    #Finding the accuracy, where accuracy= (true_positives/total Number of test examples)*100

}

-------------------------------------------------------------------------------------------------------------------------

# Sliding Window Output:

Accuracy: 98.26

-------------------------------------------------------------------------------------------------------------------------


