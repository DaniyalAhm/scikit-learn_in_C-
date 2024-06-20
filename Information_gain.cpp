
#include <iostream>
#include <Eigen/Dense>

/*
Psuedo-Code
Class: Information Gain

#Information gain represents a sort of entropy in how much uncertainty is in a data set


#? To Calculate Information Gain for a particular feature in a dataset
    Calculate Entropy:
        Formula for Entropy
            n= number of class
            H(S) = overall entropy
            sum= 0
            For i from 1 to n: #For each class i 
                p_i = (len(Examples_in(Class[i])))/ Total_examples_across_all_classes
                 if p_i != 0:
                    H_S -= p_i * log2(p_i)
                return H_S

    Function CalculateInformationGain(feature_X, dataset):
  
        H_S = CalculateEntropy(dataset)
        
        # Initialize variables
        Average_Entropy = 0
        InfoGain = 0
        n = number of classes
        subsets = split_by_unique_values(feature_X, dataset)

        For each subset S_v in subsets:
                H[S_V] = CalculateEntropy(S_v)

                proportions[S_v]= len(S_v)/len(dataset)


                Average_Entropy += proportion_S_v * H_S_v



        InfoGain= H_S - Average_Entropy


        return max(InfoGain)



    Function Calculate MaxinfoGain(Remaining Unsplit_data):
        For each feature X in unsplit data:
            calculate the info gain and return the Max info gain
        






*/