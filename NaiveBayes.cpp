#include <iostream>
#include <Eigen/Dense>


class MultinomialBayes{
    /*
    Psuedo-Code:
        

        Function_PreProcess data(data):
            Split each word by " " and turn into an array
            Data = now a list of words 


        Function_ Compute Individual Probablity
            Hashtable = {}

            For every word in Data:
                Hashtable[word]= total_number of occurances/total words


        Function_ Compute Conditional Probability:
            n= length of Data Array
            Matrix = nxn length matrix
            Set = set(data)

            For every word i in Data: 
                For every word j in Data:
                    Matrix[i][j]= Total_Occurances(i,j)



                Matrix[i][j]= Matrix[i][j]/sum(data)+len(set)


    */














};