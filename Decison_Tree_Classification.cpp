#include <iostream>


/*

PsuedoCode:
    Class Decision_Tree:
        Root Node; //? This is to represent the entire dataset
        Splitting_function; //? This is as it sounds
        Max_depth; //? This is the max hieght of the tree
        Min_depth

        Constructor (Max_depth = -1, Spliting_function="Information_Gain", Dataset, Min_depth=-1):
            Root Node = Dataset
        

        Fit_on_Data(Max depth, i, node):
            if node is Pure; I.e If the remaining set of data represented by that node belongs to the same class :
                return node

            if(i == Max Depth):
                return node

            else:
                children=Find_best_possible_split(node)
                

                node.attach_left(left_node)
                node.attach_right(right_node)

            return node


        Predict(X_test){
    
        
        }

        
        Find_best_possible_split(node):
            for attribute in Node:
                thresold = X[:attribute]
                for 


            


        
        








*/