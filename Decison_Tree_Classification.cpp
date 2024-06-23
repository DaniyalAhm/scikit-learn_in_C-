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
                thresold = all_unique_vals([X:attribute]) //? Select all values in that feature column
                    for threshold in tresholds:
                        Left_of_threshold //? This is left values of the threshold
                        Right_of_threshold //?  This is the right values of the threshold

                        if len(left_of_threshold !=0 ) and Right_of_threshold !=0"
                            gain = Support_function(y, left_indices, right_indices) //? Whether that be gini or info gain
                                  if gain > best_gain:
                                    best_gain = gain
                                    best_split = (feature_index, threshold)
            return best_split and best_gain

                        


*/