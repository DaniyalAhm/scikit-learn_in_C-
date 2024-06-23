#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <unordered_map>
#include <string>


struct VectorHash {
    size_t operator()(const vector<string>& vec) const {
        size_t hash = 0;
        for (const auto& str : vec) {
            hash ^= std::hash<string>()(str) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        return hash;
    }
};


class Decision_Tree{
    vector<vector<string>> X_train; // Correct type
    Eigen:: VectorXd y_train;
    unordered_map<int, vector<vector<string>>> combined_data;
    int Max_depth;



public:
    Decision_Tree(vector<vector<string>> :: X_train,  Eigen:: VectorXd y_train, int Max_depth=-1, string Splitting_function="Information_Gain", int Min_depth=-1){
        this->X_train = X_train;
        this->y_train = y_train;
        combine_data();
        this->Max_depth = Max_depth;


    }






        void combine_data(){

            for (int i =0; i<y_train.size(); ++i){
                int class_ = y_train(i);

                if(combined_data.find(class_) == combined_data.end()){
                    combined_data[class_] = vector<vector<string>>(); // Initialize vector if not present


                }
                combined_data[class_].push_back(X_train[i]);

            }

        };

    private:
        is_pure(vector<vector<string>> data){

            //Checks if the label of all the data points in the dataset are the same
            string first_label = data[0][data[0].size()-1];
            for (int i = 1; i<data.size(); ++i){
                if(data[i][data[i].size()-1] != first_label){
                    return false;
                }
            }
            return true;
        }

    public:
        void Fit(node, node_data, depth, node_y){
            if(is_pure(node)){
                return node;
            }


            if(depth == Max_depth){
                return node;
            }

            else{
                vector<vector<string>> left_node, right_node;
                vector<vector<string>> best_split = Find_best_possible_split(node_data, node_y);
                left_node = best_split[0];
                right_node = best_split[1];
                node.attach_left(Fit(left_node, depth+1,  node_y));
                node.attach_right(Fit(right_node, depth+1, node_y));
            }

        }


        vector<vector<string>> Find_best_possible_split(node_data,node_y){


            for(auto attribute )                



        }



}












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