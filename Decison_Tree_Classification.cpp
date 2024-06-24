#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <unordered_map>
#include <string>
#include <boost/algorithm/string.hpp>

using namespace std;

struct Node {
    bool is_leaf;
    Node* parent;
    pair<int, int> split; // X_train start index and ending index
    Node(bool is_leaf, Node* parent = nullptr,    pair<int, int> split) 
        : is_leaf(is_leaf), parent(parent), pair(pair) {}
};



class Decision_Tree {
    vector<vector<string>> X_train; // Correct type
    Eigen::VectorXd y_train;
    unordered_map<int, vector<vector<string>>> combined_data;
    int Max_depth;

public:
    Decision_Tree(vector<vector<string>> X_train, Eigen::VectorXd y_train, int Max_depth = -1, string Splitting_function = "Information_Gain", int Min_depth = -1) {
        this->X_train = X_train;
        this->y_train = y_train;
        combine_data();
        this->Max_depth = Max_depth;
    }

private:
    vector<vector<string>> preprocess_text(vector<string> Raw_data) {
        vector<vector<string>> result;
        for (const auto& doc : Raw_data) {
            vector<string> splitstring;
            boost::algorithm::split(splitstring, doc, boost::is_any_of(" "));
            result.push_back(splitstring);
        }
        return result;
    }

    void combine_data() {
        for (int i = 0; i < y_train.size(); ++i) {
            int class_ = y_train(i);
            if (combined_data.find(class_) == combined_data.end()) {
                combined_data[class_] = vector<vector<string>>(); // Initialize vector if not present
            }
            combined_data[class_].push_back(X_train[i]);
        }
    }

    bool is_pure(Node node) { // ! Function should return bool
        // Checks if the label of all the data points in the dataset are the same
        pair<int, int> thresholds=node.split;
        int class_ = y_train[thresholds.first];
        for (int i = thresholds.first; i < thresholds.second;i++ ){
                if(class_ != y_train[i]){
                    return false;
                }
            return true;

        };



public:
    Node Fit_Private(node, int depth) {
        if (is_pure(node)) { // ! Function `is_pure` expects data, not a node
            return node; // ! `return node;` does not match the return type `void`
        }

        if (depth == Max_depth) {
            return node; // ! `return node;` does not match the return type `void`
        } else {

            

            vector<vector<string>> best_split = Find_best_possible_split(, );
            left_node = best_split[0]; // ! This assumes `best_split` is a 2D vector of strings, needs to be adjusted
            right_node = best_split[1];


            node.attach_left(Fit(left_node, depth + 1, node_y)); // ! `attach_left` is not defined, nor is `node`
            node.attach_right(Fit(right_node, depth + 1, node_y)); // ! `attach_right` is not defined, nor is `node`
        }
    }







    vector<Node*, Node*> Find_best_possible_split(node, bool is_leaf) { // ! Missing parameter types
        pair<int, int> thresholds=node.split;
        int class_ = y_train[thresholds.first];
        int best_gain =0;
        int gain = 0;
        pair <int, int> new_left_threshold;
        pair <int, int> new_right_threshold;

        for (int i = thresholds.first; i < thresholds.second;i++ ){
            for (int j = 0; j< X_train[i].size(); j++){ //? All the values in that feature column
                new_left_threshold= <0, j-1>;
                new_right_threshold = <j, X_train[i].size()-1;>



                //!Add Support function...



                }
                    

                

                                

            }

        }








};


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

