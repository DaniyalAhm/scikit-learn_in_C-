#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <unordered_map>
#include <string>
#include <boost/algorithm/string.hpp>

using namespace std;

struct Node {
    bool is_leaf;
    Node* left;
    Node* right; 
    pair<int, int> split; // X_train start index and ending index
    Node(bool is_leaf, Node* left = nullptr, Node* right = nullptr, pair<int, int> split) 
        : is_leaf(is_leaf), left(left), right(right), split(split) {} //? FIXED
};

class Decision_Tree {
    vector<vector<string>> X_train; // Correct type
    Eigen::VectorXd y_train;
    unordered_map<int, vector<vector<string>>> combined_data;
    int Max_depth;
    std::function<double(const vector<vector<string>>&, const pair<int, int>&, const pair<int, int>&)> Split_function;

public:
    Decision_Tree(vector<vector<string>> X_train, Eigen::VectorXd y_train, int Max_depth = -1, string Splitting_function = "Information_Gain", int Min_depth = -1, std::function<double(const std::vector<std::vector<std::string>>&, const Eigen::VectorXd&)> Split_function = Gini) { // ! Add comma and correct the Gini function definition
        this->X_train = X_train;
        this->y_train = y_train;
        combine_data();
        this->Max_depth = Max_depth;
        this->Split_function = Split_function; 
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

    bool is_pure(Node node) { 
        // Checks if the label of all the data points in the dataset are the same
        pair<int, int> thresholds = node.split;
        int class_ = y_train[thresholds.first];
        for (int i = thresholds.first; i < thresholds.second; i++) {
            if (class_ != y_train[i]) {
                return false;
            }
        }
        return true;
    };

private:
    Node* Fit_Private(Node* currentNode, int depth) { 
        if (is_pure(*currentNode)) { 
            currentNode->is_leaf = true;
            return currentNode; 
        }

        if (depth == Max_depth) {
            currentNode->is_leaf = true; 
            return currentNode; 
        } else {
           pair<pair<int, int>, pair<int, int>> best_split = Find_best_possible_split(currentNode); // ! Adjust the call to Find_best_possible_split
            Node* left_node = new Node(false, nullptr, nullptr, best_split.first); 
            Node* right_node = new Node(false, nullptr, nullptr, best_split.second);

            currentNode->left = Fit_Private(left_node, depth + 1); 
            currentNode->right = Fit_Private(right_node, depth + 1);
        }
        return currentNode; 
    }

    pair<pair<int, int>, pair<int, int>> Find_best_possible_split(Node* node) { 
        pair<int, int> thresholds = node->split; 
        int class_ = y_train[thresholds.first];
        double best_gain = 0;
        double gain = 0;
        pair<int, int> new_left_threshold;
        pair<int, int> new_right_threshold;

       pair<pair<int, int>, pair<int, int>> result; 
        for (int i = thresholds.first; i < thresholds.second; i++) {
            for (int j = 0; j < X_train[i].size(); j++) { // Iterate over all values in that feature column
                new_left_threshold = make_pair(0, j - 1);
                new_right_threshold = make_pair(j, X_train[i].size() - 1);

                gain = Split_function(X_train, new_left_threshold, new_right_threshold); 
                    best_gain = gain;
                    result.first = new_left_threshold;
                    result.second = new_right_threshold;
                }
            }
        
        return result;
};


public:
     Eigen::VectorXd predict(vector<vector<string>> X_test){
            //? Using the tree we can predict the values of our X_Test
            


    }

/*
PsuedoCode:
    Class Decision_Tree:
        Root Node; // This is to represent the entire dataset
        Splitting_function; // This is as it sounds
        Max_depth; // This is the max height of the tree
        Min_depth

        Constructor (Max_depth = -1, Splitting_function="Information_Gain", Dataset, Min_depth=-1):
            Root Node = Dataset

        Fit_on_Data(Max depth, i, node):
            if node is Pure; I.e If the remaining set of data represented by that node belongs to the same class:
                return node

            if(i == Max Depth):
                return node

            else:
                children = Find_best_possible_split(node)

                node.attach_left(left_node)
                node.attach_right(right_node)

            return node

        Predict(X_test) {}

        Find_best_possible_split(node):
            for attribute in Node:
                threshold = all_unique_vals([X:attribute]) // Select all values in that feature column
                    for threshold in thresholds:
                        Left_of_threshold // This is left values of the threshold
                        Right_of_threshold // This is the right values of the threshold

                        if len(left_of_threshold != 0) and len(Right_of_threshold != 0):
                            gain = Support_function(y, left_indices, right_indices) // Whether that be gini or info gain
                                  if gain > best_gain:
                                    best_gain = gain
                                    best_split = (feature_index, threshold)
            return best_split and best_gain
*/
};