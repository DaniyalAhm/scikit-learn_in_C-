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
    string class_;
    Node(bool is_leaf, Node* left = nullptr, Node* right = nullptr, pair<int, int> split, string class_= "None" ) 
        : is_leaf(is_leaf), left(left), right(right), split(split), class_(class_){} //? FIXED
};

class Decision_Tree {
    vector<vector<string>> X_train; // Correct type
    Eigen::VectorXd y_train;
    unordered_map<int, vector<vector<string>>> combined_data;
    int Max_depth;
    Node* Tree; 
    string spliting_fuction = "Gini";
public:
    Decision_Tree(vector<vector<string>> X_train, Eigen::VectorXd y_train, int Max_depth = -1, string Splitting_function = "Information_Gain", int Min_depth = -1, string spliting_fuction= "Gini" ) { // ! Add comma and correct the Gini function definition
        this->X_train = X_train;
        this->y_train = y_train;
        combine_data();
        this->Max_depth = Max_depth;
        this->spliting_fuction = spliting_fuction;
        Node Tree = Node(false, nullptr, nullptr, pair<int,int>(0, X_train.size())); // ! This redeclares Tree instead of initializing the class member
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
            currentNode->class_= y_train[currentNode->split.first]; // ! y_train is Eigen::VectorXd, should convert to string
            return currentNode; 
        }

        if (depth == Max_depth) {
            currentNode->is_leaf = true; 
            currentNode->class_ = Majority_vote(currentNode);
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
                new_left_threshold = make_pair(thresholds.first, i-1);
                new_right_threshold = make_pair(j, thresholds.second);

                gain = split_gain(X_train, new_left_threshold, new_right_threshold); 
                if (gain > best_gain) { // ! This line should be inside the if block
                    best_gain = gain;
                    result.first = new_left_threshold;
                    result.second = new_right_threshold;
                }
            }
        }
        return result;
    }



private:
    double split_gain(vector<vector<string>> X_train, pair<int, int> left_threshold, pair<int, int> right_threshold) {
        if(spliting_fuction == "Gini"){
            return Gini(X_train, left_threshold, right_threshold);
        } 
        if(spliting_fuction == 'information_gain'){
            return Information_Gain(X_train, left_threshold, right_threshold);
        }

    }


    double calculate_entropy( pair<int, int> threshold ){

        unordered_map<int, int> class_counts;
        for (int i = threshold.first; i < threshold.second; i++) {
            class_counts[y_train[i]]= 0;
        }

        for (int i = threshold.first; i < threshold.second; i++) {
            class_counts[y_train[i]]++;
        }

        double entropy = 0;
        for (auto& count : class_counts) {
            double p_i = count.second / (threshold.second - threshold.first);
            if (p_i != 0) {
                entropy -= p_i * log2(p_i);
            }
        }
        return entropy;
    }


    double information_gain(vector<vector<string>> data, pair<int, int> left_threshold, pair<int, int> right_threshold){
        //!FIX THIS


        // Calculate the entropy of the entire dataset
        double H_S = calculate_entropy(0, make_pair(0, y_train.size()));

        // Calculate the entropy of the left and right datasets
        double H_S_left = calculate_entropy(0, left_threshold);
        double H_S_right = calculate_entropy(0, right_threshold);

        // Calculate the average entropy
        double average_entropy = (left_threshold.second - left_threshold.first) * H_S_left + (right_threshold.second - right_threshold.first) * H_S_right;

        // Calculate the information gain
        return H_S - average_entropy;
    }

    double Gini(vector<vector<string>> data, pair<int, int> left_threshold, pair<int, int> right_threshold){
        // Gini_index= 0 
        //? Making Class Counts
        unordered_map<int, int> left_class_counts;
        unordered_map<int, int> right_class_counts;

        for (int i = left_threshold.first; i < left_threshold.second; i++) {
            left_class_counts[y_train[i]]= 0;
        }

        
        for (int i = left_threshold.first; i < left_threshold.second; i++) {
            left_class_counts[y_train[i]]++;
        }

        
        for (int i = right_threshold.first; i < right_threshold.second; i++) {
            right_class_counts[y_train[i]]= 0;
        }

        
        for (int i = right_threshold.first; i < right_threshold.second; i++) {
            right_class_counts[y_train[i]]++;
        }
        
        double left_gini = 0;
        double right_gini = 0;
        double total_gini = 0;

        for (auto& count : left_class_counts) {
            left_gini += pow((count.second/left_threshold.second - left_threshold.first), 2);
        }
        for (auto& count : right_class_counts) {
            right_gini += pow((count.second/right_threshold.second - right_threshold.first), 2);
        }

        return 1.0 - (left_gini + right_gini)

    };

public:
     Eigen::VectorXd predict(vector<vector<string>> X_test){
            // Using the tree we can predict the values of our X_Test
            Eigen :: VectorXd y_test(X_test.size());

            for (int i = 0; i < X_test.size(); i++){
                y_test[i] = stod(predict_recursive(X_test[i], &Tree)); 
            }
            return y_test; 
    };

    string Majority_vote(Node* node) {
        unordered_map<int, int> class_count;
        pair<int, int> thresholds = node->split;
        for (int i = thresholds.first; i < thresholds.second; i++) {
            class_count[y_train[i]]++;
        }
        int max_count = 0;
        int majority_class = -1;
        for (auto& count : class_count) {
            if (count.second > max_count) {
                max_count = count.second;
                majority_class = count.first;
            }
        }
        return to_string(majority_class); // Convert to string as your Node class uses string
    }

private:
     string predict_recursive(vector<string> data, Node* currentNode){ // ! Return type should be string
        // Using the tree we can predict the values of our X_Test
        if(currentNode->is_leaf){
            return currentNode->class_;
        }

        int feature_index = currentNode->split.first;
        double threshold = currentNode->split.second;
        if (stod(data[feature_index]) < threshold) {
            return predict_recursive(data, currentNode->left);
        } else {
            return predict_recursive(data, currentNode->right);
        }
    }
};





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