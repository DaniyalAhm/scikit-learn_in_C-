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
    double value; 
    Node(bool is_leaf, Node* left = nullptr, Node* right = nullptr, pair<int, int> split = {-1, -1}, double value = 0.0) 
        : is_leaf(is_leaf), left(left), right(right), split(split), value(value) {} //? FIXED
};

class Decision_Tree {
    vector<vector<string>> X_train; // Correct type
    Eigen::VectorXd y_train;
    unordered_map<int, vector<vector<string>>> combined_data;
    int Max_depth;
    Node* Tree; 
public:
    Decision_Tree(const vector<vector<string>>& X_train, const Eigen::VectorXd& y_train, int Max_depth = -1, string Splitting_function = "Information_Gain", int Min_depth = -1) { 
        this->y_train = y_train;
        combine_data();
        this->Max_depth = Max_depth;
        Tree = new Node(false, nullptr, nullptr, pair<int, int>(0, X_train.size()));
    }



    void combine_data() {
        for (int i = 0; i < y_train.size(); ++i) {
            double value = y_train(i);
            if (combined_data.find(value) == combined_data.end()) {
                combined_data[value] = vector<vector<string>>(); // Initialize vector if not present
            }
            combined_data[value].push_back(X_train[i]);
        }
    }

    bool is_pure(Node node) { 
        // Checks if the label of all the data points in the dataset are the same
        pair<int, int> thresholds = node.split;
        double value = y_train[thresholds.first]; 
        for (int i = thresholds.first; i < thresholds.second; i++) {
            if (value != y_train[i]) {
                return false;
            }
        }
        return true;
    }

private:
    Node* Fit_Private(Node* currentNode, int depth) { 
        if (is_pure(*currentNode)) { 
            currentNode->is_leaf = true;
            currentNode->value = y_train[currentNode->split.first]; 
            return currentNode; 
        }

        if (depth == Max_depth) {
            currentNode->is_leaf = true; 

            //?Calculate the value of the current node
            currentNode->value = calculate_mean(currentNode); // ! Use calculate_mean for regression
            return currentNode; 
        } else {
            pair<pair<int, int>, pair<int, int>> best_split = Find_best_possible_split(currentNode); 
            Node* left_node = new Node(false, nullptr, nullptr, best_split.first); 
            Node* right_node = new Node(false, nullptr, nullptr, best_split.second);

            currentNode->left = Fit_Private(left_node, depth + 1); 
            currentNode->right = Fit_Private(right_node, depth + 1);
        }
        return currentNode; 
    }

    pair<pair<int, int>, pair<int, int>> Find_best_possible_split(Node* node) { 
        pair<int, int> thresholds = node->split; 
        double best_gain = 0;
        double gain = 0;
        pair<int, int> new_left_threshold;
        pair<int, int> new_right_threshold;

        pair<pair<int, int>, pair<int, int>> result; 
        for (int i = thresholds.first; i < thresholds.second; i++) {
            for (int j = 0; j < X_train[i].size(); j++) { 
                new_left_threshold = make_pair(thresholds.first, i - 1);
                new_right_threshold = make_pair(i, thresholds.second);


                //? We use the mean square error to calculate the average distance between the two thresholds
                //? the lower the mse, the better the split because that means it is more specific to the data

                gain = split_gain(X_train, new_left_threshold, new_right_threshold); 
                if (gain > best_gain) {
                    best_gain = gain;
                    result.first = new_left_threshold;
                    result.second = new_right_threshold;
                }
            }
        }
        return result;
    }

private:
    double calculate_mean(Node* node) { 
        pair<int, int> thresholds = node->split;
        double sum = 0.0;
        for (int i = thresholds.first; i < thresholds.second; i++) {
            sum += y_train[i];
        }
        return sum / (thresholds.second - thresholds.first);
    }

    double calculate_mse(const pair<int, int>& threshold) {
        double mean = 0;
        for (int i = threshold.first; i < threshold.second; i++) {
            mean += y_train[i];
        }
        mean /= (threshold.second - threshold.first);

        double mse = 0;
        for (int i = threshold.first; i < threshold.second; i++) {
            mse += pow(y_train[i] - mean, 2);
        }
        return mse;
    }

    double split_gain(const vector<vector<string>>& X_train, const pair<int, int>& left_threshold, const pair<int, int>& right_threshold) {
        double left_mse = calculate_mse(left_threshold);
        double right_mse = calculate_mse(right_threshold);
        double total_mse = left_mse * (left_threshold.second - left_threshold.first) + right_mse * (right_threshold.second - right_threshold.first);
        return -total_mse; // We aim to minimize MSE
    }

    double calculate_entropy(const pair<int, int>& threshold) {
        unordered_map<int, int> valuecounts;
        for (int i = threshold.first; i < threshold.second; i++) {
            valuecounts[y_train[i]] = 0;
        }

        for (int i = threshold.first; i < threshold.second; i++) {
            valuecounts[y_train[i]]++;
        }

        double entropy = 0;
        for (auto& count : valuecounts) {
            double p_i = static_cast<double>(count.second) / (threshold.second - threshold.first);
            if (p_i != 0) {
                entropy -= p_i * log2(p_i);
            }
        }
        return entropy;
    }

    double Information_Gain(const vector<vector<string>>& data, const pair<int, int>& left_threshold, const pair<int, int>& right_threshold) {
        double H_S = calculate_entropy(pair<int, int>(0, y_train.size()));

        double H_S_left = calculate_entropy(left_threshold);
        double H_S_right = calculate_entropy(right_threshold);

        double average_entropy = (left_threshold.second - left_threshold.first) * H_S_left + (right_threshold.second - right_threshold.first) * H_S_right;

        return H_S - average_entropy;
    }

    double Gini(const vector<vector<string>>& data, const pair<int, int>& left_threshold, const pair<int, int>& right_threshold) {
        unordered_map<int, int> left_valuecounts;
        unordered_map<int, int> right_valuecounts;

        for (int i = left_threshold.first; i < left_threshold.second; i++) {
            left_valuecounts[y_train[i]] = 0;
        }

        for (int i = left_threshold.first; i < left_threshold.second; i++) {
            left_valuecounts[y_train[i]]++;
        }

        for (int i = right_threshold.first; i < right_threshold.second; i++) {
            right_valuecounts[y_train[i]] = 0;
        }

        for (int i = right_threshold.first; i < right_threshold.second; i++) {
            right_valuecounts[y_train[i]]++;
        }

        double left_gini = 0;
        double right_gini = 0;

        for (auto& count : left_valuecounts) {
            left_gini += pow(static_cast<double>(count.second) / (left_threshold.second - left_threshold.first), 2);
        }
        for (auto& count : right_valuecounts) {
            right_gini += pow(static_cast<double>(count.second) / (right_threshold.second - right_threshold.first), 2);
        }

        return 1.0 - (left_gini + right_gini);
    }

public:
    Eigen::VectorXd predict(const vector<vector<string>>& X_test) {
        Eigen::VectorXd y_test(X_test.size());

        for (int i = 0; i < X_test.size(); i++) {
            y_test[i] = predict_recursive(X_test[i], Tree);
        }
        return y_test; 
    }

private:
    double predict_recursive(const vector<string>& data, Node* currentNode) { 
        if (currentNode->is_leaf) {
            return currentNode->value;
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
            return best_split and best gain
*/
