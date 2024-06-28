#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <unordered_map>
#include <string>
#include <boost/algorithm/string.hpp>
#include <random>

using namespace std;
using namespace Eigen;

class RandomForest {
public:
    vector<vector<string>> X_train;
    Eigen::MatrixXd y_train;
    int num_trees;
    vector<vector<vector<string>>> bootstrapped_samples;
    vector<DecisionTreeClassifier> trees;
    vector<vector<int>> trees_random_features;
    string type;
    vector<string> classes;
    unordered_map<string, int> class_map;

    RandomForest(Eigen::MatrixXd X_train, Eigen::MatrixXd y_train, int num_trees = 1000, string type = "Classification") {
        this->X_train = X_train;
        this->y_train = y_train;
        this->num_trees = num_trees;
        this->type = type;
        bootstrap();
        train();
    }

private:
    void bootstrap() {
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<> dis(0, X_train.size() - 1);

        for(int i = 0; i < num_trees; i++) {
            vector<vector<string>> current_sample;
            for(int j = 0; j < X_train.size(); j++) {
                current_sample.push_back(X_train[dis(gen)]);
            }
            bootstrapped_samples.push_back(current_sample);
        }
    }

public:
    void train() {
        int num_features = X_train[0].size();
        if (type == "Classification") {
            for (int i = 0; i < num_trees; i++) {
                vector<int> random_features;
                for (int k = 0; k < int(sqrt(num_features)); k++) {
                    random_features.push_back(rand() % num_features);
                }
                vector<vector<string>> current_samples;
                for (int j = 0; j < X_train.size(); j++) {
                    vector<string> sample;
                    for (int k = 0; k < random_features.size(); k++) {
                        sample.push_back(bootstrapped_samples[i][j][random_features[k]]);
                    }
                    current_samples.push_back(sample);
                }
                DecisionTreeClassifier tree;
                tree.fit(current_samples, y_train);
                trees.push_back(tree);
                trees_random_features.push_back(random_features);
            }
        } else {
            for (int i = 0; i < num_trees; i++) {
                vector<int> random_features;
                for (int k = 0; k < int(num_features / 3); k++) {
                    random_features.push_back(rand() % num_features);
                }
                vector<vector<string>> current_samples;
                for (int j = 0; j < X_train.size(); j++) {
                    vector<string> sample;
                    for (int k = 0; k < random_features.size(); k++) {
                        sample.push_back(bootstrapped_samples[i][j][random_features[k]]);
                    }
                    current_samples.push_back(sample);
                }
                DecisionTreeRegressor tree;
                tree.fit(current_samples, y_train); 
                trees.push_back(tree);
                trees_random_features.push_back(random_features);
            }
        }
    }

    int calculateMode(vector<string>& vec) {
        unordered_map<string, int> frequencyMap;

        for (const string& str : vec) {
            frequencyMap[str]++;
        }

        string mode = vec[0];
        int maxCount = 0;

        for (const auto& pair : frequencyMap) {
            if (pair.second > maxCount) {
                maxCount = pair.second;
                mode = pair.first;
            }
        }

        return mode; 
    }

    vector<string> predict(vector<vector<string>> X_test) {
        if (type == "Classification") {
            return predict_classification(X_test);
        } else {
            return predict_regression(X_test);
        }
    }

private:
    vector<string> predict_classification(vector<vector<string>> X_test) {
        vector<string> predictions;
        for (int i = 0; i < X_test.size(); i++) {
            vector<string> predictions_by_tree;
            for (int j = 0; j < num_trees; j++) {
                vector<string> sample;
                for (int k = 0; k < trees_random_features[j].size(); k++) {
                    sample.push_back(X_test[i][trees_random_features[j][k]]);
                }
                predictions_by_tree.push_back(trees[j].predict(sample));
            }
            predictions.push_back(calculateMode(predictions_by_tree)); 
        }
        return predictions;
    }

    vector<string> predict_regression(vector<vector<string>> X_test) {
        vector<string> predictions;
        for (int i = 0; i < X_test.size(); i++) {
            vector<double> predictions_by_tree;
            for (int j = 0; j < num_trees; j++) {
                vector<string> sample;
                for (int k = 0; k < trees_random_features[j].size(); k++) {
                    sample.push_back(X_test[i][trees_random_features[j][k]]); //? Predict on the same random features we used for training
                }
                predictions_by_tree.push_back(stod(trees[j].predict(sample))); 
            }
            predictions.push_back(to_string(calculateMean(predictions_by_tree)));
        }
        return predictions;
    }

    double calculateMean(const vector<double>& vec) {
        double sum = 0;
        for (double val : vec) {
            sum += val;
        }
        return sum / vec.size();
    }
};

int main() {
    Eigen::MatrixXd X_train; 
    Eigen::MatrixXd y_train;
    RandomForest rf(X_train, y_train);
    vector<vector<string>> X_test;
    vector<string> predictions = rf.predict(X_test);
    for (const string& pred : predictions) {
        cout << pred << endl;
    }
    return 0;
}
