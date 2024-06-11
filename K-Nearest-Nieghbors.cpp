#include <iostream>
#include <Eigen/Dense>
#include "Distances.h"
#include <cctype>
#include <string>
#include <Distances.h>
#include <cstdlib>
#include <ctime>

using namespace std;

class KNN {
    Eigen::MatrixXd X_train;
    Eigen::VectorXd y_train;
    int k;

public:
    KNN(const Eigen::MatrixXd& X_train, const Eigen::VectorXd& y_train, int k) {
        this->X_train = X_train;
        this->y_train = y_train;
        this->k = k;
        std::srand(std::time(0)); // !To ensure the numbers are truly random
    }

    Eigen::VectorXd predict(const Eigen::MatrixXd& X_test, const string& Type = "Classification", const string& distance = "euclidean", int p = 1) {
        if (X_train.cols() != X_test.size()) {
            std::cerr << "Given Matrix different in Dimensions from fit data" << std::endl;
            return Eigen::VectorXd();  
        }

        std::string lowerStr = Type;
        std::transform(lowerStr.begin(), lowerStr.end(), lowerStr.begin(), ::tolower);

        if (lowerStr != "classification" && lowerStr != "regression") {
            std::cerr << "Error: Only Types available are Regression and Classification" << std::endl;
            return Eigen::VectorXd();  
        }

        Eigen::VectorXd y_test(X_test.rows());

        for (int i = 0; i < X_test.rows(); ++i) {
            Eigen::VectorXd k_array(k);
            Eigen::VectorXd k_array_y(k);

            for (int j = 0; j < X_train.rows(); ++j) {
                // EACH ROW IS A DATA POINT
                double dist;
                if (distance == "euclidean") {
                    dist = euclidean_distance(X_test.row(i).transpose(), X_train.row(j).transpose());
                } else if (distance == "manhattan") {
                    dist = manhattan_distance(X_test.row(i).transpose(), X_train.row(j).transpose());
                } else if (distance == "minkowski") {
                    dist = minkowski_distance(X_test.row(i).transpose(), X_train.row(j).transpose(), p); // Example with p=3
                } else if (distance == "cosine") {
                    //To get the cosine distance
                    dist = 1 - cosine_similarity(X_test.row(i).transpose(), X_train.row(j).transpose());
                } else {
                    std::cerr << "Unknown distance metric: " << distance << std::endl;
                    return Eigen::VectorXd();
                }
                if (dist < k_array.maxCoeff()) {
                    int max_index = 0;
                    double max_value = k_array.maxCoeff(&max_index); 
                    k_array(max_index) = dist;
                    k_array_y(max_index) = y_train(j);
                }
            }

            if (lowerStr == "classification") {
                y_test(i) = majority_vote(k_array_y);

            } else {
                y_test(i) = k_array_y.mean();

            }
        }
        return y_test;
    }

    int majority_vote(Eigen::VectorXd values) {
        std::unordered_map<int, int> counts;

        for (int z = 0; z < values.size(); z++) {
            counts[values(z)]++;
        }

        int most_common = -1;
        int max_count = 0;

        for (const auto& pair : counts) {
            if (pair.second > max_count) {
                max_count = pair.second;
                most_common = pair.first;
            }
        }

        int count_ties = 0;
        std::vector<int> vec;
        vec.push_back(most_common);
        for (const auto& pair : counts) {
            if (pair.second == max_count && most_common != pair.first) {
                count_ties++;
                vec.push_back(pair.first);
            }
        }

        if (vec.size() > 1) {
            int randomIndex = std::rand() % vec.size();
            return vec[randomIndex];
        }

        return most_common;
    }
};
