#ifndef DISTANCES_H
#define DISTANCES_H

#include <Eigen/Dense>

// Function to calculate Euclidean distance between two vectors
double euclidean_distance(const Eigen::VectorXd& v1, const Eigen::VectorXd& v2);

// Function to calculate Manhattan distance between two vectors
double manhattan_distance(const Eigen::VectorXd& v1, const Eigen::VectorXd& v2);

// Function to calculate Minkowski distance between two vectors
double minkowski_distance(const Eigen::VectorXd& v1, const Eigen::VectorXd& v2, int p);

// Function to calculate Cosine similarity between two vectors
double cosine_similarity(const Eigen::VectorXd& v1, const Eigen::VectorXd& v2);

#endif // DISTANCES_H
