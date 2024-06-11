#include <iostream>
#include <Eigen/Dense>

int main() {
    // Example vectors
    Eigen::VectorXd v1(3);
    v1 << 1, 2, 3;
    Eigen::VectorXd v2(3);
    v2 << 4, 5, 6;

    // Testing Euclidean distance
    std::cout << "Euclidean distance: " << euclidean_distance(v1, v2) << std::endl;

    // Testing Manhattan distance
    std::cout << "Manhattan distance: " << manhattan_distance(v1, v2) << std::endl;

    // Testing Minkowski distance with p = 3
    std::cout << "Minkowski distance (p=3): " << minkowski_distance(v1, v2, 3) << std::endl;

    // Testing Cosine similarity
    std::cout << "Cosine similarity: " << cosine_similarity(v1, v2) << std::endl;

    return 0;
}
