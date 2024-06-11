#include <iostream>
#include <Eigen/Dense>
#include <cmath>
//? Implementing Distances


using namespace std;
double euclidean_distance(const Eigen::VectorXd& v1 , const Eigen::VectorXd& v2 ){
    if (v1.size() != v2.size()){


    std::cerr << "Vector Size Mismatch" << std::endl;
    return -1;  // Return an empty matrix or handle differently
    }
    Eigen::VectorXd difference= (v1-v2);

    return sqrt(difference.squaredNorm());

    /*
    This is a more effiecent way of doing the following 
    double euclidean_distance_manual(const Eigen::VectorXd& v1, const Eigen::VectorXd& v2) {
    Eigen::VectorXd diff = v1 - v2;
    double sum_of_squares = diff.array().square().sum();
    return std::sqrt(sum_of_squares);
}
    */

}

double manhattan_distance(const Eigen::VectorXd& v1 , const Eigen::VectorXd& v2){
    if (v1.size() != v2.size()){


    std::cerr << "Vector Size Mismatch" << std::endl;
    return -1;  // Return an empty matrix or handle differently
    }



    Eigen::VectorXd difference= (v1-v2);
    return difference.cwiseAbs().sum();
}

double minkowski_distance(const Eigen::VectorXd& v1 , const Eigen::VectorXd& v2, int p){
    if (v1.size() != v2.size()){


    std::cerr << "Vector Size Mismatch" << std::endl;
    return -1;  // Return an empty matrix or handle differently
    }
    double sum = 0;

    for (int i = 0; i<v1.size();i++){
        sum+= pow((v1[i]-v2[i]),p);

    }

    return pow(sum, (1.0/p));



}

double cosine_similarity(const Eigen::VectorXd& v1 , const Eigen::VectorXd& v2){
    



    return (v1.dot(v2))/(v1.norm() * v2.norm());


}