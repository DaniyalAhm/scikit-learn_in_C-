//? This is beginging of Linear Regression in C++.

#include <iostream>
#include <Eigen/Dense>

Eigen::VectorXd LinearRegression(const Eigen::MatrixXd& X, const Eigen::VectorXd& y){
    Eigen::MatrixXd XtX = X.transpose() * X;

if (XtX.determinant() == 0) {
    std::cerr << "Matrix is singular and cannot be inverted, Consider using a PseudoInverse matrix" << std::endl;
    return Eigen::VectorXd();  // Return an empty matrix or handle differently
}


    Eigen::MatrixXd transpose_X = X.transpose();

    Eigen::VectorXd beta_hat = (transpose_X * X).inverse() * transpose_X * y;  

    return beta_hat; 
}


Eigen::MatrixXd LinearRegressionMultivariate(const Eigen::MatrixXd& X, const Eigen::MatrixXd& y){
Eigen::MatrixXd XtX = X.transpose() * X;

if (XtX.determinant() == 0) {
    std::cerr << "Matrix is singular and cannot be inverted, Consider using a PsuedoInverse matrix" << std::endl;
    return Eigen::MatrixXd();  // Return an empty matrix or handle differently
}
    Eigen::MatrixXd transpose_X = X.transpose();
    Eigen::MatrixXd beta_hat = (transpose_X * X).inverse() * transpose_X * y;  

    return beta_hat;  
}



double evaluate_singular(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, const Eigen::VectorXd& beta_hat){
    //Calcuate y_hat
    Eigen::VectorXd y_hat = X * beta_hat;

    //Calculate residuals
    Eigen::VectorXd residuals = (y-y_hat)  ;


    //Calcuate the mean squared error
    double mean_sqaured_error = residuals.squaredNorm() / residuals.size();
    return mean_sqaured_error;

}

double evaluate_multivariate(const Eigen::MatrixXd& X, const Eigen::MatrixXd& y, const Eigen::MatrixXd& beta_hat){
    //Calcuate y_hat
    Eigen::MatrixXd y_hat = X * beta_hat;

    //Calculate residuals
    Eigen::MatrixXd residuals = (y-y_hat)  ;


    //Calcuate the mean squared error
    double mean_sqaured_error = residuals.squaredNorm() / residuals.size();


    return mean_sqaured_error;
}
int main() {
    // Example data
    Eigen::MatrixXd X(4, 2);
    X << 1, 1,
         1, 2,
         1, 3,
         1, 4;  // Including intercept term
    Eigen::VectorXd y(4);
    y << 2, 4, 6, 8;
    Eigen::VectorXd beta(2);
    beta << 0.5, 1.5;  // Example parameters

    double cost = evaluate_singular(X, y, beta);
    std::cout << "Cost: " << cost << std::endl;

    return 0;
}