//? This is beginging of Linear Regression in C++.

#include <iostream>
#include <Eigen/Dense>

class Linear_Regression_Singular{
    Eigen::MatrixXd X_train;
    Eigen::VectorXd y_train;

    public:
    Linear_Regression_Singular(const Eigen::MatrixXd& X_train, const Eigen::VectorXd& y_train) {
        this->X_train = X_train;
        this->y_train = y_train;
        
            }

    Eigen::VectorXd fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y){
        Eigen::MatrixXd XtX = X_train.transpose() * X_train;

    if (XtX.determinant() == 0) {
        std::cerr << "Matrix is singular and cannot be inverted, Consider using a PseudoInverse matrix" << std::endl;
        return Eigen::VectorXd();  // Return an empty matrix or handle differently
    }


        Eigen::MatrixXd transpose_X = X_train.transpose();

        Eigen::VectorXd beta_hat = (transpose_X * X_train).inverse() * transpose_X * y_train;  

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

};

class Linear_Regression_Multivariate{
    Eigen::MatrixXd X_train;
    Eigen::VectorXd y_train;

    public:
    Linear_Regression_Multivariate(const Eigen::MatrixXd& X_train, const Eigen::VectorXd& y_train) {
        this->X_train = X_train;
        this->y_train = y_train;
        


            }



    Eigen::MatrixXd fit(const Eigen::MatrixXd& X, const Eigen::MatrixXd& y){
    Eigen::MatrixXd XtX = X_train.transpose() * X_train;

    if (XtX.determinant() == 0) {
        std::cerr << "Matrix is singular and cannot be inverted, Consider using a PsuedoInverse matrix" << std::endl;
        return Eigen::MatrixXd();  // Return an empty matrix or handle differently
    }
        Eigen::MatrixXd transpose_X = X_train.transpose();
        Eigen::MatrixXd beta_hat = (transpose_X * X_train).inverse() * transpose_X * y_train;  

        return beta_hat;  
    }


    double evaluate(const Eigen::MatrixXd& X, const Eigen::MatrixXd& y, const Eigen::MatrixXd& beta_hat){
        //Calcuate y_hat
        Eigen::MatrixXd y_hat = X_train * beta_hat;

        //Calculate residuals
        Eigen::MatrixXd residuals = (y_train-y_hat)  ;


        //Calcuate the mean squared error
        double mean_sqaured_error = residuals.squaredNorm() / residuals.size();


        return mean_sqaured_error;
    }

};
