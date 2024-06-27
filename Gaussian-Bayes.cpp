#include <iostream>
#include <Eigen/Dense>
#include <boost/algorithm/string.hpp>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <cmath>
struct VectorHash {
    size_t operator()(const vector<string>& vec) const {
        size_t hash = 0;
        for (const auto& str : vec) {
            hash ^= std::hash<string>()(str) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        return hash;
    }
};
using namespace std;
using Document = vector<string>; // Each document is a vector of words
class GuassianBayes{

    Eigen :: MatrixXd X_train; 
    Eigen::VectorXd y_train;

    // this is a hashtable with all the data combined by class, the first key is the class and the second key is a vector of documents and the third key is the document
    unordered_map<int, Eigen:VectorXd > combined_data;
    /*
    Structure:
    {
        class1: [[Feature1_value, Feature2_value, Feature3_value],[Feature1_value, Feature2_value, Feature3_value] ],
        class2: [[Feature1_value, Feature2_value, Feature3_value],[Feature1_value, Feature2_value, Feature3_value] ],
        class3: [[Feature1_value, Feature2_value, Feature3_value], [Feature1_value, Feature2_value, Feature3_value]],

    }
    */

    unordered_map<string, unordered_map<int, double>> P_word_class; // Correct type

    // The first one is the class, the second one is word and the third one is the count by doc and class 
    unordered_map<int, unordered_map<string, int>> Words_by_doc_by_class;

    unordered_map<int, unordered_map<string, int>> feature_counts_by_class;
    unordered_set<string> vocabulary;
    unordered_map<int, int> all_feature_counts_by_class;

public:
    GuassianBayes(Eigen::VectorXd y_train, vector<string> Raw_data) {
        X_train = preprocess_text(Raw_data);
        this->y_train = y_train;
        combine_data();
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
                combined_data[class_] = Eigen:VectorXd; // Initialize vector if not present
            }
            combined_data[class_].push_back(X_train[i]);
        }
    }

public:
    void fit() {
        compute_prior_probabilities();
        compute_conditional_probability();
    }

private:
    void compute_prior_probabilities() {
        unordered_map<int, int> P_class;
        unordered_map<string, int> P_feature;

        for (int i = 0; i < y_train.size(); ++i) {
            int class_ = y_train[i];
            if (P_class.find(class_) == P_class.end()) {
                P_class[class_] = 0;
            }
            P_class[class_] += 1;
        }

        for (int i = 0; i < X_train.size(); i++) {
            for (int j = 0; j < X_train[i].size(); j++) {
                string feature = X_train[i][j];
                if (P_feature.find(feature) == P_feature.end()) {
                    P_feature[feature] = 0;
                }
                P_feature[feature] += 1;
            }
        }
    }

private:
    void compute_conditional_probability() {
        // Getting the vocab
        for (const auto& doc : X_train) {
            for (const auto& word : doc) {
                vocabulary.insert(word);
            }
        }
        
        for (auto it = combined_data.begin(); it != combined_data.end(); ++it) {
            for (int i = 0; i < it->second.size(); i++) {
                all_feature_counts_by_class[it->first] += it->second[i].size();
                for (int j = 0; j < it->second[i].size(); ++j) {
                    string word = it->second[i][j];
                    int class_ = it->first;
                    if (feature_counts_by_class[class_].find(word) == feature_counts_by_class[class_].end()) {
                        feature_counts_by_class[class_][word] = 0;
                    }
                    feature_counts_by_class[class_][word] += 1;
                }
            }
        }


        //? Structure: mean[Class][feature]
        unordered_map<int, unordered_map<int, double>> mean_y;
        unordered_map<int, double> overall_mean_y; // Overall mean for each class    
        for (auto it = combined_data.begin(); it != combined_data.end(); ++it){
            int class_ = it->first;
            int count =it->second.size();
            if(mean_y.find(class_)==mean_y.end()){
                mean_y[class_] = unordered_map<int,double>;
            }
            //?Over here we need to go column by column so we are going to switch i and j
                for (int i = 0; i < it->second[0].size(); i++) {
                        for (int j = 0; j < it->second.size(); ++j) {
                        int feature_ = i;
                        double sum = 0.0;

                        if (mean_y[class_].find(feature_) == mean_y[class_].end()) {
                                  mean_y[class_][feature_] = 0;
                    }                                
                            sum+= it->second[j][i];       

        }
         mean_y[class_][feature_] = (1.0/count) * sum;
                }}


        // Calculate the overall mean for each class
        for (auto it = mean_y.begin(); it != mean_y.end(); ++it) {
            int class_ = it->first;
            double sum_means = 0.0;
            int feature_count = it->second.size();
            for (auto feature_it = it->second.begin(); feature_it != it->second.end(); ++feature_it) {
                sum_means += feature_it->second;
            }
            overall_mean_y[class_] = sum_means / feature_count;
        }


        //?Calculating conditional probability
        


            for (auto it = mean_y.begin(); it != mean_y.end(); ++it) {
            int class_ = it->first;
            double sum_means = 0.0;
            int feature_count = it->second.size();
            for (auto feature_it = it->second.begin(); feature_it != it->second.end(); ++feature_it) {
                sum_means += feature_it->second;
            }
            overall_mean_y[class_] = sum_means / feature_count;
        }

        





        unordered_map<int, unordered_map<int, double>> variance_y; // Variance of each feature for each class

        
        for (auto it = combined_data.begin(); it != combined_data.end(); ++it){
            int class_ = it->first;
            int count =it->second.size();
            if(variance_y.find(class_)==variance_y.end()){
                variance_y[class_] = unordered_map<int,double>;
            }
            //?Over here we need to go column by column so we are going to switch i and j
                for (int i = 0; i < it->second[0].size(); i++) {
                        double sum_of_squares = 0.0;
                        for (int j = 0; j < it->second.size(); ++j) {    
                        double x_i= it-> second[j][i]

            sum_of_squares += pow(x_i - mean_y[class_][i], 2); // Fix the variance calculation
        }
                }
                    variance_y[class][i]= sum_of_squares/count;
                
                }



        //?Computing the Conditional Probability

        unordered_map<int, unordered_map<int, double> P_feature_Class;

         for (auto it = combined_data.begin(); it != combined_data.end(); ++it) {
            int class_ = it->first;

            for (int i = 0; i < it->second[0].size(); i++) {
                int feature_ = i;

                for (int j = 0; j < it->second.size(); ++j) {
                    double x_i = stod(it->second[j][i]); // Convert string to double
                    double mean = mean_y[class_][feature_];
                    double variance = variance_y[class_][feature_];
                                    if (variance == 0) {

                        probability = (x_i == mean) ? 1.0 : 0.0;

                    } else {

                        probability = (1 / sqrt(2 * M_PI * variance)) * exp(-pow(x_i - mean, 2) / (2 * variance));

                    }


                    if (P_feature_class.find(class_) == P_feature_class.end()) {
                        P_feature_class[class_] = unordered_map<int, double>();
                    }
                }



                        P_feature_class[class_][feature_] = (1 / sqrt(2 * M_PI * variance)) * exp(-pow(x_i - mean, 2) / (2 * variance));

            }



        }



    }

public:
    Eigen::VectorXd Predict(vector<string> Raw_data) {
        vector<vector<string>> X_test = preprocess_text(Raw_data);
        unordered_map<int, unordered_map<vector<string>, double, VectorHash>> Predictions_by_class;
        double sum_of_probabilities = 0.0; // FOR Normalization

        for (auto it = feature_counts_by_class.begin(); it != feature_counts_by_class.end(); ++it) {
            int class_ = it->first;
            for (int i = 0; i < X_test.size(); i++) {
                Predictions_by_class[class_][X_test[i]] = 1.0;
                for (int j = 0; j < X_test[i].size(); j++) {
                    string feature = X_test[i][j];
                    //? Since the test set might contain the feature with a class that wasn't in the training set, so we might need to modify and add some stuff to this dictionary
                 if (P_feature_class.find(feature) == P_feature_class.end() || P_feature_class[feature].find(class_) == P_feature_class[feature].end()) {
                    P_feature_class[feature][class_] = 1.0  //? for the condition we did not see this feature in the training set
    }
                    Predictions_by_class[class_][X_test[i]] *= P_feature_class[feature][class_];
                }
                sum_of_probabilities += Predictions_by_class[class_][X_test[i]];
            }
        }

        // Normalize probabilities so they sum up to one
        for (auto& class_predictions : Predictions_by_class) {
            for (auto& doc_prediction : class_predictions.second) {
            doc_prediction.second /= sum_of_probabilities; // Normalize each probability
            }
        }

        Eigen::VectorXd result(X_test.size());

        for (int i = 0; i < X_test.size(); i++) {
            double max_prob = 0.0;
            int predicted_class = -1;
            for (auto& class_predictions : Predictions_by_class) {
                if (class_predictions.second.find(X_test[i]) != class_predictions.second.end() && class_predictions.second[X_test[i]] > max_prob) {
                    predicted_class = class_predictions.first;
                    max_prob = class_predictions.second[X_test[i]];
                }
            }
            result[i] = predicted_class;
        }
        return result;
    }
};
