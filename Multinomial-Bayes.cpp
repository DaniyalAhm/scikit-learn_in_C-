#include <iostream>
#include <Eigen/Dense>
#include <boost/algorithm/string.hpp>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>


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
class MultinomialBayes{

    vector<vector<string>> X_train; // Correct type
    Eigen:: VectorXd y_train;
    unordered_map<int, vector<vector<string>>> combined_data;
    unordered_map<string, unordered_map<int, double>> P_word_class; // Correct type
    unordered_map<int, unordered_map<string, int>> feature_counts_by_class;
    unordered_set<string> vocabulary;
    unordered_map<int,int> all_feature_counts_by_class;



    public:
        MultinomialBayes(Eigen::VectorXd y_train, vector<string> Raw_data){
            X_train = preprocess_text(Raw_data);
            this->y_train = y_train;
            combine_data();

        }

    private:
        vector<vector<string>> preprocess_text(vector<string> Raw_data){
            vector<vector<string>> result;


            for (const auto& doc: Raw_data){
                vector<string> splitstring;
                boost::algorithm::split(splitstring ,doc, boost::is_any_of(" "));

                result.push_back(splitstring);

        
        }
            return result;

        }
        void combine_data(){

            for (int i =0; i<y_train.size(); ++i){
                int class_ = y_train(i);

                if(combined_data.find(class_) == combined_data.end()){
                    combined_data[class_] = vector<vector<string>>(); // Initialize vector if not present


                }
                combined_data[class_].push_back(X_train[i]);

            }

        };


    public:
        void fit(){
            compute_individual_probability();
            compute_conditional_probability();
            
        }
    

    private:
        void compute_individual_probability(){
            unordered_map<int,int> P_class;
            unordered_map<string,int> P_feature;

            for (int i= 0; i<y_train.size();++i){
                int class_= y_train[i];
                if (P_class.find(class_)==P_class.end()){
                    P_class[class_]= 0;
                }
                P_class[class_]+=1;

            }

            for (int i =0; i<X_train.size();i++){
                for (int j= 0; j< X_train[i].size();j++){
                    string feature= X_train[i][j];
                    if (P_feature.find(feature)==P_feature.end()){
                                P_feature[feature]= 0;}

                        P_feature[feature]+=1;
                }

            }

        };


    private:
        void compute_conditional_probability(){

            //Getting the vocab
            // Getting the vocab
            for (const auto& doc : X_train) {
                for (const auto& word : doc) {
                    vocabulary.insert(word);
                }
            }




            //? Setting up the dictionary for feature counts by class
            for (int class_: y_train){
                if (feature_counts_by_class.find(class_)==feature_counts_by_class.end()){
                    feature_counts_by_class[class_]= unordered_map<string,int>();
                }

                if (all_feature_counts_by_class.find(class_)==all_feature_counts_by_class.end()){
                    all_feature_counts_by_class[class_]= 0;
                }

            }


            for (auto it = combined_data.begin(); it != combined_data.end(); ++it) {
                    for (int i =0; i< it ->second.size();i++){
                        all_feature_counts_by_class[it->first] += it->second[i].size();
                    

                    for ( int j = 0; j< it ->second[i].size(); ++j){
                        string word = it ->second[i][j];
                        int class_ = it -> first;
                    if (feature_counts_by_class[class_].find(word)==feature_counts_by_class[class_].end()){
                            feature_counts_by_class[class_][word]= 0;
                    }





                        feature_counts_by_class[class_][word]+= 1;
                    }}

            }


            //? Now We calculate the conditional probabilities of each feature


            for (auto word = vocabulary.begin(); word != vocabulary.end(); ++word) {
                for (auto it = feature_counts_by_class.begin(); it != feature_counts_by_class.end(); ++it) {
                    int class_ = it -> first;
                    if (P_word_class.find(*word)==P_word_class.end()){
                                        P_word_class[*word]= unordered_map<int,double>();

                }

                       if (P_word_class[*word].find(class_)==P_word_class[*word].end()){
                                        P_word_class[*word][class_]= 0.0;
                           
                    }


                P_word_class[*word][class_]= (feature_counts_by_class[class_][*word]+1.0)/(all_feature_counts_by_class[class_]+vocabulary.size());

        }}};



    public:
        Eigen::VectorXd Predict (vector<string> Raw_data){

           
           vector<vector<string>>  X_test = preprocess_text(Raw_data);
            unordered_map<int, unordered_map<vector<string>, double, VectorHash>> Predictions_by_class;
            double sum_of_probabilities =0.0; //FOR Normalization
            for (auto it = feature_counts_by_class.begin(); it != feature_counts_by_class.end(); ++it) {
                int class_ = it->first;

                for(int i =0; i < X_test.size();i++){                    
                     Predictions_by_class[class_][X_test[i]]=1.0;
                    for(int j =0; j<X_test[i].size();j++){
                        //?If that word is not already associated with that feature
                        string feature= X_test[i][j];
                        if(P_word_class[feature].find(class_)==P_word_class[feature].end()){
                            P_word_class[feature][class_]= 1.0/(all_feature_counts_by_class[class_]+vocabulary.size());

                        }
                        Predictions_by_class[class_][X_test[i]]*= P_word_class[feature][class_];

                    }


                    sum_of_probabilities+=Predictions_by_class[class_][X_test[i]];

                }
                
            }

            //?NORMALIZATION SO ALL PROBABILITIES ADD UP TO ONE
                    // Normalize probabilities so they sum up to one
            for (auto& class_predictions : Predictions_by_class) {
                for (auto& doc_prediction : class_predictions.second) {
                    doc_prediction.second /= sum_of_probabilities;
                }
            }
                
            Eigen::VectorXd result(X_test.size());


            for(int i =0; i < X_test.size();i++){
                double max_prob=0.0;
                int predicted_class =-1;


            for (auto& class_predictions : Predictions_by_class) {


                    if(class_predictions.second[X_test[i]]>max_prob){
                        predicted_class= class_predictions.first;
                        max_prob= class_predictions.second[X_test[i]];

                    }

            }

            result[i]= predicted_class;
            }




        return result;

        };

};





    /*
    Psuedo-Code:
        Inputs: Y_train, X_train
        Function_PreProcess X_train(X_train):
            Split each word by " " and turn into an array
            X_train = now a list of words 


        Function_ Compute Individual Probablity
            Probablity_of_class = {}
            Probablity_of_each_feature = {}

            For every class in Y_train:
                Probablity_of_class[class]= total_count(class)/len(Y_train)


            For each document in X_train:
                For each feature in document:
                    Probablity_of_each_feature[word] = total_count[word]/all_words



        Function_Calculating_Conditional Probability:
     
            
            #Calculating P(feautre1 |spam)
            Vocab size=(al_words_in_all_docs)

            Probabailitys_Feature_given_Class = {}


            !Overall Structure Class_word_counts = {word1:count1, word2:count2...}
            class_word_counts = dictionary of dictionaries to store count of each word given a class

            !Overall Structure Class_counts = {class:total_word Count1...}
            class_counts = dictionary to store count of all words for each class

            For each document, class in zip(X_train_processed, Y_train):


                For each document, class in zip(X_train_processed, Y_train):
                        If class not in class_word_counts:
                            class_word_counts[class] = {}
                            class_counts[class] = 0   

                        For word in document:
                            If word not in class_word_counts[class]:
                                    class_word_counts[class][word] = 0
                                class_word_counts[class][word] += 1 
                                class_counts[class] += 1


            For word in set_of_words:
                        Conditional_Probability[word] = {}
                        For class in class_word_counts:
                            !Word count by class SUPPORTS MY ORIGINAL LOGIC
                            word_count = class_word_counts[class][word]
                            !TOTAL COUNT OF WORDS BY CLASS FOR THE DENOMINATOR
                            total_count = class_counts[class] #Count of all the world


                            !THE MATHIMATICAL FORMULA
                            P(Feature| Class) = count(feature)+1/total counts of words in that class + overall vocab size




                            !THE BAUETIFUL FORMULA with laplace smoothing
                            Conditional_Probability[word][class] = (word_count + 1) / (total_count + vocabulary_size)
                    

        Function Predict
            #Calculating Proportional Probabilities
            Probabilties_class_given_feature
            
            For each class:
                 Probabilties_class_given_feature[class]= Probablity_of_class[class]
                 For each feature in class:
                        Probabilties_class_given_feature[class]=  Probabilties_class_given_feature* Probabailitys_Feature_given_Class[class][feature] 


                # Normalize the probabilities
                    sum_probabilities = sum(Probabilties_class_given_feature.values()) //!Sum of total probabilties
                    For class in Probabilties_class_given_feature:
                        Probabilties_class_given_feature[class] /= sum_probabilities
                    

                Return Probabilties_class_given_feature

    */


