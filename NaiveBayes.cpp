#include <iostream>
#include <Eigen/Dense>
#include <boost/algorithm/string.hpp>


using namespace std;
using Document = vector<string>; // Each document is a vector of words
class MultinomialBayes{

    vector<string> X_train;
    Eigen:: VectorXd y_train;
    unordered_map<string, vector<vector<string>>> combined_data;

    public:
        MultinomialBayes(Eigen::VectorXd y_train, vector<string> Raw_data){
            X_train= preprocess_text(Raw_data);
            this->y_train = y_train

        }

    private:
        vector<string> preprocess_text(vector<string> Raw_data){
            vector<vector<string>> result;


            for (int i =0; i< Raw_data.size();i++){
                vector<string> splitstring;
                boost::algorithm::split(splitstring ,Raw_data[i], boost::is_any_of(" "));

                result.pushback(splitstring);

        }
                return result;
        }


    private:
        void combine_data(){

            for (int i =0; i<y_train.size(); i++){
                string class_ = y_train.rows(i);

                if(combined_data.find(class_) == combined_data.end()){
                    combined_data[class_]= [];

                }

                combined_data[class_].pushback(X_train[i]);


            }





        }


    public:
        void fit(){
            //call Compute Individual Probability
            //Call Compute Conditional Probability
            int pass;
        }
    

    private:
        void compute_individual_probability(){
            unordered_map<string,int> P_class;
            unordered_map<string,int> P_feature;

            for (string &class: y_train){
                if (P_class.find(class)==P_class.end()){
                    P_class[class]= 0;
                }
                P_class[class]+=1;

            }


            for (int i =0; i<X_train.size();i++){
                for (int j= 0; X_train(j).size();j++)
                    feature= X_train(i)(j);
                    if (P_feature.find(feature)==P_class.end()){
                                P_feature[feature]= 0;
                }

                        P_feature[class]+=1

            }


        }


    private:
        void compute_conditional_probability(){
            set<string> vocabulary;

            //Getting the vocab
            for (int i =0; i<X_train.size();i++){
                for (int j= 0; X_train(j).size();j++)
                    vocabulary.insert(X_train(i)(j));
                  
                }
            }

            unordered_map<string,unordered_map<string,int>> feature_counts_by_class;
            unordered_map<string,int> all_feature_counts_by_class;


            //? Setting up the dictionary for feature counts by class
            for (string &class: y_train){
                if (feature_counts_by_class.find(class)==feature_counts_by_class.end()){
                    feature_counts_by_class[class]= {};
                }

                if (all_feature_counts_by_class.find(class)==all_feature_counts_by_class.end()){
                    all_feature_counts_by_class[class]= {};
                }

            }















        }









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


}};