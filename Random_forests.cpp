#include <iostream>







/*
pseudocode:
Class RandomForest:
     X_train;
     y_train;
     num_trees;
     bootstrapped_samples;

    constructor(X_train, Y_train, num_trees = 1000){
        this.X_train = X_train;
        this.y_train = y_train;
        this->num_trees = num_trees;

        bootstap();
    
    }

    Function bootstrap():
        All_samples = []
        current_sample = []

        for i from 0 to num_trees:
            for j from 0 to X_train.size():
                current_sample.append(random.choice(X_train[j]))


            All_samples.append(current_sample)


        bootstrapped_samples = All_samples'

    Function train():
        #Random Feature Selection
        num_features = X_train[0].size()
        if(type = "Classification"):
            for i from 0 to num_trees:
                for j from 0 to X_train.size():
                    random_features = random.sample(range(0, num_features), int(sqrt(num_features)))
                    tree = DecisionTreeClassifier()
                    bootstrapped_samples[i] = np.array(bootstrapped_samples[i])[random_features]

                    trees_random_features.append(random_features)
                    tree.fit(bootstrapped_samples[i], y_train)
                    trees.append(tree)
                    
        else:
            for i from 0 to num_trees:
                     for j from 0 to X_train.size():
                    random_features = random.sample(range(0, num_features), int((num_features/3)))
                    tree = DecisionTreeClassifier()
                    bootstrapped_samples[i] = np.array(bootstrapped_samples[i])[random_features]


                    tree.fit(bootstrapped_samples[i], y_train)
                    trees.append(tree)



    Predict_Classification(X_test):
        predictions = []
        predictions_by_tree = []
        for i from 0 to num_trees:
            for j from 0 to X_test.size():
                predictions_by_tree.append(trees[i].predict(X_test[i][tree_random_features[i]]))

            predictions.append(predictions_by_tree.mode()




        return (predictions)


    Predict_regression(X_test):
        predictions = []
        predictions_by_tree = []
        for i from 0 to num_trees:
            for j from 0 to X_test.size():
                predictions_by_tree.append(trees[i].predict(X_test[i][tree_random_features[i]]))

            predictions.append(predictions_by_tree.mean())




        return (predictions)









*/