import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import idna
import sys
import pickle
import graphviz
from sklearn import tree
from sklearn.model_selection import train_test_split
import operator

max_depth  = "15"
run_model  = "decision_tree_" + max_depth

def split_csv(filename):

    df = pd.read_csv(filename, dtype={'user_id': int},  encoding='utf-8', na_filter=False)

    if(not os.path.isfile('ks_test.csv' or 'ks_train')):
        print("building dataframes...")
        df_row_size = df.shape[0]
        df = df.astype(str)
        print(df.shape)
        df = df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
        df = df.sample(frac=1).reset_index(drop=True)
        print("suffling dataset rows...")
        options = ['failed','successful','undefined'] 
        df = df[df['state '].isin(options)] 
        print("dropping columns... ")
        df = df.drop(['launched ', 'deadline ', 'pledged ', 'backers '],axis=1)

        print(df.shape)
        df['category '] = df['category '].replace({'&':''}, regex=True)
        df['main_category '] = df['main_category '].replace({'&':''}, regex=True)

        test_percentage        = .1
        train_percentage       = .9

        test_rows              = int(test_percentage*df_row_size)
        train_rows             = int(train_percentage*df_row_size)

        df_test                = df.iloc[:test_rows]
        df_train               = df.iloc[test_rows:]

        print('saving dataframes')
        df_test.to_csv('ks_test.csv')
        df_train.to_csv('ks_train.csv')

        print("\ntest " + str(df_test.shape) + "\n")
        print(df_test.head())
        print("\ntrain " + str(df_train.shape) + "\n")
        print(df_train.head())
    else:
        print("no need to split dataframes, test, validate and train sets already exist...")


def prepare_dataset(filename):
    df = pd.read_csv(filename, na_filter=False)
    df.drop(['ID ', 'name ','usd_pledged '], axis=1, inplace=True)               #Drop columnas para comparar (Validation set)
    df = pd.get_dummies(df, columns=['category ','main_category ','currency ','country '])
    return df        

def cross_validation(lr, x, y, folds=5, plot_results=True):
    print("running cross validation...")
    train_score = []
    test_score = []
    for i in range(folds):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
        lr.fit(x_train, y_train)
        train_score.append(lr.score(x_train, y_train))
        test_score.append(lr.score(x_test, y_test))
        f_2= open("ValidationResults.txt","a+")
    if plot_results:
        x_axis = np.array([i for i in range(1, folds+1)])
        plt.plot(x_axis, train_score, label='train score', linewidth=3, color='red')
        plt.plot(x_axis, test_score, label='validation score', linewidth=3, color='blue')
        plt.title('Cross validation results')
        plt.xlabel('Fold')
        plt.ylabel('Score')
        plt.legend()
        plt.show()
    else:
        pass    
    train_avg = str(sum(train_score)/len(train_score))
    test_avg  = str(sum(test_score)/len(test_score))
    print("train avg: " + train_avg)
    print("test avg: " + test_avg)
    log_string = "Model: " + run_model + ", Train Score: " + train_avg + ", Test Score: " + test_avg + ". \r \n"
    f_2.write(log_string)

def make_predictions(lr, x_train, y_train, x_test):
    y_pred = lr.predict(x_test)
    return y_pred

def generate_csv(ids, predictions):
    with open('kickstarter_results.csv', 'w') as f:
        print('ID ,state ', file=f)
        for id, pred in zip(ids, predictions):
            print('{},{}'.format(id, pred), file=f)


print("running...")

split_csv('kickstarters.csv')

dataset = prepare_dataset('ks_train.csv')
x = dataset.iloc[:, 2:]
y = dataset.iloc[:, 1]
print("printing y")
print(y)
print("printing x")
print(x)

print("loading model...")

clf = tree.DecisionTreeClassifier(criterion="gini",max_depth=int(max_depth), min_samples_split=2, min_samples_leaf=1,
                                   min_weight_fraction_leaf=0, max_features=None, random_state=None,
                                   max_leaf_nodes=None, min_impurity_decrease=0, min_impurity_split=None,
                                   class_weight=None, presort=False)


if(not os.path.isfile(run_model + '.pickle')):
    print("no model with name " + run_model + " found, training new model...")
    clf = clf.fit(x, y)
    with open(run_model + ".pickle", "wb") as decisiontreeclf:
        pickle.dump(clf, decisiontreeclf)
        print("model trained and saved...")
else:
    print("pickled model " + run_model + " found...")
    pickle_in = open(run_model +".pickle", "rb")
    clf = pickle.load(pickle_in)

train_score = []
test_score = []

validate = True

if validate:

    cross_validation(clf, x, y, folds=5)
    test_dataset = prepare_dataset('ks_test.csv')
    dataset_header = list(test_dataset)[1:]

else:

    test_dataset = prepare_dataset('ks_test.csv')
    dataset_header = list(test_dataset)[2:]
    project_id = test_dataset['Unnamed: 0']
    x_test = test_dataset.iloc[1,2:]
    test_dataset.to_csv()
    x_test.to_csv()
    print("test values:")
    print(x_test)
    input("Press Enter to continue...")
    x_test = pd.Series(x_test).values
    x_test = x_test.reshape(1,-1)
#    predictions = make_predictions(clf, x, y, x_test)
#    generate_csv(project_id, predictions)

dot_data = tree.export_graphviz(clf, out_file=None,
                          feature_names=dataset_header, 
                          class_names=['failed','successful','undefined'],  
                          filled=True, rounded=True,  
                          special_characters=True)   
graph = graphviz.Source(dot_data) 
graph.render(run_model) 


features = dict(zip(dataset_header, clf.feature_importances_))
features_sorted = sorted(features.items(), key=operator.itemgetter(1), reverse=True)

for feature in features_sorted:
    print(feature)

print("\n \n done")