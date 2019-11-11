import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from art import *

def split_csv(filename):

    df = pd.read_csv(filename, dtype={'user_id': int},  encoding='utf-8', na_filter=False)

    df_row_size = df.shape[0]
    df = df.astype(str)
    print(df.shape)
    df = df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
    options = ['canceled', 'failed', 'live', 'successful', 'suspended', 'undefined'] 
    df = df[df['state '].isin(options)] 
    df['state '][df['state '] == 'canceled'] = 0
    df['state '][df['state '] == 'failed'] = 1
    df['state '][df['state '] == 'live'] = 2
    df['state '][df['state '] == 'successful'] = 3
    df['state '][df['state '] == 'suspended'] = 4
    df['state '][df['state '] == 'undefined'] = 5
    print(df['state '])
    print("dropping columns... "),
    df = df.drop(['launched ', 'deadline '],axis=1)
    print(df.shape)


    test_percentage        = .1
    validate_percentage    = .2
    train_percentage       = .7

    test_rows              = int(test_percentage*df_row_size)
    validate_rows          = int(validate_percentage*df_row_size)
    train_rows             = int(train_percentage*df_row_size)

    df_test                = df.iloc[:test_rows]
    df_validate            = df.iloc[test_rows:test_rows+validate_rows]
    df_train               = df.iloc[test_rows+validate_rows:]

    if(not os.path.isfile('ks_test.csv')):
        print('saving dataframes')
        df_test.to_csv('ks_test.csv')
        df_validate.to_csv('ks_validate.csv')
        df_train.to_csv('ks_train.csv')

        print("\ntest " + str(df_test.shape) + "\n")
        print(df_test.head())
        print("\nvalidate " + str(df_validate.shape) + "\n")
        print(df_validate.head())
        print("\ntrain " + str(df_train.shape) + "\n")
        print(df_train.head())

def make_predictions(lr, x_train, y_train, x_test):
    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_test)
    return y_pred

def generate_csv(ids, predictions):
    with open('kickstarters5_results.csv', 'w') as f:
        print('ID ,Status', file=f)
        for id, pred in zip(ids, predictions):
            print('{},{}'.format(id, pred), file=f)

def prepare_dataset(filename, validate=True, train_filename='ks_test.csv'):
    df = pd.read_csv(filename, na_filter=False)
    if validate:                                                                                                #Entrenar modelo (Training set)
        df.drop(['ID ','name ','usd_pledged '], axis=1, inplace=True)                                                                    
    else:
        df.drop(['ID ', 'name ','usd_pledged '], axis=1, inplace=True)               #Drop columnas para comparar (Validation set)
        df_train = pd.read_csv(train_filename, na_filter=False)
    df = pd.get_dummies(df, columns=['category ','main_category ','currency ','country '])
    df.to_csv(r'hola.csv')
    return df        

def cross_validation(lr, x, y, folds=5, plot_results=True):
    train_score = []
    test_score = []
    for i in range(folds):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        lr.fit(x_train, y_train)
        train_score.append(lr.score(x_train, y_train))
        test_score.append(lr.score(x_test, y_test))
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
        return cv_scores

def make_predictions(lr, x_train, y_train, x_test):
    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_test)
    return y_pred

def generate_csv(ids, predictions):
    with open('kickstarter_results.csv', 'w') as f:
        print('ID ,state ', file=f)
        for id, pred in zip(ids, predictions):
            print('{},{}'.format(id, pred), file=f)


print("running...")
split_csv('kickstarters5.csv')

dataset = prepare_dataset('ks_train.csv')
x = dataset.iloc[:, 2:]
y = dataset.iloc[:, 1]
print("printing y")
print(y)
print("printing x")
print(x)

clf = tree.DecisionTreeClassifier(max_depth=4)

validate = False
if validate:
    cross_validation(clf, x, y, folds=8)
else:
    test_dataset = prepare_dataset('ks_test.csv', validate)
    print(test_dataset.head())
    project_id = test_dataset['Unnamed: 0']
    x_test = test_dataset.iloc[:,2:]

    predictions = make_predictions(clf, x, y, x_test)

generate_csv(project_id, predictions)
print("done")