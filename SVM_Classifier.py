"""
Neural Networks - Deep Learning
Heart Disease Predictor ( Binary Classification )
Author: Dimitrios Spanos Email: dimitrioss@ece.auth.gr
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import time, os, math
import seaborn as sn # for heatmaps
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from M0 import my_SVM

labels = ['no disease', 'disease']
scaler = StandardScaler()

def main():
    # Creation of trn,tst datasets
    trn_x, trn_y, tst_x, tst_y = getData()

    print("Training:", len(trn_y), "instances.")
    print("Testing:", len(tst_y), "instances.")

    # This changes the problem from 0/1 to -1/1, to help the SVM
    trn_y = trn_y.to_numpy()
    tst_y = tst_y.to_numpy()
    for i in range(len(trn_y)):
        if trn_y[i] == 0:
            trn_y[i] = -1

    for i in range(len(tst_y)):
        if tst_y[i] == 0:
            tst_y[i] = -1

    svm = my_SVM(C=1000.0, kernel='rbf', sigma=0.01)
    start = time.time()
    svm.fit(trn_x, trn_y)
    end = time.time()
    print(f"Training took {(end - start)*1000:.2f}ms")
    train_y_pred = []
    for i in trn_x:
        pred = svm.predict(i)
        train_y_pred.append(pred)
    print(f"Train_acc: {accuracy_score(y_true=trn_y, y_pred=train_y_pred) * 100:.2f}%")

    test_y_pred = []
    one_correct = False
    one_false = False
    for instance, i in enumerate(tst_x):
        pred = svm.predict(i)
        if pred == tst_y[instance] and one_correct==False:
            attributes = scaler.inverse_transform(i)
            one_correct = True
            print("Correct prediction attributes:", attributes)
            print("True label:", tst_y[instance])
        if pred != tst_y[instance] and one_false==False:
            attributes = scaler.inverse_transform(i)
            one_false = True
            print("False prediction attributes:", attributes)
            print("True label:", tst_y[instance])
        test_y_pred.append(pred)


    print(f"Test_acc: {accuracy_score(y_true=tst_y,y_pred=test_y_pred)*100:.2f}%")
    drawConfusionMatrix(tst_y, test_y_pred, 'M0_Confusion_Matrix')


def drawConfusionMatrix(tst_y, y_pred, name):

    # Draw the Confusion Matrix
    cf_matrix = confusion_matrix(tst_y, y_pred)
    plt.title('Heart Disease Prediction Confusion Matrix')
    sn.set(font_scale=0.6)
    sn.heatmap(cf_matrix / np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.savefig(name, dpi=550, bbox_inches='tight')
    plt.close()


def getData():
    """
    :return: The datasets in numpy form
    """
    # 13 attributes + target
    df = pd.read_csv('./heart.csv')
    X = df.drop('target', axis=1)
    y = df["target"]

    # 60% Training - 40% Testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, shuffle=True, random_state=42)

    # normalize the data

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test

if __name__ == '__main__':
    main()