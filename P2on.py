import numpy as np
from matplotlib import pyplot as plt, patches
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import sklearn.metrics as skm
import warnings 

def LogReg(X, y, eta, maxiter=1000):
    warnings.filterwarnings('ignore')
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    theta = np.random.rand(X.shape[1])
    for i in range(maxiter):
        updateby = np.zeros(theta.shape)
        for j in range(X.shape[0]):
            updateby = updateby + X[j, :] * (sigmoid(np.dot(theta, X[j, :])) - y[j])
        updateby = updateby / X.shape[0]
        theta = theta - eta * updateby
    return theta
def LogRegPredict(X, theta):
    return [0 if np.dot(x, theta) < 0.5 else 1 for x in X]


def P2():
    indata = pd.read_csv("emails.csv")
    datanp = indata.to_numpy()
    data   = datanp[:, 1:indata.shape[1] - 1]
    labels = datanp[:, indata.shape[1] - 1]

    for i in [0, 1000, 2000, 3000, 4000]:
        testidxs  = list(range(i, i + 1000))
        trainidxs = list(range(i)) + list(range(i + 1000, 5000))
        Xtest = data[testidxs, :].astype(float);     Ytest = labels[testidxs].astype(int)
        Xtrain = data[trainidxs, :].astype(float);   Ytrain = labels[trainidxs].astype(int)

        nbrs = KNeighborsClassifier(n_neighbors=1).fit(Xtrain, Ytrain)
        Yhat = nbrs.predict(Xtest)

        accuracy  = nbrs.score(Xtest, Ytest)
        precision = skm.precision_score(Ytest, Yhat)
        recall    = skm.recall_score(Ytest, Yhat)

        print("Test set", i, "to", i + 999)
        print("\tAccuracy:", accuracy, "\tPrecision:", precision, "\tRecall:", recall)
def P3():
    indata = pd.read_csv("emails.csv")
    datanp = indata.to_numpy()
    data   = datanp[:, 1:indata.shape[1] - 1]
    labels = datanp[:, indata.shape[1] - 1]

    for i in [0, 1000, 2000, 3000, 4000]:
        testidxs  = list(range(i, i + 1000))
        trainidxs = list(range(i)) + list(range(i + 1000, 5000))
        Xtest = data[testidxs, :].astype(float);     Ytest = labels[testidxs].astype(int)
        Xtrain = data[trainidxs, :].astype(float);   Ytrain = labels[trainidxs].astype(int)

        theta = LogReg(Xtrain, Ytrain, 1, 1000)
        Ypred = LogRegPredict(Xtest, theta)

        accuracy  = skm.accuracy_score(Ypred, Ytest)
        precision = skm.precision_score(Ytest, Ypred)
        recall    = skm.recall_score(Ytest, Ypred)

        print("Test set", i, "to", i + 999)
        print("\tAccuracy:", accuracy, "\tPrecision:", precision, "\tRecall:", recall)
def P4():
    # Run 5-fold cross validation with kNN varying k (k=1, 3, 5, 7, 10). 
    # Plot the average accuracy versus k, and list the average accuracy of each case. 
    indata = pd.read_csv("emails.csv")
    datanp = indata.to_numpy()
    data   = datanp[:, 1:indata.shape[1] - 1]
    labels = datanp[:, indata.shape[1] - 1]

    avgaccuracies = []
    numneighbors = [1, 3, 5, 7, 10]
    for k in numneighbors:
        accuracies = []
        for i in [0, 1000, 2000, 3000, 4000]:
            testidxs  = list(range(i, i + 1000))
            trainidxs = list(range(i)) + list(range(i + 1000, 5000))
            #print("lengths match:", len(trainidxs) + len(testidxs), len(labels))
            #print("\t\ttest shape:", len(testidxs), "\tsamples:", np.array(testidxs)[[0, 999]])
            #print("\t\ttrain shape:", len(trainidxs), "\tsamples:", np.array(trainidxs)[[0, 999, 1000, 1999, 2000, 2999, 3000, 3999]])
            Xtest = data[testidxs, :].astype(float);     Ytest = labels[testidxs].astype(int)
            Xtrain = data[trainidxs, :].astype(float);   Ytrain = labels[trainidxs].astype(int)
            #print("Xtrain shape:", Xtrain.shape, "Xtest shape:", Xtest.shape)

            #print("Training data length:", len(Ytrain), "\tTesting data length:", len(Ytest))
            nbrs = KNeighborsClassifier(n_neighbors=k).fit(Xtrain, Ytrain)
            #YPred = nbrs.predict(Xtest)

            accuracies.append(nbrs.score(Xtest, Ytest))
        print("Accuracies for k =", k, "is:", accuracies, "\n\tAvg:", sum(accuracies)/5)
        avgaccuracies.append(sum(accuracies)/5)

    print("\nAverage accuracies for \tk = ", numneighbors, "\n\t", avgaccuracies)
    plt.plot(numneighbors, avgaccuracies)
    plt.title("Average Accuracy for 5-Fold Cross Validation for k Nearest Neighbors")
    plt.xlabel("Number of NeighborsUsed in kNN")
    plt.ylabel("Accuracy")
    plt.savefig("P2.4.png")
def P5():
    indata = pd.read_csv("emails.csv")
    datanp = indata.to_numpy()
    data   = datanp[:, 1:indata.shape[1] - 1]
    labels = datanp[:, indata.shape[1] - 1]
    Xtest = data[4000:, :].astype(float);    Ytest = labels[4000:].astype(int)
    Xtrain = data[:4000, :].astype(float);   Ytrain = labels[:4000].astype(int)

    nbrs     = KNeighborsClassifier(n_neighbors=5).fit(Xtrain, Ytrain)
    YpredKnn = nbrs.predict_proba(Xtest)
    theta    = LogReg(Xtrain, Ytrain, 1, 3000)
    YpredLR  = LogRegPredict(Xtest, theta)
    print(theta.shape, Xtest.shape)
    YpredLR2 = Xtest @ np.reshape(theta, (theta.shape[0], 1)) 
    print("Y LR:", YpredLR2.shape)
    print("Y KNN:", YpredKnn.shape)
    fprKnn, tprKnn, t = skm.roc_curve(Ytest, YpredKnn[:, 1])
    fprLR,  tprLR,  t = skm.roc_curve(Ytest, YpredLR2)

    plt.plot(fprKnn, tprKnn, label="KNN with k=5 ROC Curve")
    plt.plot(fprLR, tprLR, label="Logistic Regression ROC Curve")
    plt.title("ROC Curves for KNN and Logistic Regression")
    plt.savefig("P2.5.png")

#P2()
#P3()
#P4()
P5()
