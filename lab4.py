import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def sim(df, img, depth):
    df = df.fillna(0)
    X_train = df.sample(frac=0.8, random_state=25)
    Y_train = X_train["Class"]
    X_train = X_train.drop(["playerID"], axis=1)
    X_train = X_train.drop(["Class"], axis=1)
    X_test = df.drop(X_train.index)
    X_test = X_test.drop(["playerID"], axis=1)
    Y_true = X_test["Class"]
    X_test = X_test.drop(["Class"], axis=1)

    clf = DecisionTreeClassifier(max_depth=depth, random_state = 13, min_samples_split=10, min_samples_leaf=7)
    clf.fit(X_train, Y_train)
    y_pred = clf.predict(X_test)
    print(confusion_matrix(Y_true, y_pred))
    print(classification_report(Y_true, y_pred))
    fig = plt.figure(figsize=(25,20))
    _ = tree.plot_tree(clf, feature_names=X_test.columns, filled=True)
    fig.savefig(img)

def main():
    df = pd.read_csv('./TaskAbaseball.csv')
    df2 = pd.read_csv('./TaskBbaseball.csv')
    sim(df,"TaskA.png",4)
    sim(df2, "TaskB.png",3)

if __name__ == "__main__": 
    main()

