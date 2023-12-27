
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


def main_function():
    dataset = pd.read_csv('G:/Users/tinys/PycharmProjects/Workshop2/Iris.csv')
    names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
    dataset.head()
    # divide the dataset into a feature set and corresponding labels
    X = dataset.drop('Species', axis=1)
    y = dataset['Species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    pca = PCA()

    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    explained_variance = pca.explained_variance_ratio_
    print(explained_variance)

    pca = PCA(n_components = 0.85)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    pca.n_components_
    #classifier = RandomForestClassifier(max_depth = 2, random_state = 0)
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print('Accuracy')
    print(accuracy_score(y_test, y_pred))

    show_graph(y_pred, y_test)


def show_graph(y_pred, y_test):
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    labels = [0, 1]
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    # create heatmap
    sns.heatmap(pd.DataFrame(cnf_matrix), annot = True, cmap = "YlGnBu", fmt = 'g')
    ax.xaxis.set_label_position("top")
    plt.title('Confusion matrix', y = 1.1)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.show()


if __name__ == '__main__':
    main_function()


