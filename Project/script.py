from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from random import randint

def create_dataframe(file_name):
    dataset = pd.read_csv('onlyddos.csv')
    #print(dataset.head)
    return dataset

def preprocess_data(dataset):
    dataset['normal.'] = dataset['normal.'].replace(['back.', 'land.', 'neptune.', 'pod.', 'smurf.', 'teardrop.'],
                                                    'attack')

    x, y = dataset.iloc[:, :-1].values, dataset.iloc[:, 41].values

    le = LabelEncoder()
    x[:, 1], x[:, 2], x[:, 3] = le.fit_transform(x[:, 1]), le.fit_transform(x[:, 2]), le.fit_transform(x[:, 3])

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, test_size=0.3, random_state=50)

    sc_x = StandardScaler()
    x_train = sc_x.fit_transform(x_train)
    x_test = sc_x.transform(x_test)
    return x_train, x_test, y_train, y_test

def prepare_classifier(x_train,y_train,classifier_type):
    if(classifier_type == 1):
        from sklearn.naive_bayes import GaussianNB
        classifier = GaussianNB()
        print("Naive Bayes Classifier created.")
    elif(classifier_type == 2):
        from sklearn.tree import DecisionTreeClassifier
        classifier = DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth=3, min_samples_leaf=5)
        print("Decision Tree Classifier created.")
    elif (classifier_type == 3):
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
        print("Random Forest Classifier created.")
    elif (classifier_type == 4):
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier(n_neighbors=3)
        print("KNN Classifier created.")
    else:
        raise ValueError

    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    return y_pred


def show_results(y_test,y_pred):
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix: ")
    print(cm)
    TP, TN, FP, FN = cm[0, 0], cm[1, 1], cm[1, 0], cm[0, 1]
    print("True Positives = " + str(TP))
    print("True Negatives = " + str(TN))
    print("False Positives = " + str(FP))
    print("False Negatives = " + str(FN))

    recall = TP / (TP + FN)
    specifity = TN / (TN + FP)
    precision = TP / (TP + FP)
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    print("Recall = " + str(recall))
    print("Specifity = " + str(specifity))
    print("Precision = " + str(precision))
    print("Accuracy = " + str(accuracy))


dataset = create_dataframe("onlyddos.csv")
x_train, x_test, y_train, y_test = preprocess_data(dataset)
y_pred = prepare_classifier(x_train,y_train, 1)
#y_pred = prepare_classifier(x_train,y_train, randint(1,4))
show_results(y_test, y_pred)
