import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import os
removeList = ["id","ID"]
def set_dataframe(nameofdf):
    global df
    prevdf = pd.read_csv(nameofdf)
    df = prevdf.dropna()
    for id in df.columns:
        for i in removeList:
            if i in id:
                remove_Col(id)
    remove_Col("Unnamed: 32")
    return df
#F ed up a whole night for this without sleeping
def remove_Col(*args):
    for arg in args:
        try:
            del df[arg]
        except KeyError:
            print("Handled")
# def plot_2Columns(col1,col2):
#     plt.scatter(x=df.loc[:,col1].values,y=df.loc[:,col2].values,c='DarkBlue')
#     plt.title(col1+" vs "+col2)
#     plt.xlabel(col1)
#     plt.ylabel(col2)
#     plt.savefig("static/images/"+col1+"_vs_"+col2+".jpg")

def classify_columns():
    headers = df.columns
    global NumColumns,CategoricalDataColumns
    NumColumns = set([])
    CategoricalDataColumns = set([])
    for i in range(0,len(headers)):
        if df.dtypes[i] == 'int64' or df.dtypes[i] == 'float64':
            NumColumns.add(df.columns[i])
        else:
            CategoricalDataColumns.add(df.columns[i])
    # print(NumColumns,CategoricalDataColumns)
    # print(len(NumColumns),len(CategoricalDataColumns))
def get_shape():
    col = df.shape[1]
    row = df.shape[0]
    return [row,col]
def handle_CatCol(sets):
    label_Encoder = LabelEncoder()
    X = df.loc[:,sets].values
    Z = label_Encoder.fit_transform(X)
    return Z

def set_get_X():
    X = df.loc[:,NumColumns].values
    return X

def set_get_Y():
    Y = handle_CatCol(CategoricalDataColumns)
    return Y
def split_train_test_scale(X,Y):
    sc = StandardScaler()
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=0)
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    list1 = [X_train,X_test,Y_train,Y_test]
    return list1


def model_LogisticRegression(X_train,Y_train,X_test):
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)
    return Y_pred

def model_KNeigh(X_train,Y_train,X_test):
    classifier = KNeighborsClassifier()
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)
    return Y_pred

def model_SVM(X_train,Y_train,X_test):
    classifier = SVC(kernel = 'linear', random_state = 0)
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)
    return Y_pred
def model_kernelSVM(X_train,Y_train,X_test):
    classifier = SVC(kernel = 'rbf', random_state = 0)
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)
    return Y_pred
def model_NaiveBayes(X_train,Y_train,X_test):
    classifier = GaussianNB()
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)
    return Y_pred
def model_DecisionTree(X_train,Y_train,X_test):
    classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)
    return Y_pred
def model_RandomForest(X_train,Y_train,X_test):
    classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)
    return Y_pred

def show_accuracy(Y_test,Y_pred):
    from sklearn.metrics import accuracy_score
    cm = accuracy_score(Y_test, Y_pred)
    return (str(round(cm*100,2))+"%")



# def save_histplots(loc):
#     for i in NumColumns:
#         plt.hist(df.loc[:,i])
#         plt.title(i)
#         if not os.path.exists(loc):
#             os.makedirs(loc)
#         plt.savefig(loc+"/"+i+".jpg")


# def model_KNeigh():
#     classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
#     classifier.fit(X_train, Y_train)
# def model_SVM():
#     classifier = SVC(kernel = 'linear', random_state = 0)
#     classifier.fit(X_train, Y_train)
# def model_kernelSVM():
#     classifier = SVC(kernel = 'rbf', random_state = 0)
#     classifier.fit(X_train, Y_train)
# def model_NaiveBayes():
#     classifier = GaussianNB()
#     classifier.fit(X_train, Y_train)
# def model_DecisionTree():
#     classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
#     classifier.fit(X_train, Y_train)
# def model_RandomForest():
#     classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
#     classifier.fit(X_train, Y_train)
