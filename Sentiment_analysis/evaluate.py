from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
from sklearn.neighbors import KNeighborsClassifier

def run_some_models(X_train, X_test, y_train, y_test):

    print("running Knn")
    neigh = KNeighborsClassifier(n_neighbors=4)
    neigh.fit(X_train,y_train)
    y_pred=neigh.predict(X_train)
    print('accuracy train : ', accuracy_score(y_train, y_pred))

    y_pred = neigh.predict(X_test)
    print('test accuracy :', accuracy_score(y_test, y_pred))
    print("classification report")
    print(classification_report(y_test, y_pred))


    # ___________________ using naive bayes__________________#
    print("Running Naive bayes...")
    NB_model = GaussianNB()
    NB_model = NB_model.fit(X=X_train, y=y_train)

    y_pred = NB_model.predict(X_train)
    print('accuracy train :', accuracy_score(y_train, y_pred))

    y_pred = NB_model.predict(X_test)
    print('test accuracy: ',  accuracy_score(y_test, y_pred))
    print("classification report")
    print(classification_report(y_test, y_pred))

    # _____________________Soppurt vector machine________________#

    print("Running Support Vector machine...")
    SVM_model = svm.SVC(kernel='rbf', gamma='auto', C=100)
    SVM_model = SVM_model.fit(X=X_train, y=y_train)

    y_pred = SVM_model.predict(X_train)
    print('accuracy train :', accuracy_score(y_train, y_pred))

    y_pred = SVM_model.predict(X_test)
    print('test accuracy: ',accuracy_score(y_test, y_pred))
    print("classification report")
    print(classification_report(y_test, y_pred))




