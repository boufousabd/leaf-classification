from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import neighbors
from src.model import HyperParamSearch


class Classifiers:
    def SVMClassifier(self, xtrain, ytrain):
        Hyperparam = HyperParamSearch.HyperParamSearch(xtrain, ytrain)
        best_param = Hyperparam.SVMSearch()
        svc = svm.SVC(**best_param).fit(xtrain, ytrain)
        return svc

    def GDClassifier(self, xtrain, ytrain):
        Hyperparam = HyperParamSearch.HyperParamSearch(xtrain, ytrain)
        best_param = Hyperparam.SGDSearch()
        SGDclf = make_pipeline(StandardScaler(),
                               SGDClassifier(max_iter=10000, penalty='l1', **best_param, shuffle=True, loss='hinge',
                                             tol=1e-3))
        SGDclf.fit(xtrain, ytrain)
        return SGDclf

    def AdaBoostClassifier(self, xtrain, ytrain):
        Hyperparam = HyperParamSearch.HyperParamSearch(xtrain, ytrain)
        best_param = Hyperparam.AdaBoostSearch()
        DecisionTCl = DecisionTreeClassifier(random_state=11, max_features="log2", max_depth=None, **best_param)
        AdaBoostCl = AdaBoostClassifier(base_estimator=DecisionTCl)
        AdaBoostCl = make_pipeline(StandardScaler(), AdaBoostCl)
        AdaBoostCl.fit(xtrain, ytrain)
        return AdaBoostCl

    def RandomForestClassifier(self, xtrain, ytrain):
        Hyperparam = HyperParamSearch.HyperParamSearch(xtrain, ytrain)
        best_param = Hyperparam.RandomForestSearch()
        randomforestcl = RandomForestClassifier(n_jobs=-1, **best_param, max_features='log2', oob_score=True)
        randomforestcl = make_pipeline(StandardScaler(),
                                       randomforestcl)
        randomforestcl.fit(xtrain, ytrain)
        return randomforestcl

    def SVMOneAgainstAllClassifier(self, xtrain, ytrain):
        Hyperparam = HyperParamSearch.HyperParamSearch(xtrain, ytrain)
        best_param = Hyperparam.SVMOneAgainstAllSearch()
        svm = LinearSVC(max_iter=100, dual=False, **best_param)
        svm.fit(xtrain, ytrain)
        return svm

    def KNNClassifier(self, xtrain, ytrain):
        Hyperparam = HyperParamSearch.HyperParamSearch(xtrain, ytrain)
        best_param = Hyperparam.KNNSearch()
        knn = neighbors.KNeighborsClassifier(**best_param)
        knn.fit(xtrain, ytrain)
        return knn