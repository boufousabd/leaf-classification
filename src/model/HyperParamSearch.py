from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, KFold
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn import neighbors

class HyperParamSearch:
    def __init__(self, xtrain, ytrain):
        self.xtrain = xtrain
        self.ytrain = ytrain

    def SVMSearch(self):
        parameters_grid = [
            {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
            {'C': [1, 10, 100, 1000], 'kernel': ['poly'], 'degree': list(range(2, 9)),
             'coef0': list(np.linspace(0.000001, 2, num=10))},
            {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': ['scale', 'auto'],
             'gamma': list(np.linspace(0.000001, 2, num=10))},
            {'C': [1, 10, 100, 1000], 'kernel': ['sigmoid'], 'coef0': list(np.linspace(0.000001, 2, num=10))}
        ]
        svc = svm.SVC()
        clf = GridSearchCV(svc, parameters_grid, cv=5, n_jobs=-1)
        clf.fit(self.xtrain, self.ytrain)
        return clf.best_params_

    def SGDSearch(self):
        parameters_grid = [{'alpha': list(np.linspace(0.000001, 2, num=10))}]
        clf_sgd = SGDClassifier(max_iter=10000, penalty='l1', shuffle=True, loss='hinge', tol=1e-3)
        clf_sgd_grid_search = make_pipeline(StandardScaler(),
                                            GridSearchCV(clf_sgd, parameters_grid, scoring='f1_macro', cv=5))
        clf_sgd_grid_search.fit(self.xtrain, self.ytrain)
        return clf_sgd_grid_search.named_steps['gridsearchcv'].best_params_

    def AdaBoostSearch(self):
        param_grid_ada = {"base_estimator__criterion": ["gini", "entropy"],
                          "base_estimator__splitter": ["best", "random"],
                          "n_estimators": list(range(0, 100, 10))
                          }
        DecisionTCl = DecisionTreeClassifier(random_state=11, max_features="log2", max_depth=None)
        AdaBoostCl = AdaBoostClassifier(base_estimator=DecisionTCl)
        grid_search_AdaBoostCl = make_pipeline(StandardScaler(),
                                               GridSearchCV(AdaBoostCl, param_grid=param_grid_ada, scoring='f1_macro',
                                                            cv=5, n_jobs=-1))
        grid_search_AdaBoostCl.fit(self.xtrain, self.ytrain)
        return grid_search_AdaBoostCl.named_steps['gridsearchcv'].best_params_

    def RandomForestSearch(self):
        param_grid_rand = {
            'criterion': ['gini', 'entropy'],
            'n_estimators': list(range(0, 20)),
            'max_depth': list(range(0, 20))
        }
        randomforestcl = RandomForestClassifier(n_jobs=-1, max_features='log2', oob_score=True)
        grid_search_randomforestcl = make_pipeline(StandardScaler(),
                                                   GridSearchCV(randomforestcl, param_grid=param_grid_rand,
                                                                scoring='f1_macro', cv=5, n_jobs=-1))
        grid_search_randomforestcl.fit(self.xtrain, self.ytrain)
        return grid_search_randomforestcl.named_steps['gridsearchcv'].best_params_

    def SVMOneAgainstAllSearch(self):
        stratifiedKflod = StratifiedKFold(n_splits=5, random_state=11, shuffle=True)
        svm = LinearSVC(max_iter=100, dual=False)
        params = {'C': np.logspace(-3, 3, 7), 'multi_class': ["crammer_singer", "ovr"], 'penalty': ["l1", "l2"]}
        GridSearchCV_svm = GridSearchCV(svm, params, scoring='accuracy', cv=stratifiedKflod)
        GridSearchCV_svm.fit(self.xtrain, self.ytrain)
        return GridSearchCV_svm.best_params_

    def KNNSearch(self):
        skf = StratifiedKFold(n_splits=5, random_state=4, shuffle=True)
        params = {'n_neighbors': np.arange(2, 15)}
        knn = neighbors.KNeighborsClassifier()
        gs_knn = GridSearchCV(knn, params, cv=skf, scoring='accuracy')
        gs_knn.fit(self.xtrain, self.ytrain)
        return gs_knn.best_params_