import numpy as np
from utils import all_x, all_y
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score
seed = 42

beta_grid = np.concatenate((np.linspace(.1, .9, 9), np.linspace(1, 10, 10)))


def reports(test_y, test_y_hat):
    print(classification_report(test_y, test_y_hat))
    print(confusion_matrix(test_y, test_y_hat))
    auroc = roc_auc_score(test_y, test_y_hat)
    f1_macro = f1_score(test_y, test_y_hat, average='macro')
    print(f'AUROC= {auroc:.4f}')
    print(f'F1 macro= {f1_macro:.4f}')
    return f1_macro

###################################
## K-nearest Neighbours


def KNN(data, n_neighbors, mode='cv', n_folds=5):
    train_x, test_x, train_y, test_y = data

    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    if mode == 'cv':
        f1_list = cross_val_score(knn, all_x(data), all_y(data),
                                   cv=n_folds,
                                   scoring='f1_macro',
                                   n_jobs=-1)
        return None, f1_list
    elif mode == 'train':
        knn = knn.fit(train_x, train_y)
        test_y_hat = knn.predict(test_x)

        f1 = reports(test_y, test_y_hat)
        return knn, f1
    else:
        raise ValueError('Mode should be either cv or train.')


def KNN_grid_search(data, n_neighbors_grid=range(1, 11)):

    n_neighbors_opt, f1_opt = 0, 0
    for k in n_neighbors_grid:
        _, f1_list = KNN(data,
                          n_neighbors=k,
                          mode='cv')
        f1_mean, f1_std = f1_list.mean(), f1_list.std()
        print(f'KNN k={k}: CV F1 macro= [{f1_mean:.4f}] + [{f1_std:.4f}]')
        if f1_mean > f1_opt:
            n_neighbors_opt, f1_opt = k, f1_mean

    knn, f1 = KNN(data,
                      n_neighbors=n_neighbors_opt,
                      mode='train')
    print(f'\nKNN k={n_neighbors_opt}: F1 macro= {f1:.4f}')
    return knn, f1

###################################
## Logistic Regression


def LogReg(data, penalty, beta, mode='cv', n_folds=5, seed=seed):
    train_x, test_x, train_y, test_y = data

    from sklearn.linear_model import LogisticRegression
    logreg = LogisticRegression(penalty=penalty,
                                C=beta,
                                class_weight='balanced',
                                max_iter=1000)
    if mode == 'cv':
        f1_list = cross_val_score(logreg, all_x(data), all_y(data),
                                   cv=n_folds,
                                   scoring='f1_macro',
                                   n_jobs=-1)
        return None, f1_list
    elif mode == 'train':
        logreg = logreg.fit(train_x, train_y)
        test_y_hat = logreg.predict(test_x)

        f1 = reports(test_y, test_y_hat)
        return logreg, f1
    else:
        raise ValueError('Mode should be either cv or train.')

def LogReg_grid_search(data, penalty_grid=['l2', 'l1'], beta_grid=beta_grid):
    penalty_opt, beta_opt, f1_opt = '', 0, 0
    for penalty in penalty_grid:
        for beta in beta_grid:
            _, f1_list = LogReg(data,
                                 penalty=penalty,
                                 beta=beta,
                                 mode='cv')
            f1_mean, f1_std = f1_list.mean(), f1_list.std()
            print(
                f'{penalty.capitalize()} LogReg β={beta:.1f}: CV F1 macro= [{f1_mean:.4f}] + [{f1_std:.4f}]')
            if f1_mean > f1_opt:
                penalty_opt, beta_opt, f1_opt = penalty, beta, f1_mean

    logreg, f1 = LogReg(data,
                         penalty=penalty_opt,
                         beta=beta_opt,
                         mode='train')
    print(
        f'\n{penalty_opt.capitalize()} LogReg β={beta_opt:.1f}: F1 macro= {f1:.4f}')
    return logreg, f1

###################################
## Support-vector machine


def SVM(data, beta, mode='cv', n_folds=5, seed=seed):
    train_x, test_x, train_y, test_y = data

    from sklearn.svm import SVC
    svm = SVC(C=beta,
              kernel='rbf',
              class_weight='balanced',
              random_state=seed)

    if mode == 'cv':
        f1_list = cross_val_score(svm, all_x(data), all_y(data),
                                   cv=n_folds,
                                   scoring='f1_macro',
                                   n_jobs=-1)
        return None, f1_list
    elif mode == 'train':
        svm = svm.fit(train_x, train_y)
        test_y_hat = svm.predict(test_x)
        f1 = reports(test_y, test_y_hat)
        return svm, f1
    else:
        raise ValueError('Mode should be either cv or train.')


def SVM_grid_search(data, beta_grid=beta_grid):
    beta_opt, f1_opt = 0, 0
    for beta in beta_grid:
        _, f1_list = SVM(data,
                          beta=beta,
                          mode='cv')
        f1_mean, f1_std = f1_list.mean(), f1_list.std()
        print(
            f'SVM β={beta:.1f}: CV F1 macro= [{f1_mean:.4f}] + [{f1_std:.4f}]')
        if f1_mean > f1_opt:
            beta_opt, f1_opt = beta, f1_mean

    svm, f1 = SVM(data,
                   beta=beta_opt,
                   mode='train')
    print(f'\nSVM β={beta_opt:.1f}: F1 macro= {f1:.4f}')
    return svm, f1

###################################
## Random Forest


def RandomForest(data, depth, mode='f1', seed=seed):
    train_x, test_x, train_y, test_y = data

    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators=1000,
                                    max_depth=depth,
                                    oob_score=True,
                                    class_weight='balanced',
                                    n_jobs=-1,
                                    random_state=seed)
    forest = forest.fit(train_x, train_y)
    test_y_hat = forest.predict(test_x)

    if mode == 'oob':
        acc_oob = forest.oob_score_
        return forest, acc_oob
    elif mode == 'f1' or mode == 'f1_macro':
        f1 = reports(test_y, test_y_hat)
        return forest, f1
    else:
        raise ValueError('Mode should be either oob or f1.')

def RandomForest_grid_search(data, depth_grid=range(2, 11)):
    depth_opt, f1_opt = 0, 0
    for depth in depth_grid:
        _, f1 = RandomForest(data,
                                  depth=depth,
                                  mode='f1')
        print(f'RandomForest depth={depth}: F1 macro= [{f1:.4f}]')
        if f1 > f1_opt:
            depth_opt, f1_opt = depth, f1

    forest, f1 = RandomForest(data,
                               depth=depth_opt,
                               mode='f1')
    print(f'\nRandomForest depth={depth_opt}: F1 macro= {f1:.4f}')
    return forest, f1

###################################
## Bagging


def Bagging(data, mode='f1', seed=seed):
    train_x, test_x, train_y, test_y = data

    from sklearn.ensemble import BaggingClassifier
    bag = BaggingClassifier(n_estimators=1000,
                            oob_score=True,
                            n_jobs=-1,
                            random_state=seed)
    bag = bag.fit(train_x, train_y)
    test_y_hat = bag.predict(test_x)

    if mode == 'oob':
        acc_oob = bag.oob_score_
        return bag, acc_oob
    elif mode == 'f1':
        f1 = reports(test_y, test_y_hat)
        return bag, f1
    else:
        raise ValueError('Mode should be either oob or f1.')

###################################
## Decision Tree

def DecisionTree(data, depth=None, seed=seed):
    train_x, test_x, train_y, test_y = data

    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(max_depth=depth,
                                  class_weight='balanced',
                                  random_state=seed)
    tree = tree.fit(train_x, train_y)
    test_y_hat = tree.predict(test_x)

    f1 = reports(test_y, test_y_hat)
    return tree, f1

###################################
## Extra Trees


def ExtraTrees(data, depth=None, mode='train', seed=seed):
    train_x, test_x, train_y, test_y = data

    from sklearn.ensemble import ExtraTreesClassifier
    extra = ExtraTreesClassifier(n_estimators=1000,
                                 max_depth=depth,
                                 oob_score=mode == 'oob',
                                 bootstrap=mode == 'oob',
                                 class_weight='balanced',
                                 n_jobs=-1,
                                 random_state=seed)
    extra = extra.fit(train_x, train_y)
    test_y_hat = extra.predict(test_x)

    if mode == 'oob':
        acc_oob = extra.oob_score_
        print(f'ExtraTrees: OOB Accuracy= [{acc_oob:.4f}]')
        auroc = reports(test_y, test_y_hat)
        return extra, acc_oob
    elif mode == 'train':
        f1 = reports(test_y, test_y_hat)
        return extra, f1
    else:
        raise ValueError('Mode should be either oob or train.')

###################################
## Gradient Boosting


def GradB(data, seed=seed):
    train_x, test_x, train_y, test_y = data

    from sklearn.ensemble import GradientBoostingClassifier
    gradb = GradientBoostingClassifier(n_estimators=1000,
                                       random_state=seed)
    gradb = gradb.fit(train_x, train_y)
    test_y_hat = gradb.predict(test_x)

    f1 = reports(test_y, test_y_hat)
    return gradb, f1

###################################
## AdaBoost


def AdaB(data, seed=seed):
    train_x, test_x, train_y, test_y = data

    from sklearn.ensemble import AdaBoostClassifier
    adab = AdaBoostClassifier(n_estimators=1000,
                              random_state=seed)
    adab = adab.fit(train_x, train_y)
    test_y_hat = adab.predict(test_x)

    f1 = reports(test_y, test_y_hat)
    return adab, f1

###################################
## XGBoost


def XGB(data, seed=seed):
    train_x, test_x, train_y, test_y = data

    from xgboost import XGBClassifier
    xgb = XGBClassifier(random_state=seed)
    xgb = xgb.fit(train_x, train_y)
    test_y_hat = xgb.predict(test_x)

    f1 = reports(test_y, test_y_hat)
    return xgb, f1

###################################
## Naïve Bayes


def NBayes(data, priors=None):
    train_x, test_x, train_y, test_y = data

    from sklearn.naive_bayes import GaussianNB
    bayes = GaussianNB(priors=priors)
    bayes = bayes.fit(train_x, train_y)
    test_y_hat = bayes.predict(test_x)

    f1 = reports(test_y, test_y_hat)
    return bayes, f1

###################################
## Gaussian Process


def GP(data, seed=seed):
    train_x, test_x, train_y, test_y = data

    from sklearn.gaussian_process import GaussianProcessClassifier
    gp = GaussianProcessClassifier(n_jobs=-1,
                                   random_state=seed)
    gp = gp.fit(train_x, train_y)
    test_y_hat = gp.predict(test_x)

    f1 = reports(test_y, test_y_hat)
    return gp, f1

###################################
## Majority Voting

def Voting(data, models, model_name, weights='None', how_to_vote='hard'):
    train_x, test_x, train_y, test_y = data
    model_zip = list(zip(model_name, models))

    from sklearn.ensemble import VotingClassifier
    vote = VotingClassifier(estimators=model_zip,
                            voting=how_to_vote,
                            weights=weights,
                            n_jobs=-1)
    vote = vote.fit(train_x, train_y)
    test_y_hat = vote.predict(test_x)

    f1 = reports(test_y, test_y_hat)

    return vote, f1
