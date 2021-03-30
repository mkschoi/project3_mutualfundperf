from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import GridSearchCV


def gridsearch_cv_scores(X, y, model, param_grid, cv):
    '''
    input: a feature set, target, classification model, and the number of cross-validations to perform
    output: a list of accuracy, precision, recall, f1, roc, and log loss scores 
    '''
    if model == KNeighborsClassifier:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        grid = GridSearchCV(model(),param_grid,refit=True,n_jobs=-1)
        grid.fit(X, y)

        accuracy_cv = cross_val_score(grid, X, y, cv=cv, scoring='accuracy',n_jobs=-1).mean()
        precision_cv = cross_val_score(grid, X, y, cv=cv, scoring='precision',n_jobs=-1).mean()
        recall_cv = cross_val_score(grid, X, y, cv=cv, scoring='recall',n_jobs=-1).mean()
        f1_cv = cross_val_score(grid, X, y, cv=cv, scoring='f1',n_jobs=-1).mean()
        roc_cv = cross_val_score(grid, X, y, cv=cv, scoring='roc_auc',n_jobs=-1).mean()
        logloss_cv = -cross_val_score(grid, X, y, cv=cv, scoring='neg_log_loss',n_jobs=-1).mean()

        return [accuracy_cv, precision_cv, recall_cv, f1_cv, roc_cv, logloss_cv]
        
    elif model == SVC:
        grid = GridSearchCV(model(),param_grid,refit=True,n_jobs=-1)
        grid.fit(X, y)
    
        accuracy_cv = cross_val_score(grid, X, y, cv=cv, scoring='accuracy',n_jobs=-1).mean()
        precision_cv = cross_val_score(grid, X, y, cv=cv, scoring='precision',n_jobs=-1).mean()
        recall_cv = cross_val_score(grid, X, y, cv=cv, scoring='recall',n_jobs=-1).mean()
        f1_cv = cross_val_score(grid, X, y, cv=cv, scoring='f1',n_jobs=-1).mean()
        roc_cv = cross_val_score(grid, X, y, cv=cv, scoring='roc_auc',n_jobs=-1).mean()
        
        return [accuracy_cv, precision_cv, recall_cv, f1_cv, roc_cv]
        
    else:
        grid = GridSearchCV(model(),param_grid,refit=True,n_jobs=-1)
        grid.fit(X, y)

        accuracy_cv = cross_val_score(grid, X, y, cv=cv, scoring='accuracy',n_jobs=-1).mean()
        precision_cv = cross_val_score(grid, X, y, cv=cv, scoring='precision',n_jobs=-1).mean()
        recall_cv = cross_val_score(grid, X, y, cv=cv, scoring='recall',n_jobs=-1).mean()
        f1_cv = cross_val_score(grid, X, y, cv=cv, scoring='f1',n_jobs=-1).mean()
        roc_cv = cross_val_score(grid, X, y, cv=cv, scoring='roc_auc',n_jobs=-1).mean()
        logloss_cv = -cross_val_score(grid, X, y, cv=cv, scoring='neg_log_loss',n_jobs=-1).mean()

        return [accuracy_cv, precision_cv, recall_cv, f1_cv, roc_cv, logloss_cv]