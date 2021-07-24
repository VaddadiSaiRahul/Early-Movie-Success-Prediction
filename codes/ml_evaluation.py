import numpy as np
import os
import pickle
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.svm import SVR,SVC
from xgboost.sklearn import XGBRegressor,XGBClassifier
from sklearn.ensemble import GradientBoostingRegressor,AdaBoostRegressor,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.metrics import f1_score,cohen_kappa_score,matthews_corrcoef,mean_squared_error, precision_score,recall_score,accuracy_score
from sklearn.model_selection import cross_val_score

os.chdir("C:/Users/VADDADISAIRAHUL/Downloads/")
mse_threshold = 0.25

#np.random.seed(0)

def evaluate_classifier(clf_,Y_train_true,Y_test_true,X_train,X_test):
    Y_train_pred = clf_.predict(X_train)
    Y_test_pred = clf_.predict(X_test)
    #print(set(list(Y_train_pred)),set(list(Y_test_pred)))
    return [[f1_score(Y_train_true,Y_train_pred),f1_score(Y_test_true,Y_test_pred)],
            [cohen_kappa_score(Y_train_true,Y_train_pred),cohen_kappa_score(Y_test_true,Y_test_pred)]]

# to be used only if you want to change the model's weight    
def init_regressors_(X_train,Y_train_true):
    lr_ = LinearRegression().fit(X_train,Y_train_true)   
    svm_reg_ = SVR(gamma='scale').fit(X_train,Y_train_true)
    gbm_reg_ = GradientBoostingRegressor().fit(X_train,Y_train_true)
    ada_reg_ = AdaBoostRegressor().fit(X_train,Y_train_true)
    xgboost_reg_ = XGBRegressor().fit(X_train,Y_train_true)

    return [lr_,svm_reg_,gbm_reg_,ada_reg_,xgboost_reg_]

# to be used only if you want to change the model's weight 
def init_classifiers_(X_train,Y_train_true):
    lr_clf_ = LogisticRegression(solver='lbfgs').fit(X_train,Y_train_true)
    svm_clf_ = SVC(gamma='scale').fit(X_train,Y_train_true)
    gbm_clf_ = GradientBoostingClassifier().fit(X_train,Y_train_true)
    ada_clf_ = AdaBoostClassifier().fit(X_train,Y_train_true)
    xgboost_clf_ = XGBClassifier(use_label_encoder=False).fit(X_train,Y_train_true)

    return [lr_clf_,svm_clf_,gbm_clf_,ada_clf_,xgboost_clf_] 
    
def get_optimal_model_and_metrics(regressors,classifiers,X_train,X_test,Y1_train_true,Y1_test_true,Y2_train_true,Y2_test_true):
    best_classifier = None
    best_regressor = None
    optimal_mse_scores = None # added new
    optimal_f1_scores = None
    optimal_KK_scores = None

    for regressor in regressors:
        Y1_train_pred = regressor.predict(X_train).reshape(-1,1)
        Y1_test_pred = regressor.predict(X_test).reshape(-1,1)

        # these 4 lines, are for checking optimal regressor as well.
        optimal_mse_scores = [mean_squared_error(Y1_train_pred[:,0],Y1_train_true),mean_squared_error(Y1_test_pred[:,0],Y1_test_true)]

        for classifier in classifiers:
            f1_scores_list,KK_scores_list = evaluate_classifier(classifier,Y2_train_true,Y2_test_true,Y1_train_pred,Y1_test_pred)
            # only if train scores and greater than test scores
            if best_classifier == best_regressor == optimal_f1_scores == optimal_KK_scores == None:
                if (f1_scores_list[0]>f1_scores_list[1] and KK_scores_list[0]>KK_scores_list[1]):
                    best_classifier, best_regressor = classifier, regressor
                    optimal_f1_scores, optimal_KK_scores = f1_scores_list, KK_scores_list
            else:
                #train and test scores for f1 and cohens ; test data cohen's score and f1 score greater; this condition eliminates models trained on imbalanced data
                if (f1_scores_list[0]>f1_scores_list[1] and KK_scores_list[0]>KK_scores_list[1]) and (KK_scores_list[1]>optimal_KK_scores[1] and f1_scores_list[1]>optimal_f1_scores[1]):
                    best_classifier, best_regressor = classifier, regressor
                    optimal_f1_scores, optimal_KK_scores = f1_scores_list, KK_scores_list
              
    return [best_classifier,best_regressor,optimal_mse_scores,optimal_f1_scores,optimal_KK_scores]

def init_indices(len_):    
    return np.random.permutation(len_)

def get_optimal_model_and_metrics_final(regressor,classifier,X_train,X_test,Y1_train_true,Y2_train_true,Y1_test_true,Y2_test_true):
    Y1_train_pred = regressor.predict(X_train).reshape(-1,1)
    Y1_test_pred = regressor.predict(X_test).reshape(-1,1)
    
    Y2_train_pred = classifier.predict(Y1_train_pred)
    Y2_test_pred = classifier.predict(Y1_test_pred)

    return [mean_squared_error(Y1_test_true,Y1_test_pred[:,0]),matthews_corrcoef(Y2_test_true,Y2_test_pred),f1_score(Y2_test_true,Y2_test_pred),
            cohen_kappa_score(Y2_test_true,Y2_test_pred),precision_score(Y2_test_true,Y2_test_pred),recall_score(Y2_test_true,Y2_test_pred),
            accuracy_score(Y2_test_true,Y2_test_pred)]
    
def model_evaluation(data,target1,target2):
    m = data.shape[0]
    
    # to be used only if you want to update indices list and ml models weights
    '''
    indices = init_indices(m)
    
    X_train = data[indices[:int(0.9*m)],:]
    Y1_train_true = target1[indices[:int(0.9*m)]]
    Y2_train_true = target2[indices[:int(0.9*m)]]

    regressors = init_regressors_(X_train,Y1_train_true)
    Y1_train_true = Y1_train_true.reshape(-1,1)
    classifiers = init_classifiers_(Y1_train_true,Y2_train_true)

    file = open('merged_ml_dict_100d_100_1.pkl','wb')
    obj = [indices,regressors,classifiers]
    pickle.dump(obj,file)
    file.close()

    file = open('merged_ml_dict_100d_100_1.pkl','rb')
    ml_data = pickle.load(file)
    file.close()

    indices = ml_data[0]
    classifiers = ml_data[2]
    regressors = ml_data[1]

    X_train = data[indices[:int(0.9*m)],:]
    Y1_train_true = target1[indices[:int(0.9*m)]]
    Y2_train_true = target2[indices[:int(0.9*m)]]
    
    X_test = data[indices[int(0.9*m):],:]
    Y1_test_true = target1[indices[int(0.9*m):]]                                    
    Y2_test_true  = target2[indices[int(0.9*m):]]

    res = get_optimal_model_and_metrics(regressors,classifiers,X_train,X_test,Y1_train_true,Y1_test_true,Y2_train_true,Y2_test_true)

    print("Optimal Classifier :",res[0])
    print("Optimal Regressor :",res[1])
    print("Optimal MSE scores :",res[2])
    print("Optimal F1 scores :",res[3])
    print("Optimal Cohen's Kappa scores :",res[4])

    # save optimal regressor and classifier model
    file = open('optimal_merged_models_100d_100_1.pkl','wb')
    obj = [res[0],res[1]]
    pickle.dump(obj,file)
    file.close()
    '''
    file = open('merged_ml_dict_100d_10.pkl','rb')
    data1 = pickle.load(file)
    file.close()
    
    file = open('optimal_merged_models_100d_10.pkl','rb')
    data2 = pickle.load(file)
    file.close()
    
    indices = data1[0]
    classifiers = data2[0]
    regressors = data2[1]

    X_train = data[indices[:int(0.9*m)],:]
    Y1_train_true = target1[indices[:int(0.9*m)]]
    Y2_train_true = target2[indices[:int(0.9*m)]]
    
    X_test = data[indices[int(0.9*m):],:]
    Y1_test_true  = target1[indices[int(0.9*m):]]
    Y2_test_true  = target2[indices[int(0.9*m):]]
    
    res = get_optimal_model_and_metrics_final(regressors,classifiers,X_train,X_test,Y1_train_true,Y2_train_true,Y1_test_true,Y2_test_true)
    score1 = cross_val_score(regressors,data,target1,cv=10,scoring='neg_mean_squared_error')
    score2 = cross_val_score(classifiers,target1.reshape(-1,1),target2,cv=10)
    
    print("10-fold Cross Validation score :",abs(score1.mean()))
    print("Mean Squared Error for optimal regressor :",res[0])

    print("10-fold Cross Validation score for optimal classifier:",score2.mean())
    print("MCC Score for optimal classifier :",res[1])
    print("F1 score for optimal classifier :",res[2])
    print("Cohen's Kappa score for optimal classifier :",res[3])
    print("Precision score for optimal classifier :", res[4])
    print("Recall score for optimal classifier :",res[5])
    print("Accuracy Score for optimal classifier :",res[6])
    
    #print(res)
    #'''
