from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import numpy as np
import pandas as pd
import time
# =============================================================================
# data
# =============================================================================
train = pd.read_csv('PATH/train.csv')
val = pd.read_csv('PATH/val.csv')
test = pd.read_csv('PATH/test.csv')

samples = pd.concat([train, val])
all_inputs = [
            # age, sex
            'age','female',
            
            # sleep variables
            'restdurat', 'restsdate', 'restedate', 'ACSL',
            'SNOOZE', 'ACSE', 'ACWASO', 'ACWKTOT', 
            'ACSLPTOT', 'ACBOUTOT','ACIMPCNT', 'ACFRAGID', 
           
            # sequential variables
            'Mean_A', 'Max_A', 'Kurtosis_A', 
            'Skewness_A', 'Shannon Entropy_A', 'Stdev_A',
            
            'Mean_R', 'Max_R', 'Kurtosis_R', 
            'Skewness_R', 'Shannon Entropy_R', 'Stdev_R',
            
            'Mean_G', 'Max_G', 'Kurtosis_G', 
            'Skewness_G', 'Shannon Entropy_G', 'Stdev_G',
            
            'Mean_B', 'Max_B', 'Kurtosis_B', 
            'Skewness_B', 'Shannon Entropy_B', 'Stdev_B',            
            ]
# =============================================================================
# =============================================================================
# =============================================================================
#
#
#
# =============================================================================
# iterate through all final models
# =============================================================================
models = {
# =============================================================================
    'lr_elasticnet': 
    (
     LogisticRegression(penalty='elasticnet', C=100, solver='saga', 
                               max_iter=10000, l1_ratio = 0.25,
                               class_weight={0: 1 / np.bincount(samples['class_mean'])[0], 
                                             1: (1+0.1) / np.bincount(samples['class_mean'])[1]})
    ),
# =============================================================================
    'lr_none': 
    (
     LogisticRegression(penalty=None,solver='saga', )
    ),
# =============================================================================
    'lr_l1': 
    (
     LogisticRegression(penalty='l1',solver='saga', )
    ),
# =============================================================================
    'lr_l2': 
    (
     LogisticRegression(penalty='l2',solver='saga', )
    ),
# =============================================================================
    'svm':
    (
     SVC(C=9, kernel='rbf', probability=True, tol=0.01, random_state=42,
         class_weight={0: 1 / np.bincount(samples['class_mean'])[0], 
                       1: (1+10) / np.bincount(samples['class_mean'])[1]})
    ),
# =============================================================================
    'mlp':
    (
     MLPClassifier(hidden_layer_sizes=(128, 512, 1024, 512, 128), 
                           learning_rate_init=0.01, activation='relu', 
                           solver='adam', max_iter=1000, tol=1e-5, 
                           verbose=False, random_state=42)
     ),   
# =============================================================================
    'rf':
    (
     RandomForestClassifier(criterion='entropy',
                                    max_features=None,
                                    n_estimators=200, random_state=42,
                                    class_weight= {0: 1 / np.bincount(samples['class_mean'])[0], 
                                                    1: (1+0.1) / np.bincount(samples['class_mean'])[1]})
     ),  
# =============================================================================
    'dt':
    (
    DecisionTreeClassifier(criterion='gini',
                                   max_depth=15,min_samples_split=5,min_samples_leaf=1,
                                   max_features='log2',random_state=42,
                                   class_weight={0: 1 / np.bincount(samples['class_mean'])[0], 
                                                   1: (1+0.1) / np.bincount(samples['class_mean'])[1]})
    ),
# =============================================================================
    'gb':
    (
     GradientBoostingClassifier(loss='exponential', learning_rate=0.01, max_features=None, 
                                        max_depth=None,random_state=42)
     ),
# =============================================================================
    'nb':
    (
     GaussianNB(var_smoothing=0.01)
     ),
# # =============================================================================
    'knn':
    (
     KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='auto')
     ),
# =============================================================================    
}
# =============================================================================
# =============================================================================
# =============================================================================
#
#
#
# =============================================================================
# =============================================================================
# init df
# =============================================================================
feature_sets = ["none", "sleep_variables", "activity", "red_light", "green_light", "blue_light", "gender"]
index = ['acc', 'auroc', 'sens', 'spec', 'training_time_(seconds)']  
multi_index_columns = pd.MultiIndex.from_product([models.keys(), feature_sets, ['val', 'test']], names=['algorithm', 'removed', 'testing'])
res = pd.DataFrame(index=index, columns=multi_index_columns)
# =============================================================================
# =============================================================================
def get_inputs(removal_type):
    if removal_type == "none":
        return all_inputs
    elif removal_type == "sleep_variables":
        return all_inputs[:2] + all_inputs[14:]
    elif removal_type == "activity":
        return all_inputs[:14] + all_inputs[20:]
    elif removal_type == "red_light":
        return all_inputs[:20] + all_inputs[26:]
    elif removal_type == "green_light":
        return all_inputs[:26] + all_inputs[32:]
    elif removal_type == "blue_light":
        return all_inputs[:32] + all_inputs[38:]
    elif removal_type == "gender":
        return [all_inputs[0]] + all_inputs[2:]
    return all_inputs
# =============================================================================
# =============================================================================
for model_name, model in models.items():
    for feature_set in feature_sets:
        inputs = get_inputs(feature_set)
# =============================================================================
# scale
# =============================================================================

        # ***********
        # start timer
        # ***********
        start_time = time.time()
        # ***********
        
        x_train = train[inputs]
        y_train = train['class_mean']
        x_val = val[inputs]
        y_val = val['class_mean']
        x_test = test[inputs]
        y_test = test['class_mean']
        
        scaler = MinMaxScaler()
        x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train), columns=inputs)
        x_val_scaled = pd.DataFrame(scaler.transform(x_val), columns=inputs)
        x_test_scaled = pd.DataFrame(scaler.transform(x_test), columns=inputs)
# ============================================================================
# train model
# =============================================================================
        model.fit(x_train_scaled, y_train)
        
        # ***********
        # end timer
        # ***********
        end_time = time.time()  # time to scale and fit model
        elapsed_time = end_time - start_time
        # ***********
        
        y_proba = model.predict_proba(x_val_scaled)[:, 1]  
        fpr, tpr, thresholds = roc_curve(y_val, y_proba, drop_intermediate=False)
        yj = sorted(zip(tpr-fpr, thresholds))[-1][1]
        y_pred = (y_proba >= yj).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()        
        res.loc['acc', (model_name, feature_set, 'val')] = round(accuracy_score(y_val, y_pred), 4)
        res.loc['auroc', (model_name, feature_set, 'val')] = round(roc_auc_score(y_val, y_proba), 4)
        res.loc['sens', (model_name, feature_set, 'val')] = round(tp / (tp + fn), 4)
        res.loc['spec', (model_name, feature_set, 'val')] = round(tn / (tn + fp), 4)
# =============================================================================
# test model
# =============================================================================    
        y_proba = model.predict_proba(x_test_scaled)[:, 1]  
        y_pred = (y_proba >= yj).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        test_accuracy = accuracy_score(y_test, y_pred)
        test_sensitivity =tp / (tp + fn) 
        test_specificity = tn / (tn + fp) 
        test_auroc = roc_auc_score(y_test, y_proba)
        res.loc['acc', (model_name, feature_set, 'test')] = round(accuracy_score(y_test, y_pred), 4)
        res.loc['auroc', (model_name, feature_set, 'test')] = round(roc_auc_score(y_test, y_proba), 4)
        res.loc['sens', (model_name, feature_set, 'test')] = round(tp / (tp + fn), 4)
        res.loc['spec', (model_name, feature_set, 'test')] = round(tn / (tn + fp), 4)
    
# =============================================================================    
# print debug, ensure acc matches
# =============================================================================       
        if len(inputs) == 38:
            print(f'\n{model_name}')
            print(f"val acc {res.loc['acc', (model_name, feature_set, 'val')]}")
            print(f"test acc {res.loc['acc', (model_name, feature_set, 'test')]}")
        
        # add time to df
        res.loc['training_time_(seconds)', (model_name, feature_set, 'val')] = round(elapsed_time, 4)
        


# =============================================================================       
# append lstm results without re-running
# =============================================================================       
# lstm_results = pd.read_csv('PATH/lstm_results.csv')
# for i, feature_set in enumerate(feature_sets):
#     # Assign each metric for 'val' from lstm_results
#     res.loc['acc', ('lstm', feature_set, 'val')] = lstm_results.iloc[2, i+1]
#     res.loc['auroc', ('lstm', feature_set, 'val')] = lstm_results.iloc[3, i+1]
#     res.loc['sens', ('lstm', feature_set, 'val')] = lstm_results.iloc[4, i+1]
#     res.loc['spec', ('lstm', feature_set, 'val')] = lstm_results.iloc[5, i+1]
    
#     # Assign each metric for 'test' from lstm_results
#     res.loc['acc', ('lstm', feature_set, 'test')] = lstm_results.iloc[2, i+8]  # Adjust column index for test metrics
#     res.loc['auroc', ('lstm', feature_set, 'test')] = lstm_results.iloc[3, i+8]
#     res.loc['sens', ('lstm', feature_set, 'test')] = lstm_results.iloc[4, i+8]
#     res.loc['spec', ('lstm', feature_set, 'test')] = lstm_results.iloc[5, i+8]

# =============================================================================       
# save
# =============================================================================       
# res.to_csv('all_results.csv')

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
