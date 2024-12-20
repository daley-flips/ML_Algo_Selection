# =============================================================================
# run all base models
# =============================================================================
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import pandas as pd
# =============================================================================
# =============================================================================
train = pd.read_csv('PATH/train.csv')
val = pd.read_csv('PATH/val.csv')
# =============================================================================
# =============================================================================
inputs = [
            # age, sex
            'age', 
            'female',
            
            # sleep variables
            'restdurat', 'restsdate', 'restedate', 'ACSL',
            'SNOOZE', 'ACSE', 'ACWASO', 'ACWKTOT', 
            'ACSLPTOT', 'ACBOUTOT','ACIMPCNT', 'ACFRAGID', 
           
            # sequential variables
            'Mean_A', 
            'Max_A', 
            'Kurtosis_A', 
            'Skewness_A', 
            'Shannon Entropy_A', 
            'Stdev_A',
            
            'Mean_R', 
            'Max_R', 
            'Kurtosis_R', 
            'Skewness_R', 
            'Shannon Entropy_R', 
            'Stdev_R',
            
            'Mean_G', 
            'Max_G', 
            'Kurtosis_G', 
            'Skewness_G', 
            'Shannon Entropy_G', 
            'Stdev_G',
            
            'Mean_B', 
            'Max_B', 
            'Kurtosis_B', 
            'Skewness_B', 
            'Shannon Entropy_B', 
            'Stdev_B',            
            ]
# =============================================================================
# =============================================================================
res = pd.DataFrame(columns=[
    "BaseLogisticRegression", 
    "NoneLogisticRegression", 
    "l1LogisticRegression", 
    "l2LogisticRegression", 
    "elasticnetLogisticRegression", 

    
    "DecisionTreeClassifier", 
    "RandomForestClassifier", 
    "GradientBoostingClassifier", 
    "SVM", 
    "GaussianNB", 
    "KNeighbors",
    "MLP"
])
# =============================================================================
# =============================================================================
x_train = train[inputs]
y_train = train['class_mean']
x_val = val[inputs]
y_val = val['class_mean']
scaler = MinMaxScaler()
x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train), columns=inputs)
x_val_scaled = pd.DataFrame(scaler.transform(x_val), columns=inputs)
# =============================================================================
# =============================================================================
models = [   
     LogisticRegression(),
     LogisticRegression(penalty=None, solver='saga'),
     LogisticRegression(penalty='l1', solver='saga'),
     LogisticRegression(penalty='l2', solver='saga'),
     LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5),
     
     DecisionTreeClassifier(random_state=42),
     RandomForestClassifier(random_state=42),
     GradientBoostingClassifier(random_state=42),
     SVC(probability=True, random_state=42),
     GaussianNB(),
     KNeighborsClassifier(),
     MLPClassifier(max_iter=2000, random_state=42)
     ]
    
for model, model_name in zip(models, res.columns):
    model.fit(x_train_scaled, y_train)
    y_proba = model.predict_proba(x_val_scaled)[:, 1]  
    fpr, tpr, thresholds = roc_curve(y_val, y_proba, drop_intermediate=False)
    yj = sorted(zip(tpr-fpr, thresholds))[-1][1]
    y_pred = (y_proba >= yj).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    
    
    accuracy = accuracy_score(y_val, y_pred)
    sensitivity =tp / (tp + fn)  
    specificity = tn / (tn + fp) 
    auroc = roc_auc_score(y_val, y_proba)
    
    print(accuracy)
    
    metrics = [round(accuracy, 4), 
               round(auroc, 4), 
               round(sensitivity, 4), 
               round(specificity, 4)]
    res[model_name] = metrics

res.index = ["Accuracy", "AUROC", "Sensitivity", "Specificity"]
res.to_csv('base_classifier_results.csv', index=True)