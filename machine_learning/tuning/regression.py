# =============================================================================
# run all base models
# =============================================================================
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
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
    "LinearRegression",
    "Lasso",
    "Ridge",
    "ElasticNet"
])
# =============================================================================
# =============================================================================
x_train = train[inputs]
y_train = train['P-mean']
x_val = val[inputs]
y_val = val['P-mean']
scaler = MinMaxScaler()
x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train), columns=inputs)
x_val_scaled = pd.DataFrame(scaler.transform(x_val), columns=inputs)
# =============================================================================
# =============================================================================
models = [   
     LinearRegression(),
     ElasticNet(l1_ratio = 1),  # Lasso
     ElasticNet(l1_ratio = 0),  # Ridge
     ElasticNet(l1_ratio = 0.5)  # elastic net 
     ]
    
for model, model_name in zip(models, res.columns):
    model.fit(x_train_scaled, y_train)
    y_pred = model.predict(x_val_scaled)
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)  # Calculate RMSE
    r2 = model.score(x_val_scaled, y_val)
    
    metrics = [round(rmse, 4), 
               round(r2, 4)]
    res[model_name] = metrics

res.index = ["RMSE", "r^2"]
res.to_csv('base_regression_results.csv', index=True)

    
    
    
    
    
    
    
