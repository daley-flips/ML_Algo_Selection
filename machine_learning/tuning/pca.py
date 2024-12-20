from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
import pandas as pd


train = pd.read_csv('PATH/train.csv')
val = pd.read_csv('PATH/val.csv')
inputs = [
    'restdurat', 'restsdate', 'restedate', 'ACSL', 'SNOOZE', 'ACSE', 'ACWASO', 
    'ACWKTOT', 'ACSLPTOT', 'ACBOUTOT', 'ACIMPCNT', 'ACFRAGID', 'Mean_A', 'Max_A', 
    'Kurtosis_A', 'Skewness_A', 'Shannon Entropy_A', 'Stdev_A', 'Mean_R', 'Max_R', 
    'Kurtosis_R', 'Skewness_R', 'Shannon Entropy_R', 'Stdev_R', 'Mean_G', 'Max_G', 
    'Kurtosis_G', 'Skewness_G', 'Shannon Entropy_G', 'Stdev_G', 'Mean_B', 'Max_B', 
    'Kurtosis_B', 'Skewness_B', 'Shannon Entropy_B', 'Stdev_B',  'age', 'female']

x_train = train[inputs]
x_val = val[inputs]
y_train = train["P-mean"]
y_val = val["P-mean"]

# Scale data before performing PCA
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.transform(x_val)

# Perform PCA
pca = PCA()
pca.fit(x_train_scaled)

# Visualize PCA results
cumulative_variance = pca.explained_variance_ratio_.cumsum()
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance vs Number of Components')
plt.show()

# Show PCA results, select 85+% explained variance
print("Cumulative variance: {}".format(cumulative_variance))


# Choose our new number of principle components to do ML with
res = []
for components in range(1, 39):
    pca = PCA(n_components=components)
    x_train_pca = pca.fit_transform(x_train_scaled)
    x_val_pca = pca.transform(x_val_scaled)
    
    
    # linear regression, get R^2 values
    lr = LinearRegression()
    lr.fit(x_train_pca, y_train)
    y_pred = lr.predict(x_val_pca)
    r2 = r2_score(y_val, y_pred)
    print(f'R² score: {r2}')
    
    res.append([components, round(r2, 4)])
    
    
# res_df = pd.DataFrame(res, columns=["#PrincipleComponents", "r^2"])
# res_df.to_csv('pca_results.csv', index=False)


