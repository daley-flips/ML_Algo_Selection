from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
import pandas as pd


train = pd.read_csv('PATH/train.csv')
val = pd.read_csv('PATH/val.csv')
test = pd.read_csv('PATH/test.csv')

tv = pd.concat([train, val])
samples = pd.concat([tv, test])

inputs = [
    'restdurat', 'restsdate', 'restedate', 'ACSL', 'SNOOZE', 'ACSE', 'ACWASO', 
    'ACWKTOT', 'ACSLPTOT', 'ACBOUTOT', 'ACIMPCNT', 'ACFRAGID', 'Mean_A', 'Max_A', 
    'Kurtosis_A', 'Skewness_A', 'Shannon Entropy_A', 'Stdev_A', 'Mean_R', 'Max_R', 
    'Kurtosis_R', 'Skewness_R', 'Shannon Entropy_R', 'Stdev_R', 'Mean_G', 'Max_G', 
    'Kurtosis_G', 'Skewness_G', 'Shannon Entropy_G', 'Stdev_G', 'Mean_B', 'Max_B', 
    'Kurtosis_B', 'Skewness_B', 'Shannon Entropy_B', 'Stdev_B',  'age', 'female']

x = samples[inputs]

y = samples["P-mean"]


# Scale data before performing PCA
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Perform PCA
pca = PCA()
pca.fit(x_scaled)

# Visualize PCA results
cumulative_variance = pca.explained_variance_ratio_.cumsum()
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance vs Number of Components')
plt.axhline(y=0.85, color='r', linestyle='dashed')
plt.show()

# Show PCA results, select 85+% explained variance
print("Cumulative variance: {}".format(cumulative_variance))


# Choose our new number of principle components to do ML with
res = []
for components in range(1, 39):
    pca = PCA(n_components=components)
    x_pca = pca.fit_transform(x_scaled)
    
    # linear regression, get R^2 values
    lr = LinearRegression()
    lr.fit(x_pca, y)
    y_pred = lr.predict(x_pca)
    r2 = r2_score(y, y_pred)
    print(f'R² score: {r2}')
    
    res.append([components, round(r2, 4)])
    
    
res_df = pd.DataFrame(res, columns=["#PrincipleComponents", "r^2"])
res_df.to_csv('pca_all_data.csv', index=False)


