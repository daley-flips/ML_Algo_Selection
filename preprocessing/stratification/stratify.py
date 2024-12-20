from sklearn.model_selection import train_test_split

import pandas as pd


# Load the data
df = pd.read_csv("PATH/all_data.csv")

# Create a high/low age bin
df['age_cat'] = df['age'] > 16

# Combine all the vars we need to stratify on
df['key'] = df['Study'] + '_' + df['female'].astype(str) + '_' + df['age_cat'].astype(str)

# Get all unique participant and stratify key pairings
u_df = df[['ID', 'key']].drop_duplicates()

# Split participants into train, val, test on stratification params
train, test = train_test_split(u_df, test_size=0.38, random_state=42, stratify=u_df['key'])
test, val = train_test_split(test, test_size=0.5, random_state=42, stratify=test['key'])

# Create train, val, test data splits by how the participants were split
dat_t = df[df['ID'].isin(train['ID'])]
dat_v = df[df['ID'].isin(val['ID'])]
dat_T = df[df['ID'].isin(test['ID'])]

# Check the variables to make sure they are roughly even across the splits
var = 'class_mean'
print(dat_t[var].value_counts() / len(dat_t))
print(dat_v[var].value_counts() / len(dat_v))
print(dat_T[var].value_counts() / len(dat_T))

# dat_t.to_csv('train.csv', index=False)
# dat_v.to_csv('val.csv', index=False)
# dat_T.to_csv('test.csv', index=False)