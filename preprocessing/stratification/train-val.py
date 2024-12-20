import pandas as pd
from sklearn.model_selection import train_test_split
# =============================================================================
# stratify train into a train+val
# =============================================================================
samples = pd.read_csv('PATH/training_data.csv')

# Split participants into train, val, test on stratification params
u_df = samples[['ID', 'key']].drop_duplicates()
train_ids, val_ids = train_test_split(
    u_df['ID'], test_size=0.2, random_state=42, stratify=u_df['key']
)
train = samples[samples['ID'].isin(train_ids)]
val = samples[samples['ID'].isin(val_ids)]

# =============================================================================
#
#
#
# =============================================================================
# Verification between training and validation
# =============================================================================
train_size = len(train)
val_size = len(val)
total_size = train_size + val_size
train_percentage = (train_size / total_size) * 100
val_percentage = (val_size / total_size) * 100
print(f'Training set size: {train_size} samples ({train_percentage:.2f}%)')
print(f'Validation set size: {val_size} samples ({val_percentage:.2f}%)')


print('\nTraining distribution')

train_counts = train['class_mean'].value_counts()
# Baseline accuracy for training set (standard to beat)
train_baseline_accuracy = train_counts.max() / train_counts.sum() * 100
print(f'Baseline accuracy (Training): {train_baseline_accuracy:.2f}%')
print(train_counts)


print('\nValidation distribution')
val_counts = val['class_mean'].value_counts()
# Baseline accuracy for validation set (standard to beat)
val_baseline_accuracy = val_counts.max() / val_counts.sum() * 100
print(f'Baseline accuracy (Validation): {val_baseline_accuracy:.2f}%')
print(val_counts)
# =============================================================================
#
#
#
# =============================================================================
# Gender distribution in training and validation datasets
# =============================================================================
train_gender_counts = train['female'].value_counts()
train_gender_percentages = (train_gender_counts / train_gender_counts.sum()) * 100
print('\nTraining gender percentages:')
print(train_gender_percentages)

val_gender_counts = val['female'].value_counts()
val_gender_percentages = (val_gender_counts / val_gender_counts.sum()) * 100
print('\nValidation gender percentages:')
print(val_gender_percentages)

# =============================================================================
# age distribution in training and validation datasets
# =============================================================================
train_gender_counts = train['age_cat'].value_counts()
train_gender_percentages = (train_gender_counts / train_gender_counts.sum()) * 100
print('\nTraining age percentages:')
print(train_gender_percentages)

val_gender_counts = val['age_cat'].value_counts()
val_gender_percentages = (val_gender_counts / val_gender_counts.sum()) * 100
print('\nValidation age percentages:')
print(val_gender_percentages)

# train.to_csv('PATH/train.csv', index=False)
# val.to_csv('PATH/val.csv', index=False)


