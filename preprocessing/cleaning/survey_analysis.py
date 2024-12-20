import pandas as pd
from datetime import datetime
import sys
# =============================================================================
# =============================================================================
#
#
#
# =============================================================================
# =============================================================================
# load all survey data
# =============================================================================
# =============================================================================
cart_wake = pd.read_csv('PATH/cart_wake.csv')
cart_day = pd.read_csv('PATH//cart_day.csv')
cart_bed = pd.read_csv('PATH//cart_bed.csv')

early_wake = pd.read_csv('PATH//slate_wake_early.csv')
early_day = pd.read_csv('PATH//slate_day_early.csv')
early_bed = pd.read_csv('PATH//slate_bed_early.csv')

late_wake = pd.read_csv('PATH//slate_wake_late.csv')
late_day = pd.read_csv('PATH//slate_day_late.csv')
late_bed = pd.read_csv('PATH//slate_bed_late.csv')

carrs = pd.read_csv('PATH//carrs_survey.csv')
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
#
#
#
# =============================================================================
# =============================================================================
# re-format cart
# =============================================================================
# =============================================================================
# new df, columns = study, id, date, p-affect

# start with cart
cart = pd.concat([cart_wake, cart_day, cart_bed], ignore_index=True)

# create df
temp = pd.DataFrame(columns=['study', 'ID', 'date', 'p-affect'])
temp['ID'] = cart['ID']
temp['study'] = 'cart'
temp['date'] = cart['CDate']
temp['p-affect'] = cart['PANASSFPA']

# nan check
cart = temp.dropna()
print(f'cart lost {len(temp)-len(cart)} surveys to nan\n')

# re-format dates to ignore time
cart.loc[:, 'date'] = cart['date'].str.split(' ').str[0]

# sort
cart = cart.sort_values(by=['ID', 'date'])
cart = cart.reset_index(drop=True)

# nan check
before_drop = len(cart)
cart = cart.dropna()
if before_drop != len(cart):
    print('**invalid date parsing**')
    sys.exit()

# keep only days with 3 or more surveys
grouped = cart.groupby(['ID', 'date']).size().reset_index(name='counts')
valid_dates = grouped[grouped['counts'] >= 3][['ID', 'date']]
cart = pd.merge(cart, valid_dates, on=['ID', 'date'], how='inner')
cart = cart.reset_index(drop=True)

# print(cart)
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
#
#
#
# =============================================================================
# =============================================================================
# repeat for early + late
# =============================================================================
# =============================================================================

slate = pd.concat([early_wake, early_day, early_bed,
                   late_wake, late_day, late_bed], 
                  ignore_index=True)

# create df
temp = pd.DataFrame(columns=['study', 'ID', 'date', 'p-affect'])
temp['ID'] = slate['ID']
temp['study'] = 'slate'
temp['date'] = slate['CDate']
temp['p-affect'] = slate['PANASSFPA']

# nan check
slate = temp.dropna()
print(f'slate lost {len(temp)-len(slate)} surveys to nan\n')

# re-format dates to ignore time
slate.loc[:, 'date'] = slate['date'].str.split(' ').str[0]

# sort
slate = slate.sort_values(by=['ID', 'date'])
slate = slate.reset_index(drop=True)

# nan check
before_drop = len(slate)
slate = slate.dropna()
if before_drop != len(slate):
    print('**invalid date parsing**')
    sys.exit()

# keep only days with 3 or more surveys
grouped = slate.groupby(['ID', 'date']).size().reset_index(name='counts')
valid_dates = grouped[grouped['counts'] >= 3][['ID', 'date']]
slate = pd.merge(slate, valid_dates, on=['ID', 'date'], how='inner')
slate = slate.reset_index(drop=True)

# print(slate)
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
#
#
#
# =============================================================================
# =============================================================================
# repeat with carrs
# =============================================================================
# =============================================================================

# create df
temp = pd.DataFrame(columns=['study', 'ID', 'date', 'p-affect'])
temp['ID'] = carrs['ParticipantID']
temp['study'] = 'carrs'
temp['date'] = carrs['CDate']
temp['p-affect'] = carrs['PANASSFPA']

# nan check
carrs = temp.dropna()
print(f'carrs lost {len(temp)-len(carrs)} surveys to nan\n')

# re-format dates to ignore time
carrs.loc[:, 'date'] = carrs['date'].str.split(' ').str[0]

# sort
carrs = carrs.sort_values(by=['ID', 'date'])
carrs = carrs.reset_index(drop=True)

# nan check
before_drop = len(carrs)
carrs = carrs.dropna()
if before_drop != len(carrs):
    print('**invalid date parsing**')
    sys.exit()

# keep only days with 3 or more surveys
grouped = carrs.groupby(['ID', 'date']).size().reset_index(name='counts')
valid_dates = grouped[grouped['counts'] >= 3][['ID', 'date']]
carrs = pd.merge(carrs, valid_dates, on=['ID', 'date'], how='inner')
carrs = carrs.reset_index(drop=True)

# print(carrs)
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
#
#
#
# =============================================================================
# =============================================================================
# concat all
# =============================================================================
# =============================================================================
al = pd.concat([cart, slate, carrs], ignore_index=True)

before_drop = len(al)
al = al.dropna()

if len(al) != before_drop:
    print('**invalid file**')
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
#
#
#
# =============================================================================
# =============================================================================
# pull ml samples
# =============================================================================
# =============================================================================
train = pd.read_csv('C:/Users/dfras/.spyder-py3/Sleep Research/samples/training_raw.csv')
test = pd.read_csv('C:/Users/dfras/.spyder-py3/Sleep Research/samples/test_raw.csv')
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
#
#
#
# =============================================================================
# =============================================================================
# function to ensure all dates are full
# some dates like 2018 were pulled in as just '18' which didnt match on inner join
# =============================================================================
# =============================================================================


def ensure_full_year(date_str):
    parts = date_str.split('/')
    if len(parts[-1]) == 2:  # Check if the year is in two-digit format
        if int(parts[-1]) <= 99:  # Assuming dates are in the 1900s or 2000s
            parts[-1] = '20' + parts[-1] if int(parts[-1]) < 50 else '19' + parts[-1]
    return '/'.join(parts)


def format_date_to_mmddyyyy(date_str):
    try:
        # Convert the date to datetime
        date = pd.to_datetime(date_str)
        # Format manually to avoid leading zeros
        return f"{date.month}/{date.day}/{date.year}"
    except Exception as e:
        # Handle any exceptions that might occur during conversion
        print(f"Error converting date: {date_str} - {e}")
        return date_str
    

al['date'] = al['date'].apply(ensure_full_year)
al['date'] = al['date'].apply(format_date_to_mmddyyyy)
test['date'] = test['date'].apply(ensure_full_year)
test['date'] = test['date'].apply(format_date_to_mmddyyyy)
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
#
#
#
# =============================================================================
# =============================================================================
# create set of tuples
# (ID, date)
# O(1) lookup to see if any given row is in our ml samples
# =============================================================================
# =============================================================================

id_date = set()

for idx, row in train.iterrows():
    id_date.add((row['ID'], row['date']))
    
for idx, row in test.iterrows(): 
    id_date.add((row['ID'], row['date']))

# print(id_date)
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
#
#
#
# =============================================================================
# =============================================================================
# QA check
# =============================================================================
# =============================================================================

# Combine train and test DataFrames
combined_samples = pd.concat([train, test])

# Create a set of tuples from the combined samples
combined_set = set((row['ID'], row['date']) for idx, row in combined_samples.iterrows())

# Find the difference between combined_set and id_date
missing_in_id_date = combined_set - id_date

# Print the missing tuples
if len(missing_in_id_date) != 0:
    print(f"Missing tuples in id_date: {missing_in_id_date}")
    sys.exit()

# Check for duplicates in train and test
train_duplicates = train.duplicated(subset=['ID', 'date']).sum()
test_duplicates = test.duplicated(subset=['ID', 'date']).sum()

if train_duplicates != 0 or test_duplicates != 0:
    print(f"Duplicate rows in train: {train_duplicates}")
    print(f"Duplicate rows in test: {test_duplicates}")
    sys.exit()

# Check for duplicates in the combined samples
combined_samples_duplicates = combined_samples.duplicated(subset=['ID', 'date']).sum()

if combined_samples_duplicates != 0:
    print(f"Duplicate rows in combined samples: {combined_samples_duplicates}")
    sys.exit()

# Identify the duplicate row in the test DataFrame
duplicate_rows = test[test.duplicated(subset=['ID', 'date'], keep=False)]

if len(duplicate_rows) != 0:
    print("Duplicate row in test DataFrame:")
    print(duplicate_rows)
    sys.exit()
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
#
#
#
# =============================================================================
# =============================================================================
# drop rows not in ml set
# =============================================================================
# =============================================================================
rows_to_drop = []
for idx, row in al.iterrows():
    if ((row['ID'], row['date'])) not in id_date:
        rows_to_drop.append(idx)
        
al.drop(rows_to_drop, inplace=True)
al.reset_index(drop=True, inplace=True)

print(al)
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
#
#
#
# =============================================================================
# =============================================================================
# QA check
# =============================================================================
# =============================================================================

new_set = set()

for idx, row in train.iterrows():
    new_set.add((row['ID'], row['date']))
    
for idx, row in test.iterrows(): 
    new_set.add((row['ID'], row['date']))

# Combine train and test DataFrames
combined_samples = pd.concat([train, test])

# Create a set of tuples from the combined samples
combined_set = set((row['ID'], row['date']) for idx, row in combined_samples.iterrows())

# Find the difference between combined_set and new_set
missing_in_new_set = combined_set - new_set

# Print the missing tuples
if len(missing_in_new_set) != 0:
    print(f"Missing tuples in new_set: {missing_in_new_set}")
    sys.exit()
    
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
before_drop = len(al)
al = al.dropna()

if len(al) != before_drop:
    print('**invalid file**')
    
# al.to_csv('all_ml_surveys.csv', index=False)










