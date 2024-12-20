import pandas as pd
import torch
import os
import sys
from datetime import datetime
# =============================================================================
# =============================================================================
data = pd.read_csv(r'PATH/ml_data.csv', encoding='latin1')
directory_path = 'PATH/actigraphy.csv'
# =============================================================================
#
#
#
# =============================================================================
# =============================================================================
# for each day in data:
#    1. the the actigraphy file
#    2. get the data from that night and make a tensor
# =============================================================================
# =============================================================================

x=0  # tracks number of pt files
samples_lost_to_nan = []

for index, row in data.iterrows():
    # print(x) 

# =============================================================================
# pull data
# =============================================================================
    start = row['start index']
    end = row['end index']
    ID = row['ID']
    date = row["Date"]
# =============================================================================
    
    csv_files = [file for file in os.listdir(directory_path) if file.endswith('.csv')]

    for file_name in csv_files:
        file_path = os.path.join(directory_path, file_name)
        if file_name[:6] == str(ID):
            try:
                night = pd.read_csv(file_path)
            except pd.errors.EmptyDataError:
                continue
# =============================================================================
# =============================================================================
# actigrapy file is found
# =============================================================================
# =============================================================================
            if end < len(night):
                check_dates = night.iloc[end]
                date_actigraphy = check_dates['date']
                date_actigraphy = datetime.strptime(date_actigraphy, '%m/%d/%Y')
                date_actigraphy = date_actigraphy.strftime('%Y-%m-%d')
                if str(date) == str(date_actigraphy):
# =============================================================================
# =============================================================================
# day is found, now get the data
# =============================================================================
# =============================================================================
                    night = night[start:end]
                    night = night[['activity', 'red_light', 'green_light', 'blue_light']]
                    
                    nan_cells = night.isna().any().any()  # NaN detection
                    if nan_cells:
                        print("NAN:", file_name)
                        samples_lost_to_nan.append(f"{ID}_{date}")
                        
                    else:
                        my_tensor = torch.tensor(night.values, dtype=torch.float32)
                        if my_tensor.shape[0]== 0:
                            print('error')
                            sys.exit()
                            # print(night)
                            # print(file_name)
                            # print("oh this is just an edge case when that person has 2 actigraphy files")
                        else:
                            date = row['Date']
                            filename = f"{ID}_{date}.pt"
                            filename = filename.replace('/', '_')
                            path_to_save = 'PATH'
                            # print('emsure this is 720-4:')
                            # print(my_tensor.shape)
                            
                            signals = night[['activity', 'red_light', 'green_light', 'blue_light']].values
                            
                            tensor = torch.tensor(signals, dtype=torch.float32)

                            # Check the shape of the tensor
                            print(tensor.shape)
                            
                            if 720 != tensor.shape[0]:
                                print("incorrect shape------------------------------------------------------------")
                                sys.exit()
                            print(filename)
                            torch.save(tensor, path_to_save + filename)
                            
                            
                            # ************************
                            # for downsampling
                            # ************************
                            # # Initialize an empty list to hold the decimated signals
                            # decimated_signals = []
                            
                            # # Decimate each column separately
                            # for i in range(signals.shape[1]):  # Assuming signals is a 2D array with shape (n_samples, n_channels)
                            #     decimated_channel = signal.decimate(signals[:, i], 2, ftype='fir')
                            #     decimated_signals.append(decimated_channel)
                            
                            # # Convert the list of arrays back into a single 2D array
                            # decimated_array = np.stack(decimated_signals, axis=-1)
                            
                            # # Convert the decimated numpy array into a PyTorch tensor
                            # decimated_tensor = torch.tensor(decimated_array, dtype=torch.float32)
                            
                            # # Check the shape of the tensor
                            # print(decimated_tensor.shape)
                         
                            # if 360 != decimated_tensor.shape[0]:
                            #     print("incorrect shape------------------------------------------------------------")
                            #     sys.exit()
                            # print(filename)
                            
                            
                            # torch.save(decimated_tensor, path_to_save + filename)
                            # x+=1
                else:
                    print(str(date))
                    print(str(date_actigraphy))
                    print(f'no matching day for {file_name}******************')
                    continue
            else:
                print(f'no matching day for {file_name}')
                continue
                
