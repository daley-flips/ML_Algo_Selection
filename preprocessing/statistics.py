import torch
import os
from scipy.stats import kurtosis
from scipy.stats import skew
import pandas as pd
from scipy.stats import entropy
import numpy as np
import sys
# =============================================================================
# =============================================================================
#
#
#
# =============================================================================
# =============================================================================
columns = ['File', 'Mean_A', 'Max_A', 'Kurtosis_A', 'Skewness_A', 'Shannon Entropy_A', 'Stdev_A',
           'Mean_R', 'Max_R', 'Kurtosis_R', 'Skewness_R', 'Shannon Entropy_R', 'Stdev_R',
           'Mean_G', 'Max_G', 'Kurtosis_G', 'Skewness_G', 'Shannon Entropy_G', 'Stdev_G',
           'Mean_B', 'Max_B', 'Kurtosis_B', 'Skewness_B', 'Shannon Entropy_B', 'Stdev_B']
sample_level = pd.DataFrame(columns=columns)
# =============================================================================
# =============================================================================
#
#
#
# =============================================================================
# =============================================================================
directory_path = "PATH"
pt_files = [file for file in os.listdir(directory_path) if file.endswith('.pt')]

for file_name in pt_files: 
    file_path = os.path.join(directory_path, file_name) 
    colors = torch.load(file_path) 
# =============================================================================
# =============================================================================
# define arrays to be populated with the 720 values for that night
# =============================================================================
# =============================================================================
    activity = []
    red = []
    green = []
    blue = []
# =============================================================================
# =============================================================================
# populate those arrays
# =============================================================================
# =============================================================================
    for i in range(colors.size(0)):  # tensor.size(0) is the size of the first dimension
        for j in range(colors.size(1)):  # tensor.size(1) is the size of the second dimension
        
            if j == 0:
                activity.append(colors[i][j].item())
            if j == 1:
                red.append(colors[i][j].item())
            if j == 2:
                green.append(colors[i][j].item())
            if j == 3:
                blue.append(colors[i][j].item())
# =============================================================================
# =============================================================================
# now each array is populated
# check and ensure its only that nights data
# =============================================================================
# =============================================================================
    size = 720
    # size = 360
    if (len(activity) != size or
        len(red) != size or
        len(green) != size or
        len(blue) != size):
        
        print('night data was not pulled correctly')
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
# shannon entropy
# =============================================================================
# =============================================================================
    channel = activity
    probs = np.unique(channel, return_counts=True)[1]/len(channel)
    shannon_entropy_a = entropy(probs,base=2)
    # print("Shannon Entorpy:", shannon_entropy_a)
    
    channel = red
    probs = np.unique(channel, return_counts=True)[1]/len(channel)
    shannon_entropy_r = entropy(probs,base=2)
    # print("Shannon Entorpy:", shannon_entropy_r)
    
    channel = green
    probs = np.unique(channel, return_counts=True)[1]/len(channel)
    shannon_entropy_g = entropy(probs,base=2)
    # print("Shannon Entorpy:", shannon_entropy_g)
    
    channel = blue
    probs = np.unique(channel, return_counts=True)[1]/len(channel)
    shannon_entropy_b = entropy(probs,base=2)
    # print("Shannon Entorpy:", shannon_entropy_b)
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
# skewness
# =============================================================================
# =============================================================================
# do we want bias corrected? No
# use bias = True
    #print("For skews, a value of 0 is balanced\nPositive is a left skew\nNegative is a right skew")
    skew_a = skew(activity, axis=0, bias=True, nan_policy='raise', keepdims=False)
    skew_r = skew(red, axis=0, bias=True, nan_policy='raise', keepdims=False)
    skew_g = skew(green, axis=0, bias=True, nan_policy='raise', keepdims=False)
    skew_b = skew(blue, axis=0, bias=True, nan_policy='raise', keepdims=False)
    # sys.exit()
    # print("skew for activity: ", activity_skew)
    # print("skew for red: ", red_skew)
    # print("skew for green: ", green_skew)
    # print("skew for blue: ", blue_skew)
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
# kurtosis
# =============================================================================
# =============================================================================
# fishers or pearsons?
# we are using fishers = True 
    kurt_a = kurtosis(activity, axis=0, fisher=True, bias=True, nan_policy='raise', keepdims=False)
    kurt_r = kurtosis(red, axis=0, fisher=True, bias=True, nan_policy='raise', keepdims=False)
    kurt_g = kurtosis(green, axis=0, fisher=True, bias=True, nan_policy='raise', keepdims=False)
    kurt_b = kurtosis(blue, axis=0, fisher=True, bias=True, nan_policy='raise', keepdims=False)
    kurt_w = kurtosis(activity, axis=0, fisher=True, bias=True, nan_policy='raise', keepdims=False)
    # sys.exit()
    # print("kurtosis for activity: ", activity_kurt)
    # print("kurtosis for red: ",red_kurt)
    # print("kurtosis for green: ",green_kurt)
    # print("kurtosis for blue: ",blue_kurt)
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
# max-mean-std
# =============================================================================
# =============================================================================
    
    mean_A = np.mean(activity)
    mean_R = np.mean(red)
    mean_G = np.mean(green)
    mean_B = np.mean(blue)
    
    max_activity = np.max(activity)
    max_red = np.max(red)
    max_green = np.max(green)
    max_blue = np.max(blue)
    
    std_a = np.std(activity)
    std_r = np.std(red)
    std_g = np.std(green)
    std_b = np.std(blue)
# ============================================================================= 
# add to df
# =============================================================================
    # at this point we have all the variables we need, so lets add an entry to our df
    # 'File', 'Mean_A', 'Max_A', 'Kurtosis_A', 'Skewness_A, Shannon Entropy_A, Stdev _A',
    
    # shape: 1 row, 31 columns
    

    # columns = ['File', 'Mean_A', 'Max_A', 'Kurtosis_A', 'Skewness_A, Shannon Entropy_A, Stdev _A',
    
    data_dict = {'File': file_name, 'Mean_A': mean_A, 'Max_A': max_activity, 'Kurtosis_A': kurt_a, 'Skewness_A': skew_a, 'Shannon Entropy_A': shannon_entropy_a,'Stdev_A': std_a,
                 'Mean_R': mean_R, 'Max_R': max_red, 'Kurtosis_R': kurt_r, 'Skewness_R': skew_r, 'Shannon Entropy_R': shannon_entropy_r,'Stdev_R': std_r,
                 'Mean_G': mean_G, 'Max_G': max_green, 'Kurtosis_G': kurt_g, 'Skewness_G': skew_g, 'Shannon Entropy_G': shannon_entropy_g,'Stdev_G': std_g,
                 'Mean_B': mean_B, 'Max_B': max_blue, 'Kurtosis_B': kurt_b, 'Skewness_B': skew_b, 'Shannon Entropy_B': shannon_entropy_b,'Stdev_B': std_b}
    
    new_row = pd.DataFrame([data_dict])
    
    sample_level = pd.concat([sample_level, new_row], ignore_index=True)
    

# =============================================================================
# end add to df
# =============================================================================

# to CSV

path_to_save = "PATH/something.csv"
sample_level.to_csv(path_to_save, index=False)