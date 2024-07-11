
import matplotlib.pyplot as plt
import pandas as pd
import os

# Function to replace values <= 0 with the nearest non-zero value in the same column and record changes
def replace_nonpositive_with_nearest(df):
    changes = []
    for col in range(df.shape[1]):
        for i in range(len(df)):
            if df.iloc[i, col] <= 0:
                original_value = df.iloc[i, col]
                if i > 0 and df.iloc[i - 1, col] > 0:
                    df.iloc[i, col] = df.iloc[i - 1, col]
                elif i < len(df) - 1 and df.iloc[i + 1, col] > 0:
                    df.iloc[i, col] = df.iloc[i + 1, col]
                changes.append((i, col, original_value, df.iloc[i, col]))
    return df, changes


def read_columns(file, sheet_name, columns, skiprows=1):
    df = pd.read_excel(file, sheet_name=sheet_name, header=None)
    return df.iloc[2:, columns]

# Read three xlsx files
file2 = 'F:/Geo/Data/Real/Rrs_real_Sep.xlsx'
file3 = 'F:/Geo/Data/Real/Chl a_real_Sep.xlsx'


wv_HICO = [
    404.080, 409.808, 415.536, 421.264, 426.992, 432.720, 438.448, 444.176, 449.904, 
    455.632, 461.360, 467.088, 472.816, 478.544, 484.272, 490.000, 495.728, 501.456, 
    507.184, 512.912, 518.640, 524.368, 530.096, 535.824, 541.552, 547.280, 553.008, 
    558.736, 564.464, 570.192, 575.920, 581.648, 587.376, 593.104, 598.832, 604.560, 
    610.288, 616.016, 621.744, 627.472, 633.200, 638.928, 644.656, 650.384, 656.112, 
    661.840, 667.568, 673.296, 679.024, 684.752, 690.480, 696.208

]


wv_PACE = [
    400, 403, 405, 408, 410, 413, 415, 418, 420, 422, 425, 427, 430, 432, 435, 437, 
    440, 442, 445, 447, 450, 452, 455, 457, 460, 462, 465, 467, 470, 472, 475, 477, 
    480, 482, 485, 487, 490, 492, 495, 497, 500, 502, 505, 507, 510, 512, 515, 517, 
    520, 522, 525, 527, 530, 532, 535, 537, 540, 542, 545, 547, 550, 553, 555, 558, 
    560, 563, 565, 568, 570, 573, 575, 578, 580, 583, 586, 588, 591, 593, 596, 598, 
    601, 603, 605, 608, 610, 613, 615, 618, 620, 623, 625, 627, 630, 632, 635, 637, 
    640, 641, 642, 643, 645, 646, 647, 648, 650, 651, 652, 653, 655, 656, 657, 658, 
    660, 661, 662, 663, 665, 666, 667, 668, 670, 671, 672, 673, 675, 676, 677, 678, 
    679, 681, 682, 683, 684, 686, 687, 688, 689, 691, 692, 693, 694, 696, 697, 698, 
    699
]

wv_EMIT = [
    403.2254, 410.638, 418.0536, 425.47214, 432.8927, 
    440.31726, 447.7428, 455.17035, 462.59888, 470.0304, 477.46292, 484.89743, 492.33292, 
    499.77142, 507.2099, 514.6504, 522.0909, 529.5333, 536.9768, 544.42126, 551.8667, 
    559.3142, 566.7616, 574.20905, 581.6585, 589.108, 596.55835, 604.0098, 611.4622, 
    618.9146, 626.36804, 633.8215, 641.2759, 648.7303, 656.1857, 663.6411, 671.09753, 
    678.5539, 686.0103, 693.4677, 700.9251
]

wv = [int(val) for val in wv_HICO]

start=350

columns = [col - start +1  for col in wv] 

print(columns)

df2 = read_columns(file2, 'Rrs', columns)
df3 = pd.read_excel(file3, sheet_name='Chl-a',header=None).iloc[2:, 1:]


# Replace non-positive values in df1 and df2
df2, changes2 = replace_nonpositive_with_nearest(df2)

# Identify rows with NaN values in either file
nan_indices_df2 = pd.isnull(df2).any(axis=1)
nan_indices_df3 = pd.isnull(df3).any(axis=1)

# Align indices
#nan_indices_df2.index = df2.index
#nan_indices_df3.index = df3.index

nan_indices = nan_indices_df2 | nan_indices_df3

# Delete these rows and reset the index
df2_clean = df2[~nan_indices].reset_index(drop=True)
df3_clean = df3[~nan_indices].reset_index(drop=True)


# Find the indices of the top 20% largest values in df3_clean
threshold = df3_clean.quantile(0.90)
large_indices = df3_clean[df3_clean >= threshold].dropna().index

# Save the indices of the rows to be deleted to a local file
with open('F:/Geo/Data/Real/deleted_rows_indices.txt', 'w') as f:
    for index in large_indices:
        f.write(f"{index}\n")

# Delete these rows in both df3_clean and df2_clean
df3_clean = df3_clean.drop(index=large_indices).reset_index(drop=True)
df2_clean = df2_clean.drop(index=large_indices).reset_index(drop=True)


# Save the cleaned data to CSV files
df2_clean.to_csv('F:/Geo/Data/Real/Rrs_RC_HICO_Sep.csv', index=False, header=False,float_format='%.5f')
df3_clean.to_csv('F:/Geo/Data/Real/Chl_RC_HICO_Sep.csv', index=False, header=False,float_format='%.5f')



# output_dir = 'F:/Geo/Data/Real/Distribution_Plots'
# os.makedirs(output_dir, exist_ok=True)

# for col in range(df2_clean.shape[1]):
#     plt.figure(figsize=(10, 5))
#     plt.hist(df2_clean.iloc[:, col], bins=100, alpha=0.7, color='blue', edgecolor='black')
#     plt.title(f'Distribution of Rrs410_690 Column {col}')
#     plt.xlabel('Value')
#     plt.ylabel('Frequency')
    
#     plt.savefig(f'{output_dir}/Rrs410_690_Column_{col}.png')
#     plt.close()


# plt.figure(figsize=(10, 5))
# plt.hist(df3_clean.values.flatten(), bins=50, alpha=0.7, color='green', edgecolor='black')
# plt.title('Distribution of Chl Values')
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.show()


# Save the cleaned data to CSV files
#df1_clean.to_csv('F:/Geo/Data/Real/aphy410_690.csv', index=False, header=False,float_format='%.5f')
#df2_clean.to_csv('F:/Geo/Data/Real/Rrs410_690.csv', index=False, header=False,float_format='%.5f')
#df3_clean.to_csv('F:/Geo/Data/Real/Chl.csv', index=False, header=False,float_format='%.5f')

#print("Cleaned data has been saved to CSV files, and deleted rows information has been saved to deleted_rows.txt.")
