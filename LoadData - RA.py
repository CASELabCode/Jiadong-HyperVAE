import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np


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
    return df.iloc[1:, columns]


def wavelength_to_rgb(wavelength, brightness=0.8):
    """
    Convert a given wavelength of light to an RGB color value with adjustable brightness.
    
    Parameters:
    wavelength (float): Wavelength in nanometers (nm).
    brightness (float): Brightness factor to adjust the color intensity (0.0 to 1.0).
    
    Returns:
    tuple: Corresponding RGB values as a tuple (R, G, B).
    """
    gamma = 0.8
    intensity_max = 255
    factor = 0.0
    R = G = B = 0
    
    if (380 <= wavelength <= 440):
        R = -(wavelength - 440) / (440 - 380)
        G = 0.0
        B = 1.0
    elif (440 <= wavelength <= 490):
        R = 0.0
        G = (wavelength - 440) / (490 - 440)
        B = 1.0
    elif (490 <= wavelength <= 510):
        R = 0.0
        G = 1.0
        B = -(wavelength - 510) / (510 - 490)
    elif (510 <= wavelength <= 580):
        R = (wavelength - 510) / (580 - 510)
        G = 1.0
        B = 0.0
    elif (580 <= wavelength <= 645):
        R = 1.0
        G = -(wavelength - 645) / (645 - 580)
        B = 0.0
    elif (645 <= wavelength <= 780):
        R = 1.0
        G = 0.0
        B = 0.0
    else:
        R = G = B = 0.0

    # Adjust intensity
    if (380 <= wavelength <= 420):
        factor = 0.3 + 0.7*(wavelength - 380) / (420 - 380)
    elif (420 <= wavelength <= 645):
        factor = 1.0
    elif (645 <= wavelength <= 780):
        factor = 0.3 + 0.7*(780 - wavelength) / (780 - 645)
    else:
        factor = 0.0

    R = int(intensity_max * ((R * factor * brightness) ** gamma))
    G = int(intensity_max * ((G * factor * brightness) ** gamma))
    B = int(intensity_max * ((B * factor * brightness) ** gamma))

    # Increase saturation
    saturation_factor = 1.1  # Increase this factor to increase saturation
    R = min(int(R * saturation_factor), 255)
    G = min(int(G * saturation_factor), 255)
    B = min(int(B * saturation_factor), 255)

    return (R, G, B)

def plot_column_distributions(df):

    num_columns = df.shape[1]
    wavelengths = np.linspace(400, 700, num_columns)
    colors = [wavelength_to_rgb(wavelength) for wavelength in wavelengths]
    colors = [tuple(val / 255 for val in color) for color in colors]

    plt.figure(figsize=(10, 6))
    box = plt.boxplot(df.values, patch_artist=True, showfliers=False, widths=0.8)

    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    tick_labels = np.arange(400, 701, 100)
    tick_positions = [np.argmin(np.abs(wavelengths - label)) for label in tick_labels]

    plt.xticks(tick_positions, tick_labels, fontsize=16, fontname='Times New Roman')
    plt.yticks(fontsize=16, fontname='Times New Roman')  # 增大字体大小
    plt.xlabel('Wavelength (nm)', fontsize=24, fontname='Times New Roman')
    plt.ylabel('Value', fontsize=24, fontname='Times New Roman')

    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))

    plt.tight_layout()
    plt.show()

def plot_spectral_distribution(df, wavelengths, label):
    """
    Plot a single spectral distribution boxplot with wavelength-colored boxes.
    
    Parameters:
    df: DataFrame containing spectral data.
    wavelengths: List of corresponding wavelengths.
    label: Label for the plot.
    """
    colors = [wavelength_to_rgb(wv) for wv in wavelengths]
    colors = [tuple(c/255 for c in color) for color in colors]
    
    plt.figure(figsize=(8, 5))
    box = plt.boxplot(df.values, patch_artist=True, showfliers=False, widths=0.8)
    
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.ylabel('$R_{rs}$ Value', fontsize=18, fontname='Times New Roman')
    plt.ylim(bottom=0)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.3f}'))
    
    tick_labels = np.arange(400, 701, 100)
    tick_positions = [np.argmin(np.abs(np.array(wavelengths) - label)) for label in tick_labels]
    
    plt.xticks(tick_positions, tick_labels, fontsize=16, fontname='Times New Roman')
    plt.yticks(fontsize=16, fontname='Times New Roman')
    plt.xlabel('Wavelength (nm)', fontsize=20, fontname='Times New Roman')
    
    plt.tight_layout()
    plt.show()

# Data Path
file1 = 'F:/aphy-chla-predictions/Data/Clean/aph/Global+AGID_aphy_filtered.xlsx'
file2 = 'F:/aphy-chla-predictions/Data/Clean/aph/Global+AGID_Rrs_filtered.xlsx'

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


wv = [int(val) for val in wv_EMIT]

start1=400
start2=400

columns = [col - start1 +1  for col in wv] 
columns2 = [col - start2 +1 for col in wv] 

print(columns2)

df1 = read_columns(file1, 'aph', columns)
df2 = read_columns(file2, 'Rrs', columns2)


invalid_rows_df1 = (df1.isna() | (df1 < 0)).any(axis=1)
invalid_rows_df2 = (df2.isna() | (df2 < 0)).any(axis=1)


invalid_rows = invalid_rows_df1 | invalid_rows_df2

# Delete these rows and reset the index
df1_clean = df1[~invalid_rows].reset_index(drop=True)
df2_clean = df2[~invalid_rows].reset_index(drop=True)


df1_clean.to_csv('F:/aphy-chla-predictions/Data/Clean/aphy_EMIT.csv', index=False, header=False, float_format='%.5f')
df2_clean.to_csv('F:/aphy-chla-predictions/Data/Clean/Rrs_EMIT.csv', index=False, header=False, float_format='%.5f')

print("Cleaned data has been saved to CSV files, and deleted rows information has been saved to deleted_rows.txt.")


plot_spectral_distribution(df2_clean, wv, "R_{rs}")
#plot_spectral_distribution(df1_clean, wv, "a_{phy}")








    
# output_dir = 'F:/aphy-chla-predictions/Data/Real/Distribution_Plots'
# os.makedirs(output_dir, exist_ok=True)

# for col in range(df1_clean.shape[1]):
#     plt.figure(figsize=(10, 8))
#     plt.hist(df1_clean.iloc[:, col], bins=np.linspace(0, 2, 40), alpha=0.7, color='blue', edgecolor='black')
#     plt.xticks(np.linspace(0, 2, 6), fontsize=20, fontname='Times New Roman')
#     plt.yticks(fontsize=20, fontname='Times New Roman')
#     plt.xlim(-0.2, 2.2)  # 固定x轴范围为0-5
#     plt.xlabel('Value', fontsize=28, fontname='Times New Roman')
#     plt.ylabel('Frequency', fontsize=28, fontname='Times New Roman')   
#     plt.savefig(f'{output_dir}/Rrs410_690_Column_{col}.pdf', bbox_inches='tight')
#     plt.close()



