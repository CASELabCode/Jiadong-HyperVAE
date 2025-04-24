
# def wavelength_to_rgb(wavelength, brightness=0.8):
#     """
#     Convert a given wavelength of light to an RGB color value with adjustable brightness.
    
#     Parameters:
#     wavelength (float): Wavelength in nanometers (nm).
#     brightness (float): Brightness factor to adjust the color intensity (0.0 to 1.0).
    
#     Returns:
#     tuple: Corresponding RGB values as a tuple (R, G, B).
#     """
#     gamma = 0.8
#     intensity_max = 255
#     factor = 0.0
#     R = G = B = 0
    
#     if (380 <= wavelength <= 440):
#         R = -(wavelength - 440) / (440 - 380)
#         G = 0.0
#         B = 1.0
#     elif (440 <= wavelength <= 490):
#         R = 0.0
#         G = (wavelength - 440) / (490 - 440)
#         B = 1.0
#     elif (490 <= wavelength <= 510):
#         R = 0.0
#         G = 1.0
#         B = -(wavelength - 510) / (510 - 490)
#     elif (510 <= wavelength <= 580):
#         R = (wavelength - 510) / (580 - 510)
#         G = 1.0
#         B = 0.0
#     elif (580 <= wavelength <= 645):
#         R = 1.0
#         G = -(wavelength - 645) / (645 - 580)
#         B = 0.0
#     elif (645 <= wavelength <= 780):
#         R = 1.0
#         G = 0.0
#         B = 0.0
#     else:
#         R = G = B = 0.0

#     # Adjust intensity
#     if (380 <= wavelength <= 420):
#         factor = 0.3 + 0.7*(wavelength - 380) / (420 - 380)
#     elif (420 <= wavelength <= 645):
#         factor = 1.0
#     elif (645 <= wavelength <= 780):
#         factor = 0.3 + 0.7*(780 - wavelength) / (780 - 645)
#     else:
#         factor = 0.0

#     R = int(intensity_max * ((R * factor * brightness) ** gamma))
#     G = int(intensity_max * ((G * factor * brightness) ** gamma))
#     B = int(intensity_max * ((B * factor * brightness) ** gamma))

#     # Increase saturation
#     saturation_factor = 1.1  # Increase this factor to increase saturation
#     R = min(int(R * saturation_factor), 255)
#     G = min(int(G * saturation_factor), 255)
#     B = min(int(B * saturation_factor), 255)

#     return (R, G, B)

# def plot_column_distributions(df):

#     num_columns = df.shape[1]
#     wavelengths = np.linspace(400, 700, num_columns)
#     colors = [wavelength_to_rgb(wavelength) for wavelength in wavelengths]
#     colors = [tuple(val / 255 for val in color) for color in colors]

#     plt.figure(figsize=(10, 6))
#     box = plt.boxplot(df.values, patch_artist=True, showfliers=False, widths=0.8)

#     for patch, color in zip(box['boxes'], colors):
#         patch.set_facecolor(color)

#     tick_labels = np.arange(400, 701, 100)
#     tick_positions = [np.argmin(np.abs(wavelengths - label)) for label in tick_labels]

#     plt.xticks(tick_positions, tick_labels, fontsize=16, fontname='Times New Roman')
#     plt.yticks(fontsize=16, fontname='Times New Roman')  # 增大字体大小
#     plt.xlabel('Wavelength (nm)', fontsize=24, fontname='Times New Roman')
#     plt.ylabel('Value', fontsize=24, fontname='Times New Roman')

#     plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))

#     plt.tight_layout()
#     plt.show()