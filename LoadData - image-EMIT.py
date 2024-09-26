import numpy as np
import os
import csv
import hypercoast
import matplotlib.pyplot as plt
import pandas as pd

# Set file path and data directory
data_dir = "F:\\aphy-chla-predictions\\image"
file_path = os.path.join(data_dir, "EMIT_L2A_RFL_001_20240404T161230_2409511_009.nc")

# Read EMIT dataset and filter out pixels where Rrs near 700nm > 0.03
EMIT_dataset = hypercoast.read_emit(file_path)
da = EMIT_dataset["reflectance"]
wl = da.wavelength.values  # Wavelengths
Rrs = da.values  # Reflectance
latitude = da.latitude.values  # 1D latitude array
longitude = da.longitude.values  # 1D longitude array

# Filter data where wavelengths are between 400 and 701 nm
indices = np.where((wl >= 400) & (wl <= 701))[0]
filtered_Rrs = Rrs[:, :, indices] / 3.14  # Filtered reflectance
filtered_wl = wl[indices]  # Filtered wavelengths

# Find the index for wavelength near 700nm
wl_700_idx = np.argmin(np.abs(filtered_wl - 700))

# Set all negative Rrs values to 0
filtered_Rrs[filtered_Rrs < 0] = 0

# File paths
coords_file = os.path.join(data_dir, 'coordinates_EMIT.csv')
reflectance_file = os.path.join(data_dir, 'reflectance_values_EMIT.csv')

# Initialize counters
coord_rows = 0
reflectance_rows = 0
deleted_rows = 0  # To track the number of rows skipped
reflectance_cols = len(filtered_wl)

# Latitude and longitude bounds
lon_min, lon_max = -95, -85
lat_min, lat_max = 28, 30.5

# Set step size
step_size = 3

with open(coords_file, 'w', newline='') as coord_f, open(reflectance_file, 'w', newline='') as refl_f:
    coord_writer = csv.writer(coord_f)
    refl_writer = csv.writer(refl_f)

    # Iterate through each pixel's latitude, longitude, and reflectance, selecting data based on the step size
    for i in range(0, filtered_Rrs.shape[0], step_size):  # Iterate over rows (latitude) with step size
        for j in range(0, filtered_Rrs.shape[1], step_size):  # Iterate over columns (longitude) with step size
            # Check if the pixel's latitude and longitude are within the specified bounds
            if lon_min <= longitude[j] <= lon_max and lat_min <= latitude[i] <= lat_max:
                # Check if the pixel's reflectance values contain NaN or Rrs near 700nm > 0.03
                if not np.isnan(filtered_Rrs[i, j, :]).any() and filtered_Rrs[i, j, wl_700_idx] <= 0.028:
                    # Save latitude and longitude (1D indexing)
                    coord_writer.writerow([f'{latitude[i]:.8f}', f'{longitude[j]:.8f}'])
                    coord_rows += 1

                    # Save reflectance values
                    refl_writer.writerow([f'{val:.8f}' for val in filtered_Rrs[i, j, :]])
                    reflectance_rows += 1
                else:
                    deleted_rows += 1  # Track rows skipped due to NaN or Rrs > 0.03 near 700nm

# Output the number of rows and columns saved in the CSV files
print(f"Coordinates saved to {coords_file}: {coord_rows} rows, 2 columns (Latitude, Longitude)")
print(f"Reflectance values saved to {reflectance_file}: {reflectance_rows} rows, {reflectance_cols} columns (Reflectance at different wavelengths)")
print(f"Rows with NaN values or Rrs > 0.03 at 700nm that were skipped: {deleted_rows}")

# Reshape the filtered_Rrs data into a 2D array where each column corresponds to one wavelength
reshaped_Rrs = filtered_Rrs.reshape(-1, filtered_Rrs.shape[2])

# Create a DataFrame for easier handling with columns as the wavelengths
df = pd.DataFrame(reshaped_Rrs, columns=filtered_wl)
df.columns = df.columns.astype(float)

# Plot a boxplot for the Rrs values at each wavelength
plt.figure(figsize=(12, 6))
df.boxplot(showfliers=False)

# Set x-ticks to display only 4 labels
wavelengths = df.columns
ticks_to_display = np.linspace(0, len(wavelengths) - 1, 4).astype(int)  # Select 4 evenly spaced ticks
plt.xticks(ticks_to_display, [f'{wavelengths[i]:.1f}' for i in ticks_to_display], rotation=90)

plt.xlabel('Wavelength (nm)')
plt.ylabel('Rrs')
plt.title('Distribution of Rrs Values at Different Wavelengths')
plt.tight_layout()
plt.show()
