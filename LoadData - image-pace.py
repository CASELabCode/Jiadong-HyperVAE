import numpy as np
import os
import csv
import hypercoast
import matplotlib.pyplot as plt
import pandas as pd

# File paths
data_dir = "F:\\aphy-chla-predictions\\image"
file_path = os.path.join(data_dir, "PACE_OCI.20240515T182840.L2.OC_AOP.V2_0.NRT.nc")

# Read PACE dataset
PACE_dataset = hypercoast.read_pace(file_path)
da = PACE_dataset["Rrs"]
wl = da.wavelength.values  # Wavelength
Rrs = da.values  # Reflectance (Rrs)
latitude = da.latitude.values  # Latitude
longitude = da.longitude.values  # Longitude

# Filter data for wavelengths between 400 and 699 nm
indices = np.where((wl >= 400) & (wl <= 699))[0]
filtered_Rrs = Rrs[:, :, indices]  # Filtered Rrs values
filtered_wl = wl[indices]  # Filtered wavelengths

# Set all negative Rrs values to 0
filtered_Rrs[filtered_Rrs < 0] = 0

# File paths for output
coords_file = os.path.join(data_dir, 'coordinates_PACE.csv')
reflectance_file = os.path.join(data_dir, 'reflectance_values_PACE.csv')

# Initialize counters
coord_rows = 0
reflectance_rows = 0
deleted_rows = 0  # Track skipped rows due to NaN values
skipped_by_coord = 0  # Track skipped rows due to out-of-bound coordinates
reflectance_cols = len(filtered_wl)

# Latitude and longitude bounds
lon_min, lon_max = -95, -85
lat_min, lat_max = 28, 30.5

# Open CSV files for writing
with open(coords_file, 'w', newline='') as coord_f, open(reflectance_file, 'w', newline='') as refl_f:
    coord_writer = csv.writer(coord_f)
    refl_writer = csv.writer(refl_f)
    
    # Loop through each pixel's latitude, longitude, and reflectance
    for i in range(filtered_Rrs.shape[0]):  # Latitude (rows)
        for j in range(filtered_Rrs.shape[1]):  # Longitude (columns)
            # Get latitude and longitude of the current pixel
            lat = latitude[i, j]
            lon = longitude[i, j]
            
            # Check if latitude and longitude are within the desired range
            if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
                # Check if the reflectance values do not contain NaN
                if not np.isnan(filtered_Rrs[i, j, :]).any():
                    # Save latitude and longitude if valid
                    coord_writer.writerow([f'{lat:.5f}', f'{lon:.5f}'])
                    coord_rows += 1

                    # Save reflectance values
                    refl_writer.writerow([f'{val:.5f}' for val in filtered_Rrs[i, j, :]])
                    reflectance_rows += 1
                else:
                    deleted_rows += 1  # Increment counter for skipped NaN rows
            else:
                skipped_by_coord += 1  # Increment counter for out-of-bound coordinates

# Output statistics
print(f"Coordinates saved to {coords_file}: {coord_rows} rows, 2 columns (Latitude, Longitude)")
print(f"Reflectance values saved to {reflectance_file}: {reflectance_rows} rows, {reflectance_cols} columns (Reflectance at different wavelengths)")
print(f"Rows with NaN values that were skipped: {deleted_rows}")
print(f"Rows skipped due to out-of-bound coordinates: {skipped_by_coord}")

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
