import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the predicted Chl-a values and coordinates
predicted_chla_path = 'F:\\aphy-chla-predictions\\predicted_chla_PACE.csv'
coordinates_path = 'F:\\aphy-chla-predictions\\image\\coordinates_PACE.csv'

# Read the predicted Chl-a values and coordinates
predicted_chla = pd.read_csv(predicted_chla_path, header=None)
coordinates = pd.read_csv(coordinates_path, header=None, names=['Latitude', 'Longitude'])

# Flatten the predicted values if necessary
predicted_chla = predicted_chla.values.flatten()

# Extract latitude and longitude from coordinates
latitudes = coordinates['Latitude'].values
longitudes = coordinates['Longitude'].values

# Calculate percentiles to remove extreme values
vmin = np.percentile(predicted_chla, 0)  # 1st percentile
vmax = np.percentile(predicted_chla, 98)  # 99th percentile

# Plot the data on a scatter plot with jet colormap
plt.figure(figsize=(10, 6))
scatter = plt.scatter(longitudes, latitudes, c=predicted_chla, cmap='jet', marker='o', s=1, vmin=0, vmax=vmax)

# Add colorbar
cbar = plt.colorbar(scatter)
cbar.set_label('Predicted Chl-a Concentration')

# Add labels and title
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Predicted Chlorophyll-a Concentration with Coordinates')

# Show the plot
plt.grid(True)
plt.show()
