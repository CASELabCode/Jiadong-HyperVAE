import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split  
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy.io
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch.optim as optim
from scipy.signal import medfilt

def smooth_output_gaussian(recon_x, kernel_size=3, sigma=1.0):
    # 创建高斯核
    import numpy as np
    x = np.arange(kernel_size) - kernel_size // 2
    gaussian_weights = np.exp(-x**2 / (2 * sigma**2))
    gaussian_weights /= gaussian_weights.sum()  # 归一化
    gaussian_weights = torch.tensor(gaussian_weights, dtype=torch.float32)
    
    padding = kernel_size // 2
    recon_x = recon_x.unsqueeze(1)  # Shape: (batch_size, channels, length)
    recon_x = F.pad(recon_x, (padding, padding), mode='replicate')
    
    # 转为卷积核格式
    kernel = gaussian_weights.view(1, 1, -1).to(recon_x.device)
    smoothed_output = F.conv1d(recon_x, kernel, stride=1)
    
    return smoothed_output.squeeze(1)

def minmax_scale(data, min_val, max_val, feature_range=(1, 10)):
    scale = (feature_range[1] - feature_range[0]) / (max_val - min_val)
    min_range = feature_range[0]
    data_scaled = (data - min_val) * scale + min_range
    return data_scaled

def inverse_minmax_scale(data_scaled, min_val, max_val, feature_range=(1, 10)):
    scale = (max_val - min_val) / (feature_range[1] - feature_range[0])
    min_range = feature_range[0]
    data_original = (data_scaled - min_range) * scale + min_val
    return data_original


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

def calculate_metrics(predictions, actuals):
    """
    Calculate epsilon, beta and additional metrics (RMSE, RMSLE, MAPE, Bias, MAE).
    
    :param predictions: array-like, predicted values
    :param actuals: array-like, actual values
    :param threshold: float, relative error threshold
    :return: epsilon, beta, rmse, rmsle, mape, bias, mae
    """
    # Remove negative and invalid values
    predictions = np.where(predictions < 0, np.nan, predictions)
    actuals = np.where(actuals < 0, np.nan, actuals)

    # Remove NaN values after filtering
    valid_indices = ~np.isnan(predictions) & ~np.isnan(actuals)
    predictions = predictions[valid_indices]
    actuals = actuals[valid_indices]

    # Calculate epsilon and beta
    log_ratios = np.log(predictions / (actuals))
    Y = np.median(np.abs(log_ratios))
    Z = np.median(log_ratios)
    epsilon = 100 * (np.exp(Y) - 1)
    beta = 100 * np.sign(Z) * (np.exp(np.abs(Z)) - 1)
    
    # Calculate additional metrics
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    rmsle = np.sqrt(np.mean((np.log(predictions + 1) - np.log(actuals + 1)) ** 2))
    mape = 100 * np.median(np.abs((predictions - actuals) / actuals))
    bias = 10 ** (np.mean(np.log10(predictions + 1e-10) - np.log10(actuals + 1e-10)))
    mae = 10 **(np.mean(np.abs(np.log10(predictions + 1e-10) - np.log10(actuals + 1e-10))))
    
    return epsilon, beta, rmse, rmsle, mape, bias, mae


def plot_results(predictions_rescaled, actuals_rescaled, save_dir, mode='test'):

    num_columns = actuals_rescaled.shape[1]

    epsilon_list, beta_list, rmse_list, rmsle_list, mape_list, bias_list, mae_list, slope_list  = [], [], [], [], [], [], [], []

    for n in range(num_columns):

        actuals = actuals_rescaled[:, n]
        predictions = predictions_rescaled[:, n]

        log_actual = np.log10(np.where(actuals == 0, 1e-10, actuals))
        log_prediction = np.log10(np.where(predictions == 0, 1e-10, predictions))
        
        log_actual = np.log10(np.where(actuals == 0, 1e-10, actuals))
        log_prediction = np.log10(np.where(predictions == 0, 1e-10, predictions))

        epsilon, beta, rmse, rmsle, mape, bias, mae = calculate_metrics(predictions, actuals)

        epsilon_list.append(epsilon)
        beta_list.append(beta)
        rmse_list.append(rmse)
        rmsle_list.append(rmsle)
        mape_list.append(mape)
        bias_list.append(bias)
        mae_list.append(mae)

        plt.figure(figsize=(6, 6))

        lims = [-4, 2]
        plt.plot(lims, lims, linestyle='-',color='black', linewidth=0.8)

        sns.scatterplot(x=log_actual, y=log_prediction, alpha=0.5)

        if mode == 'test':
            valid_mask = np.isfinite(log_actual) & np.isfinite(log_prediction)
            slope, intercept = np.polyfit(log_actual[valid_mask], log_prediction[valid_mask], 1)
            slope_list.append(slope)
            x = np.array([-4, 2])
            y = slope * x + intercept
            plt.plot(x, y, linestyle='--', color='blue', linewidth=0.8)
            sns.kdeplot(x=log_actual, y=log_prediction, levels=3, color="black", fill=False, linewidths=0.8)   

        plt.xlabel('Actual $a_{phy}$ Values', fontsize=24, fontname='Times New Roman')
        plt.ylabel('Predicted $a_{phy}$ Values', fontsize=24, fontname='Times New Roman')
        plt.xlim(-4, 2)
        plt.ylim(-4, 2)
        plt.grid(True, which="both", ls="--")

        if mode == 'test':
            legend_title = (f'MALE = {mae:.2f}, RMSE = {rmse:.2f}, RMSLE = {rmsle:.2f} \n'
                            f'Log-Bias = {bias:.2f}, Slope = {slope:.2f} \n'
                            f'MAPE = {mape:.2f}%, ε = {epsilon:.2f}%, β = {beta:.2f}%')
        else:
            legend_title = (f'MALE = {mae:.2f}, RMSE = {rmse:.2f}, RMSLE = {rmsle:.2f} \n'
                            f'Log-Bias = {bias:.2f} \n'
                            f'MAPE = {mape:.2f}%, ε = {epsilon:.2f}%, β = {beta:.2f}%')

        plt.legend(title=legend_title,
                   fontsize=16, title_fontsize=12, prop={'family': 'Times New Roman'})

        plt.xticks(fontsize=20, fontname='Times New Roman')
        plt.yticks(fontsize=20, fontname='Times New Roman')

        plt.savefig(os.path.join(save_dir, f'{mode}_plot_column_{n}.pdf'), bbox_inches='tight')
        plt.close()

    metrics = ['ε [%]', 'β [%]', 'RMSE', 'RMSLE', 'MAPE [%]', 'Log-Bias', 'MALE', 'Slope']

    metrics_values = [epsilon_list, beta_list, rmse_list, rmsle_list, mape_list, bias_list, mae_list, slope_list]

    wavelengths = np.linspace(400, 700, num_columns) 

    y_ranges = {
        'ε [%]': (0, 100),
        'β [%]': (-100, 100),
        'RMSE': (0, 5),
        'RMSLE': (0, 1),
        'MAPE [%]': (0, 100),
        'Log-Bias': (0, 2),
        'MALE': (0, 5),
        'Slope': (0, 2)
    }

    for metric, values in zip(metrics, metrics_values):

        plt.figure(figsize=(12, 6))

        colors = [wavelength_to_rgb(wl, brightness=0.8) for wl in wavelengths]


        if metric in ['Log-Bias', 'Slope']:
            for i, (value, (r, g, b)) in enumerate(zip(values, colors)):
                plt.scatter(i, value, color=(r/255, g/255, b/255), s=50, zorder=3)
            plt.plot(range(len(values)), values, color='black', linestyle='-', linewidth=1, zorder=2)
        else:
            plt.bar(range(num_columns), values, color=[(r/255, g/255, b/255) for r, g, b in colors], alpha=1, width=1.0, edgecolor='none', zorder=3)


        plt.xlabel('Wavelength (nm)', fontsize=28, fontname='Times New Roman')
        plt.ylabel(metric, fontsize=28, fontname='Times New Roman')

        plt.ylim(y_ranges[metric])

        plt.grid(True, which="both", ls="--")

        tick_labels = np.arange(400, 701, 100)
        tick_positions = [np.argmin(np.abs(wavelengths - label)) for label in tick_labels]

        plt.xticks(tick_positions, tick_labels, fontsize=20, fontname='Times New Roman')
        plt.yticks(fontsize=20, fontname='Times New Roman')

        plt.grid(True, which='both', linestyle='--', linewidth=0.7, zorder=0)
        plt.gca().set_axisbelow(True)  
   
        plt.savefig(os.path.join(save_dir, f'{mode}_{metric}_bar_chart.pdf'), bbox_inches='tight')
        plt.close()


# def plot_metrics(predictions_rescaled, actuals_rescaled, save_dir, threshold=1.0, mode='test'):

#     num_columns = actuals_rescaled.shape[1]

#     epsilon_list, beta_list, rmse_list, rmsle_list, mape_list, bias_list, mae_list, slope_list  = [], [], [], [], [], [], [], []

#     for n in range(num_columns):
#         actuals = actuals_rescaled[:, n]
#         predictions = predictions_rescaled[:, n]

#         mask = np.abs(predictions - actuals) / np.abs(actuals+1e-10) < 1.0
#         predictions = predictions[mask]
#         actuals = actuals[mask]

#         epsilon, beta, rmse, rmsle, mape, bias, mae = calculate_metrics(predictions, actuals,threshold)

#         epsilon_list.append(epsilon)
#         beta_list.append(beta)
#         rmse_list.append(rmse)
#         rmsle_list.append(rmsle)
#         mape_list.append(mape)
#         bias_list.append(bias)
#         mae_list.append(mae)

#     metrics = ['Epsilon [%]', 'Beta [%]', 'RMSE', 'RMSLE', 'MAPE [%]', 'Bias', 'MAE', 'Slope']

#     metrics_values = [epsilon_list, beta_list, rmse_list, rmsle_list, mape_list, bias_list, mae_list, slope_list]

#     wavelengths = np.linspace(400, 700, num_columns) 

#     y_ranges = {
#         'Epsilon [%]': (0, 50),
#         'Beta [%]': (-4, 4),
#         'RMSE': (0, 2),
#         'RMSLE': (0, 0.2),
#         'MAPE [%]': (0, 50),
#         'Bias': (0.5, 1.5),
#         'MAE': (0, 5),
#         'Slope': (0.5, 1.5)
#     }

#     for metric, values in zip(metrics, metrics_values):

#         plt.figure(figsize=(12, 6))

#         colors = [wavelength_to_rgb(wl, brightness=0.8) for wl in wavelengths]


#         if metric in ['Bias', 'Slope']:
#             for i, (value, (r, g, b)) in enumerate(zip(values, colors)):
#                 plt.scatter(i, value, color=(r/255, g/255, b/255), s=50, zorder=3)
#             plt.plot(range(len(values)), values, color='black', linestyle='-', linewidth=1, zorder=2)
#         else:
#             plt.bar(range(num_columns), values, color=[(r/255, g/255, b/255) for r, g, b in colors], alpha=1, width=1.0, edgecolor='none', zorder=3)


#         plt.xlabel('Wavelength (nm)', fontsize=28, fontname='Times New Roman')
#         plt.ylabel(metric, fontsize=28, fontname='Times New Roman')

#         plt.ylim(y_ranges[metric])

#         plt.grid(True, which="both", ls="--")

#         tick_labels = np.arange(400, 701, 100)
#         tick_positions = [np.argmin(np.abs(wavelengths - label)) for label in tick_labels]

#         plt.xticks(tick_positions, tick_labels, fontsize=20, fontname='Times New Roman')
#         plt.yticks(fontsize=20, fontname='Times New Roman')

#         plt.grid(True, which='both', linestyle='--', linewidth=0.7, zorder=0)
#         plt.gca().set_axisbelow(True)  
   
#         plt.savefig(os.path.join(save_dir, f'{mode}_{metric}_bar_chart.pdf'), bbox_inches='tight')
#         plt.close()

def plot_line_comparison_all(predictions_rescaled, actuals_rescaled, save_dir, mode='test'):

    num_columns = actuals_rescaled.shape[1]
    wavelengths = np.linspace(400, 700, num_columns) 

    for i in range(predictions_rescaled.shape[0]):
        plt.figure(figsize=(12, 6))
        plt.plot(predictions_rescaled[i], label='Predicted Values', linestyle='-', marker='o', color='blue')
        plt.plot(actuals_rescaled[i], label='Actual Values', linestyle='-', marker='x', color='#d62828')
        plt.xlabel('Wavelengths',fontsize=28, fontname='Times New Roman')
        plt.ylabel(mode,fontsize=28, fontname='Times New Roman')
        tick_labels = np.arange(400, 701, 100)
        tick_positions = [np.argmin(np.abs(wavelengths - label)) for label in tick_labels]
        plt.xticks(tick_positions, tick_labels, fontsize=20, fontname='Times New Roman')
        plt.yticks(fontsize=20, fontname='Times New Roman')
        plt.ylim(bottom=0)
        plt.legend(prop={'size': 24, 'family': 'Times New Roman'})
        plt.grid(True, which='both', linestyle='--', linewidth=0.6, zorder=0)
        plt.savefig(os.path.join(save_dir, f'{mode}_line_plot_comparison_{i}.pdf'), bbox_inches='tight')
        plt.close()

def save_to_csv(data, file_path):
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)

def inverse_transform_by_row(scalers, data):
    return np.array([scalers[i].inverse_transform(row.reshape(-1, 1)).flatten() for i, row in enumerate(data)])