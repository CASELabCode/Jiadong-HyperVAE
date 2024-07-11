
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

class MDN(nn.Module):
    def __init__(self, input_dim, output_dim, num_gaussians=10):
        super(MDN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_gaussians = num_gaussians

        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        
        self.pi = nn.Linear(32, num_gaussians)
        self.sigma = nn.Linear(32, num_gaussians * output_dim)
        self.mu = nn.Linear(32, num_gaussians * output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        pi = F.softmax(self.pi(x), dim=-1)
        sigma = torch.exp(self.sigma(x)).view(-1, self.num_gaussians, self.output_dim)
        mu = self.mu(x).view(-1, self.num_gaussians, self.output_dim)
        
        return pi, sigma, mu

def mdn_loss(pi, sigma, mu, y):
    m = torch.distributions.Normal(loc=mu, scale=sigma)
    y = y.unsqueeze(1).expand_as(mu)
    log_probs = m.log_prob(y)
    pi = pi.unsqueeze(-1).expand_as(log_probs)
    weighted_log_probs = log_probs + torch.log(pi)
    log_sum = torch.logsumexp(weighted_log_probs, dim=1)
    return -log_sum.mean()

def train_mdn(model, train_dl, epochs=200):
    model.train()
    min_total_loss = float('inf')
    best_model_total_path = 'F:\\aphy-and-Chla-prediction\\Model\\mdn_model_best_total.pth'

    for epoch in range(epochs):
        total_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            pi, sigma, mu = model(x)
            loss = mdn_loss(pi, sigma, mu, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        avg_total_loss = total_loss / len(train_dl)
        print(f'epoch = {epoch + 1}, total_loss = {avg_total_loss:.4f}')

        if avg_total_loss < min_total_loss:
            min_total_loss = avg_total_loss
            torch.save(model.state_dict(), best_model_total_path)

    torch.save(model.state_dict(), 'F:\\Geo\\Model\\mdn_model.pth')

def evaluate_mdn(model, test_dl):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            pi, sigma, mu = model(x)
            m = torch.distributions.Normal(loc=mu, scale=sigma)
            y_samples = m.sample()  # Shape: [batch_size, num_gaussians, output_dim]
            pi_expanded = pi.unsqueeze(-1)  # Shape: [batch_size, num_gaussians, 1]
            weighted_samples = pi_expanded * y_samples  # Broadcasting to match shapes
            y_pred = torch.sum(weighted_samples, dim=1)  # Shape: [batch_size, output_dim]
            predictions.append(y_pred.cpu().numpy())
            actuals.append(y.cpu().numpy())
    return np.vstack(predictions), np.vstack(actuals)

def minmax_scale(data, min_val, max_val, feature_range=(0, 1)):
    scale = (feature_range[1] - feature_range[0]) / (max_val - min_val)
    min_range = feature_range[0]
    data_scaled = (data - min_val) * scale + min_range
    return data_scaled

def inverse_minmax_scale(data_scaled, min_val, max_val, feature_range=(0, 1)):
    scale = (max_val - min_val) / (feature_range[1] - feature_range[0])
    min_range = feature_range[0]
    data_original = (data_scaled - min_range) * scale + min_val
    return data_original

def load_real_data(aphy_file_path, rrs_file_path):

    array1 = np.loadtxt(aphy_file_path, delimiter=',', dtype=float)
    array2 = np.loadtxt(rrs_file_path, delimiter=',', dtype=float)
   
    Rrs_real = array2
    a_phy_real = array1

    input_dim = Rrs_real.shape[1]
    output_dim = a_phy_real.shape[1]

    min_val = 0
    max_val = 3

    scalers_Rrs_real = [MinMaxScaler(feature_range=(1, 10)) for _ in range(Rrs_real.shape[0])]

    Rrs_real_normalized = np.array([scalers_Rrs_real[i].fit_transform(row.reshape(-1, 1)).flatten() for i, row in enumerate(Rrs_real)])
    a_phy_real_normalized = np.array([minmax_scale(row, min_val, max_val, feature_range=(1, 10)) for row in a_phy_real])
    #a_phy_real_normalized = a_phy_real

    Rrs_real_tensor = torch.tensor(Rrs_real_normalized, dtype=torch.float32)
    a_phy_real_tensor = torch.tensor(a_phy_real_normalized, dtype=torch.float32)
    
    dataset_real = TensorDataset(Rrs_real_tensor, a_phy_real_tensor)

    train_size = int(0.5 * len(dataset_real))
    test_size = len(dataset_real) - train_size
    train_dataset_real, test_dataset_real = random_split(dataset_real, [train_size, test_size])

    train_real_dl = DataLoader(train_dataset_real, batch_size=train_size, shuffle=True, num_workers=12)
    test_real_dl = DataLoader(test_dataset_real, batch_size=test_size, shuffle=False, num_workers=12)

    return train_real_dl, test_real_dl, input_dim, output_dim, min_val, max_val

def load_real_test(aphy_file_path, rrs_file_path):

    array1 = np.loadtxt(aphy_file_path, delimiter=',', dtype=float)
    array2 = np.loadtxt(rrs_file_path, delimiter=',', dtype=float)
   
    Rrs_real = array2
    a_phy_real = array1

    input_dim = Rrs_real.shape[1]
    output_dim = a_phy_real.shape[1]

    min_val = 0
    max_val = 3

    scalers_Rrs_real = [MinMaxScaler(feature_range=(1, 10)) for _ in range(Rrs_real.shape[0])]

    Rrs_real_normalized = np.array([scalers_Rrs_real[i].fit_transform(row.reshape(-1, 1)).flatten() for i, row in enumerate(Rrs_real)])
    a_phy_real_normalized = np.array([minmax_scale(row, min_val, max_val, feature_range=(1, 10)) for row in a_phy_real])

    Rrs_real_tensor = torch.tensor(Rrs_real_normalized, dtype=torch.float32)
    
    a_phy_real_tensor = torch.tensor(a_phy_real_normalized, dtype=torch.float32)
    
    dataset_real = TensorDataset(Rrs_real_tensor, a_phy_real_tensor)
    dataset_size = int(len(dataset_real))
    test_real_dl = DataLoader(dataset_real, batch_size=dataset_size, shuffle=False, num_workers=12)

    return test_real_dl, input_dim, output_dim, min_val, max_val


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

def calculate_metrics(predictions, actuals, threshold=10):
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

    # Apply the threshold to filter out predictions with large relative error
    mask = np.abs(predictions - actuals) / np.abs(actuals + 1e-10) < threshold
    filtered_predictions = predictions[mask]
    filtered_actuals = actuals[mask]

    # Remove NaN values after filtering
    valid_indices = ~np.isnan(filtered_predictions) & ~np.isnan(filtered_actuals)
    filtered_predictions = filtered_predictions[valid_indices]
    filtered_actuals = filtered_actuals[valid_indices]

    # Calculate epsilon and beta
    log_ratios = np.log10(filtered_predictions / (filtered_actuals + 1e-10))
    Y = np.median(np.abs(log_ratios))
    Z = np.median(log_ratios)
    epsilon = 100 * (10**Y - 1)
    beta = 100 * np.sign(Z) * (10**np.abs(Z) - 1)
    
    # Calculate additional metrics
    rmse = np.sqrt(np.mean((filtered_predictions - filtered_actuals) ** 2))
    rmsle = np.sqrt(np.mean((np.log10(filtered_predictions + 1) - np.log10(filtered_actuals + 1)) ** 2))
    mape = 100 * np.median(np.abs((filtered_predictions - filtered_actuals) / filtered_actuals))
    bias = 10 ** (np.mean(np.log10(filtered_predictions + 1e-10) - np.log10(filtered_actuals + 1e-10)))
    mae = 10 **(np.mean(np.abs(np.log10(filtered_predictions + 1e-10) - np.log10(filtered_actuals + 1e-10))))
    
    return epsilon, beta, rmse, rmsle, mape, bias, mae



def plot_results(predictions_rescaled, actuals_rescaled, save_dir, threshold=20, mode='test'):

    num_columns = actuals_rescaled.shape[1]

    epsilon_list, beta_list, rmse_list, rmsle_list, mape_list, bias_list, mae_list, slope_list  = [], [], [], [], [], [], [], []

    for n in range(num_columns):
        actuals = actuals_rescaled[:, n]
        predictions = predictions_rescaled[:, n]

        mask = np.abs(predictions - actuals) / np.abs(actuals+1e-10) < threshold
        filtered_predictions = predictions[mask]
        filtered_actuals = actuals[mask]

        log_actual = np.log10(np.where(actuals == 0, 1e-10, actuals))
        log_prediction = np.log10(np.where(predictions == 0, 1e-10, predictions))

        filtered_log_actual = np.log10(np.where(filtered_actuals == 0, 1e-10, filtered_actuals))
        filtered_log_prediction = np.log10(np.where(filtered_predictions == 0, 1e-10, filtered_predictions))

        epsilon, beta, rmse, rmsle, mape, bias, mae = calculate_metrics(filtered_predictions, filtered_actuals,threshold)

        epsilon_list.append(epsilon)
        beta_list.append(beta)
        rmse_list.append(rmse)
        rmsle_list.append(rmsle)
        mape_list.append(mape)
        bias_list.append(bias)
        mae_list.append(mae)

        
        valid_mask = np.isfinite(filtered_log_actual) & np.isfinite(filtered_log_prediction)
        slope, intercept = np.polyfit(filtered_log_actual[valid_mask], filtered_log_prediction[valid_mask], 1)
        slope_list.append(slope)
        x = np.array([-4, 2])
        y = slope * x + intercept

        plt.figure(figsize=(6, 6))

        plt.plot(x, y, linestyle='--', color='blue', linewidth=0.8)
        lims = [-4, 2]
        plt.plot(lims, lims, linestyle='-',color='black', linewidth=0.8)

        sns.scatterplot(x=log_actual, y=log_prediction, alpha=0.5)
        sns.kdeplot(x=filtered_log_actual, y=filtered_log_prediction, levels=3, color="black", fill=False, linewidths=0.8)
    

        plt.xlabel('Actual $a_{ph}$ Values', fontsize=24, fontname='Times New Roman')
        plt.ylabel('Predicted $a_{ph}$ Values', fontsize=24, fontname='Times New Roman')
        plt.xlim(-4, 2)
        plt.ylim(-4, 2)
        plt.grid(True, which="both", ls="--")

        plt.legend(title=(f'MAE = {mae:.2f}, RMSE = {rmse:.2f}, RMSLE = {rmsle:.2f} \n'
                        f'Bias = {bias:.2f}, Slope = {slope:.2f} \n'
                        f'MAPE = {mape:.2f}%, ε = {epsilon:.2f}%, β = {beta:.2f}%'),
                fontsize=16, title_fontsize=12, prop={'family': 'Times New Roman'})

        plt.xticks(fontsize=20, fontname='Times New Roman')
        plt.yticks(fontsize=20, fontname='Times New Roman')

        plt.savefig(os.path.join(save_dir, f'{mode}_plot_column_{n}.pdf'), bbox_inches='tight')
        plt.close()

    metrics = ['Epsilon [%]', 'Beta [%]', 'RMSE', 'RMSLE', 'MAPE [%]', 'Bias', 'MAE', 'Slope']

    metrics_values = [epsilon_list, beta_list, rmse_list, rmsle_list, mape_list, bias_list, mae_list, slope_list]

    wavelengths = np.linspace(400, 700, num_columns) 

    y_ranges = {
        'Epsilon [%]': (0, 50),
        'Beta [%]': (-4, 4),
        'RMSE': (0, 2),
        'RMSLE': (0, 0.2),
        'MAPE [%]': (0, 50),
        'Bias': (0.5, 1.5),
        'MAE': (0, 5),
        'Slope': (0.5, 1.5)
    }

    for metric, values in zip(metrics, metrics_values):

        plt.figure(figsize=(12, 6))

        colors = [wavelength_to_rgb(wl, brightness=0.8) for wl in wavelengths]


        if metric in ['Bias', 'Slope']:
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

        plt.grid(True, which='both', linestyle='--', linewidth=0.8, zorder=0)
        plt.gca().set_axisbelow(True)  
   
        plt.savefig(os.path.join(save_dir, f'{mode}_{metric}_bar_chart.pdf'), bbox_inches='tight')
        plt.close()




def calculate_l1_loss(predictions, actuals):
    l1_losses = np.abs(predictions - actuals).sum(axis=1)
    return l1_losses

def plot_line_comparison_top10(predictions_rescaled, actuals_rescaled, save_dir, mode='test'):
    #l1_losses = calculate_l1_loss(predictions_rescaled, actuals_rescaled)
    #top10_indices = np.argsort(l1_losses)[:10]

    top50_indices = np.arange(10)

    num_columns = actuals_rescaled.shape[1]
    wavelengths = np.linspace(400, 700, num_columns) 

    for i in top50_indices:
        plt.figure(figsize=(12, 6))
        plt.plot(predictions_rescaled[i], label='Predicted Values', linestyle='-', marker='o')
        plt.plot(actuals_rescaled[i], label='Actual Values', linestyle='-', marker='x')
        plt.xlabel('Wavelengths',fontsize=28, fontname='Times New Roman')
        plt.ylabel('$a_{ph}$ Values',fontsize=28, fontname='Times New Roman')
        tick_labels = np.arange(400, 701, 100)
        tick_positions = [np.argmin(np.abs(wavelengths - label)) for label in tick_labels]
        plt.xticks(tick_positions, tick_labels, fontsize=12, fontname='Times New Roman')
        plt.yticks(fontsize=20, fontname='Times New Roman')
        plt.legend(prop={'size': 24, 'family': 'Times New Roman'})
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f'{mode}_line_plot_comparison_top10_{i}.pdf'), bbox_inches='tight')
        plt.close()



def save_to_csv(data, file_path):
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)

def inverse_transform_by_row(scalers, data):
    return np.array([scalers[i].inverse_transform(row.reshape(-1, 1)).flatten() for i, row in enumerate(data)])

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_real_dl, test_real_dl, input_dim, output_dim, min_val, max_val  = load_real_data('F:\\Geo\\Data\\Real\\aphy_RA_HICO.csv','F:\\Geo\\Data\\Real\\Rrs_RA_HICO.csv')
    test_real_Sep, _, _,_,_  = load_real_test('F:\\Geo\\Data\\Real\\aphy_RA_HICO_Sep.csv','F:\\Geo\\Data\\Real\\Rrs_RA_HICO_Sep.csv')
    test_real_Oct, _, _,_,_  = load_real_test('F:\\Geo\\Data\\Real\\aphy_RA_HICO_Oct.csv','F:\\Geo\\Data\\Real\\Rrs_RA_HICO_Oct.csv')

    save_dir = "F:\\aphy-and-Chla-prediction\\plots\\MDN_aph_HICO"
    os.makedirs(save_dir, exist_ok=True)

    model = MDN(input_dim, output_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    train_mdn(model, train_real_dl, epochs=300)

    model.load_state_dict(torch.load('F:\\aphy-and-Chla-prediction\\Model\\mdn_model_best_total.pth', map_location=device))

    predictions, actuals = evaluate_mdn(model, test_real_dl)
    predictions_original = np.array([inverse_minmax_scale(pred, min_val, max_val, feature_range=(1, 10)) for pred in predictions])
    actuals_original = np.array([inverse_minmax_scale(act, min_val, max_val, feature_range=(1, 10)) for act in actuals])

    predictions_Sep, actuals_Sep = evaluate_mdn(model, test_real_Sep)
    predictions_original_Sep = np.array([inverse_minmax_scale(pred, min_val, max_val, feature_range=(1, 10)) for pred in predictions])
    actuals_original_Sep = np.array([inverse_minmax_scale(act, min_val, max_val, feature_range=(1, 10)) for act in actuals])

    predictions_Oct, actuals_Oct = evaluate_mdn(model, test_real_Oct)
    predictions_original_Oct = np.array([inverse_minmax_scale(pred, min_val, max_val, feature_range=(1, 10)) for pred in predictions])
    actuals_original_Oct = np.array([inverse_minmax_scale(act, min_val, max_val, feature_range=(1, 10)) for act in actuals])

    save_to_csv(predictions_original, os.path.join(save_dir, 'predictions_rescaled.csv'))
    save_to_csv(actuals_original, os.path.join(save_dir, 'actuals_rescaled.csv'))

    plot_line_comparison_top10(predictions_original, actuals_original, save_dir, mode='test')
    plot_results(predictions_original, actuals_original, save_dir, mode='test')

    plot_line_comparison_top10(predictions_original_Sep, actuals_original_Sep, save_dir, mode='Sep')
    plot_results(predictions_original_Sep, actuals_original_Sep, save_dir, mode='Sep')

    plot_line_comparison_top10(predictions_original_Oct, actuals_original_Oct, save_dir, mode='Oct')
    plot_results(predictions_original_Oct, actuals_original_Oct, save_dir, mode='Oct')

