import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from scipy import integrate
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, KernelDensity


def import_dataset(filename):
    # read data
    train = pd.read_csv(filename)
    return train


def sliding_window_rms(df, window_size):
    """
    Calculate the sliding window root-mean-square (RMS) for a multidimensional pandas DataFrame.

    Parameters:
    df (pandas.DataFrame): Multidimensional DataFrame containing the data.
    window_size (int): The size of the sliding window.

    Returns:
    pandas.DataFrame: A DataFrame containing the RMS values for each sliding window.
    """
    rms_df = pd.DataFrame()

    # Apply sliding window calculation for each column in the DataFrame
    for column in df.columns:
        # Calculate the squared values
        squared_values = df[column] ** 2

        # Create a rolling window and calculate the mean of the squared values
        rolling_mean_squared = squared_values.rolling(window=window_size, min_periods=1).mean()

        # Calculate the RMS by taking the square root of the mean squared values
        rms = np.sqrt(rolling_mean_squared)

        # Store the RMS values in the result DataFrame
        rms_df[column] = rms

    return rms_df


def FindThreshold(x,h,p):
    tau=0
    x.sort()
    for i in range(len(x)):
        int_K = integrate.quad(lambda s: (1/(h*np.sqrt(2*np.pi)))*np.exp(-0.5*(s-p)/h), (i-1)/len(x), i/len(x))
        tau=tau+int_K[0]*x[i]
    return tau

def kqe(train_df, anomaly_df):
    # Filter the columns to create datasets of expected and predicted values
    train_true = train_df.filter(regex='^expected').values
    train_pred = train_df.filter(regex='^predicted').values

    anomaly_true = anomaly_df.filter(regex='^expected').values
    anomaly_pred = anomaly_df.filter(regex='^predicted').values

    mse = np.mean(np.power((anomaly_true - anomaly_pred), 2), axis=1)
    mse_train = np.mean(np.power((train_true - train_pred), 2), axis=1)
    params = {'bandwidth': np.linspace(0, 0.5, 10)}
    grid = GridSearchCV(KernelDensity(), params, cv = 20)
    # mse = sliding_window_rms(pd.DataFrame(mse), 40).values.flatten()
    print(mse.shape)
    mse_train = sliding_window_rms(pd.DataFrame(mse_train), 40).values.flatten()
    print(mse_train.shape)
    grid.fit(mse_train[:, None])

    print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))
    h = grid.best_estimator_.bandwidth
    tau = FindThreshold(mse_train, h, 0.42)

    y_test1 = np.loadtxt("datasets/smap-msl/msl/p11_anomaly_y.csv", delimiter=",", skiprows=1)
    # y_test1 = y_test1[3:]
    y_test1[y_test1 == 1] = -1
    y_test1[y_test1 == 0] = 1
    # y_test1=np.ones(X_test.shape[0])
    # y_test1[999:1499]=-1
    y_scores=np.ones(y_test1.shape[0])
    y_scores[(mse-tau)>0]=-1
    precision = precision_score(y_test1, y_scores)
    recall    = recall_score(y_test1, y_scores)
    accuracy = accuracy_score(y_test1, y_scores)
    f1 = f1_score(y_test1, y_scores)
    print ('Tau : ', tau)
    print ('Precision : ', precision)
    print ('Recall: ', recall)
    print ('Accuracy : ', accuracy)
    print ('F1_score: ', f1)


def ocsvm(train_df, anomaly_df):
    # Filter the columns to create datasets of expected and predicted values
    print(train_df.shape, anomaly_df.shape)
    train_true = train_df.filter(regex='^expected').values
    train_pred = train_df.filter(regex='^predicted').values
    print(train_true.shape, train_pred.shape)
    # Calculate deviations for each feature
    # TODO switch to MSE
    train_residuals = np.abs(train_true - train_pred)
    train_residuals = pd.DataFrame(train_residuals)

    anomaly_true = anomaly_df.filter(regex='^expected').values
    anomaly_pred = anomaly_df.filter(regex='^predicted').values
    # Calculate deviations for each feature
    anomaly_residuals = np.abs(anomaly_true - anomaly_pred)
    anomaly_residuals = pd.DataFrame(anomaly_residuals)

    # train_residuals = train_residuals.ewm(span=50).mean()
    train_residuals = sliding_window_rms(train_residuals, 50)
    print("train_residuals shape: ", train_residuals.shape)
    # anomaly_residuals = anomaly_residuals.ewm(span=50).mean()
    anomaly_residuals = sliding_window_rms(anomaly_residuals, 50)
    # Train the OneClassSVM
    ocsvm = OneClassSVM(nu=0.045, gamma=0.004)
    ocsvm.fit(train_residuals)
    # -1 for anomalies, 1 for normal points
    predictions = ocsvm.predict(anomaly_residuals)

    np.savetxt("ocsvm.csv", predictions, fmt='%i', delimiter=",")

    true_labels = pd.read_csv("/Users/aryanjha/Documents/exact/datasets/cats/anomaly_y.csv")
    true_labels = true_labels.replace({0: 1, 1: -1})
    true_labels = true_labels.values.flatten()
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)

    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print()

    # Plot results
    plt.figure()
    plt.title("train_residuals")
    plt.plot(train_residuals, 'r')
    plt.figure()
    plt.title("anomaly_residuals")
    plt.plot(anomaly_residuals, 'b')

    plt.show()


def main():
    train_df = import_dataset("test_genomes/train_predictions.csv")
    anomaly_df = import_dataset("test_genomes/predictions.csv")
    ocsvm(train_df, anomaly_df)
    # kqe(train_df, anomaly_df)


if __name__ == '__main__':
    main()
