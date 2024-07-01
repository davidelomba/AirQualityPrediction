import os
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
from timeit import default_timer as timer

import scipy

from StaticGraphTemporalSignal import StaticGraphTemporalSignal
from TemporalGNN import TemporalGNN
import seaborn as sns

matplotlib.use('Qt5Agg')
import numpy as np
import pandas as pd
import torch
import api_airqino
from geopy.distance import geodesic


### Variabili globali ###

NUM_TIMESTEPS_IN = 24
NUM_TIMESTEPS_OUT = 1
STEP_SLICE = 1
EPOCH = 100
OUT_CHANNELS = 256
SENSOR = 2  # 0 = 'co2', 1 = 'extT', 2 = 'pm10', 3 = 'pm25', 4 = 'rh', 5 = 'wind', 6 = 'tp'
DATE_START = '2024-01-15'
DATE_END = '2024-04-15'
THRESHOLD_DISTANCE = 4
NUM_STATIONS = 15
FEATURES_LIST = ['co2', 'extT', 'pm10', 'pm25', 'rh', 'wind', 'tp', 'day_of_week']     # 'co2', 'extT', 'pm10', 'pm25', 'rh', 'wind', 'tp', 'day_of_week'
#FEATURES_LIST = ['co2', 'extT', 'pm10', 'pm25', 'rh', 'day_of_week']
NUM_FEATURES = len(FEATURES_LIST)
SENSORS_LIST = ['co2', 'extT', 'pm10', 'pm25', 'rh', 'wind', 'tp']
#SENSORS_LIST = ['co2', 'extT', 'pm10', 'pm25', 'rh']
NUM_SENSORS = len(SENSORS_LIST)


def get_station_data(dt_from_string, dt_to_string):
    if os.path.exists('data/file.csv'):
        os.remove('data/file.csv')
    with open('station/tot', 'r', newline='') as file:
        for linea in file:
            station_name = linea.strip().split()[0]
            data = api_airqino.get_hourly_avg(station_name, dt_from_string, dt_to_string)
            with open('data/file.csv', 'a', newline='') as csvfile:
                csvfile.write(data)


def extract_rows(data, column):
    new_data = data[column]
    return new_data


def remove_rows(data, column, value):
    delete_index = data[data[column] == value].index
    new_data = data.drop(delete_index)
    return new_data


# Crea il dataframe eliminando dal data le colonne e le righe inutili
def create_dataframe():
    data = pd.read_csv('data/file.csv', sep=';')

    data = remove_rows(data, 'bucket_start_timestamp', 'bucket_start_timestamp')

    sensors_to_remove = ['AUX1', 'AUX2', 'voc', 'co', 'o3', 'no2', 'intT']
    for sensor in sensors_to_remove:
        data = remove_rows(data, 'sensor', sensor)

    columns_to_drop = ['k1', 'k2', 'k3', 'k4', 'r', 'session_name', 'raw_value']
    data = data.drop(columns_to_drop, axis=1)

    print(data)
    print(data.columns)
    return data


def lon_lat_dataframe(df):
    df_lon_lat = extract_rows(df, ['station_name', 'latitude', 'longitude'])
    df_lon_lat = df_lon_lat.drop_duplicates(subset='station_name', keep='first')

    print(df_lon_lat)
    print(df_lon_lat.columns)
    return df_lon_lat


def extract_feature(data, column, feature):
    data_feature = data[data[column] == feature]
    print(data_feature)
    return data_feature


# Crea il dataframe con i sensori per colonne e ordina le righe per ora di rilevazione
def create_dataframe_graph(df):
    df_pivot = df.pivot_table(index=['station_name', 'bucket_start_timestamp'], columns='sensor',
                              values='calibrated_value', aggfunc='first')
    df_pivot = df_pivot.sort_values(by='bucket_start_timestamp')
    print(df_pivot)
    return df_pivot


# Calcola la distanza tra tutte le coppie di stazioni
def calculate_distance(df):
    n = len(df)
    distances = np.zeros((n, n))

    for i in range(n):
        coord_i = (df.iloc[i]['latitude'], df.iloc[i]['longitude'])
        for j in range(i + 1, n):
            coord_j = (df.iloc[j]['latitude'], df.iloc[j]['longitude'])
            dist = geodesic(coord_i, coord_j).kilometers
            distances[i][j] = dist
            distances[j][i] = dist

    return distances


def build_graph(df, threshold_distance, data_file, num_timesteps_in=NUM_TIMESTEPS_IN,
                num_timesteps_out=NUM_TIMESTEPS_OUT):  # Guardiamo i dati di tre giorni e prevediamo le 12 ore successive
    # Calcola le distanze tra tutte le coppie di stazioni
    distances = calculate_distance(df)

    # Calcola i pesi degli archi come inversi delle distanze (evita divisione per zero)
    weights = 1 / (distances + 0.01)

    # Applica la soglia di distanza per costruire la matrice di adiacenza
    adjacency_matrix = (distances < threshold_distance).astype(float)
    np.fill_diagonal(adjacency_matrix, 0.0)

    # Trova gli indici non zero nella matrice di adiacenza
    edge_index = np.transpose(np.transpose(np.nonzero(adjacency_matrix)))

    # Costruisci gli attributi degli archi come tensore PyTorch

    edge_weight = weights[adjacency_matrix > 0].astype(float)

    # Carica i dati delle misurazioni dai data CSV
    measurement_df = pd.read_csv(data_file, sep=';')

    # Estrai le informazioni necessarie per costruire la matrice delle features
    stations = measurement_df['station_name'].unique()
    timestamps = measurement_df['bucket_start_timestamp'].unique()
    num_stations = len(stations)
    num_timestamps = len(timestamps)
    num_sensors = NUM_FEATURES

    all_features = np.zeros((num_stations, num_sensors, num_timestamps))

    # Riempie la matrice delle features con i valori dei sensori
    for j, timestamp in enumerate(timestamps):
        for i, station in enumerate(stations):
            mask = (measurement_df['station_name'] == station) & (measurement_df['bucket_start_timestamp'] == timestamp)
            sensor_values = measurement_df[mask][FEATURES_LIST].values
            sensor_values_flat = sensor_values.flatten()

            # Assegna i valori a all_features
            all_features[i, :, j] = sensor_values_flat

    # Calcola la media e la deviazione standard

    # means = np.mean(all_features, axis=(0, 2))
    # data_norm = all_features - means.reshape(1, -1, 1)
    # stds = np.std(all_features, axis=(0, 2))
    # all_features = data_norm / stds.reshape(1, -1, 1)

    # Calcola la media e la deviazione standard solo per le prime features (escludendo day_of_week)
    means = np.mean(all_features[:, :NUM_FEATURES-1, :], axis=(0, 2))
    stds = np.std(all_features[:, :NUM_FEATURES-1, :], axis=(0, 2))

    # Normalizzazione usando media e deviazione standard per i primi 5 features
    data_norm = all_features[:, :NUM_FEATURES-1, :] - means.reshape(1, NUM_FEATURES-1, 1)
    all_features[:, :NUM_FEATURES-1, :] = data_norm / stds.reshape(1, NUM_FEATURES-1, 1)

    # Seleziona il giorno della settimana
    day_of_week = all_features[:, NUM_FEATURES-1, :]

    scaled_feature = day_of_week/6

    all_features[:, NUM_FEATURES-1, :] = scaled_feature


    np.savez("data/matrici.npz", mean=means, std=stds, edge_index=edge_index, edge_weight=edge_weight)

    # Mappa gli indici numerici ai nomi delle stazioni
    node_index_to_station_name = {i: station_name for i, station_name in enumerate(df['station_name'])}

    # Calcola gli indici delle finestre temporali
    indices = [
        (i, i + (num_timesteps_in + num_timesteps_out))
        for i in range(0, all_features.shape[2] - (num_timesteps_in + num_timesteps_out) + 1, STEP_SLICE)
    ]

    # Inizializza liste per le features e i target
    features, target = [], []

    # Genera le osservazioni per ciascuna finestra temporale indicata dagli indici
    for i, j in indices:
        # Estrai le features per la finestra temporale
        features_window = all_features[:, :, i:i + num_timesteps_in].copy()  # Copia la finestra temporale
        features.append(features_window)

        # Estrai il target per il timestep successivo
        target_timestep = all_features[:, :, i + num_timesteps_in:j].copy()  # Copia il target
        target.append(target_timestep)

    # Converte le liste di features e target in array numpy
    features = np.array(features)
    target = np.array(target)

    dataset = StaticGraphTemporalSignal(edge_index, edge_weight, features, target)
    print('Dimensione all_features: ', all_features.shape)
    return dataset, node_index_to_station_name

# the function to return data in original scale (reversing Z score)

def reverse_zscore(pandas_series, mean, std):
    '''Mean and standard deviation should be of original variable before standardisation'''
    yis=pandas_series*std+mean
    return yis


def add_data(csvfile):
    station = {}
    count = 0

    # Apri il data in modalità lettura
    with open('station/tot', 'r') as file:
        # Itera attraverso ogni riga nel data
        for line in file:
            # Dividi la riga in parole utilizzando lo spazio come delimitatore
            words = line.strip().split()

            if words:
                # Ottieni la prima parola della riga
                first_word = words[0]

                # Mappa la prima parola a un numero intero progressivo
                station[first_word] = count

                # Incrementa il contatore per il prossimo numero intero
                count += 1

    co2_matrix = np.zeros((len(station), 24))
    extT_matrix = np.zeros((len(station), 24))
    pm10_matrix = np.zeros((len(station), 24))
    pm25_matrix = np.zeros((len(station), 24))
    rh_matrix = np.zeros((len(station), 24))

    prev_day = None

    with open(csvfile, 'r') as csvfile:
        df = pd.read_csv(csvfile, sep=';')
        df_update = pd.DataFrame(columns=df.columns)

        # Itera attraverso le righe del data CSV
        for index, row in df.iterrows():
            # Estrae la stazione alla riga corrente
            station_row = row['station_name']

            # Estrae l'ora dall'orario timestamp
            timestamp = pd.to_datetime(row['bucket_start_timestamp'])
            hour = timestamp.hour
            current_day = timestamp.date()

            # Verifica se cambia il giorno
            if current_day != prev_day:
                if prev_day is not None:
                    # Crea una maschera per identificare i valori nulli
                    co2_mask = (co2_matrix != 0.0)
                    extT_mask = (extT_matrix != 0.0)
                    pm10_mask = (pm10_matrix != 0.0)
                    pm25_mask = (pm25_matrix != 0.0)
                    rh_mask = (rh_matrix != 0.0)
                    co2_matrix_masked = np.where(co2_mask, co2_matrix, np.nan)
                    extT_matrix_masked = np.where(extT_mask, extT_matrix, np.nan)
                    pm10_matrix_masked = np.where(pm10_mask, pm10_matrix, np.nan)
                    pm25_matrix_masked = np.where(pm25_mask, pm25_matrix, np.nan)
                    rh_matrix_masked = np.where(rh_mask, rh_matrix, np.nan)

                    # Calcola la media di ciascuna colonna della matrice, escludendo i valori nulli (zero)
                    co2_mean = np.nanmean(co2_matrix_masked, axis=0)
                    extT_mean = np.nanmean(extT_matrix_masked, axis=0)
                    pm10_mean = np.nanmean(pm10_matrix_masked, axis=0)
                    pm25_mean = np.nanmean(pm25_matrix_masked, axis=0)
                    rh_mean = np.nanmean(rh_matrix_masked, axis=0)

                    # Trova le posizioni nelle matrici dove i valori sono nulli (zero)
                    co2_pos_zero = np.where(co2_matrix == 0)
                    extT_pos_zero = np.where(extT_matrix == 0)
                    pm10_pos_zero = np.where(pm10_matrix == 0)
                    pm25_pos_zero = np.where(pm25_matrix == 0)
                    rh_pos_zero = np.where(rh_matrix == 0)

                    # Aggiunge la media delle colonne solo alle posizioni dove sono presenti valori nulli
                    co2_matrix[co2_pos_zero] += co2_mean[co2_pos_zero[1]]
                    extT_matrix[extT_pos_zero] += extT_mean[extT_pos_zero[1]]
                    pm10_matrix[pm10_pos_zero] += pm10_mean[pm10_pos_zero[1]]
                    pm25_matrix[pm25_pos_zero] += pm25_mean[pm25_pos_zero[1]]
                    rh_matrix[rh_pos_zero] += rh_mean[rh_pos_zero[1]]

                    rows_to_add = []

                    for station_name, station_index in station.items():
                        for ora in range(24):
                            timestamp = f"{prev_day} {ora:02d}:00:00"
                            co2_value = co2_matrix[station_index, ora]
                            extT_value = extT_matrix[station_index, ora]
                            pm10_value = pm10_matrix[station_index, ora]
                            pm25_value = pm25_matrix[station_index, ora]
                            rh_value = rh_matrix[station_index, ora]

                            # Creazione del dizionario rappresentante la riga da aggiungere
                            row_dict = {
                                'station_name': station_name,
                                'bucket_start_timestamp': timestamp,
                                'co2': co2_value,
                                'extT': extT_value,
                                'pm10': pm10_value,
                                'pm25': pm25_value,
                                'rh': rh_value,
                                'day_of_week': datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S").weekday()
                            }

                            # Aggiunta del dizionario alla lista delle righe da aggiungere
                            rows_to_add.append(row_dict)

                    # Creazione di un DataFrame dalle righe da aggiungere
                    new_rows_df = pd.DataFrame(rows_to_add)

                    # Concatenazione del DataFrame esistente con il nuovo DataFrame delle righe aggiunte
                    df_update = pd.concat([df_update, new_rows_df], ignore_index=True)

                    # Azzera le matrici
                    co2_matrix = np.zeros((len(station), 24))
                    extT_matrix = np.zeros((len(station), 24))
                    pm10_matrix = np.zeros((len(station), 24))
                    pm25_matrix = np.zeros((len(station), 24))
                    rh_matrix = np.zeros((len(station), 24))

            # Verifica se la stazione è presente nella mappa delle stazioni
            if station_row in station:
                # Inserisci il valore nella matrice alla posizione corrispondente
                co2_matrix[station[station_row], hour] = row['co2']
                pm10_matrix[station[station_row], hour] = row['pm10']
                pm25_matrix[station[station_row], hour] = row['pm25']
                rh_matrix[station[station_row], hour] = row['rh']
                extT_matrix[station[station_row], hour] = row['extT']

            prev_day = current_day

    # Quando termina il data devo comunque memorizzare l'ultimo giorno

    # Crea una maschera per identificare i valori nulli
    co2_mask = (co2_matrix != 0.0)
    extT_mask = (extT_matrix != 0.0)
    pm10_mask = (pm10_matrix != 0.0)
    pm25_mask = (pm25_matrix != 0.0)
    rh_mask = (rh_matrix != 0.0)
    co2_matrix_masked = np.where(co2_mask, co2_matrix, np.nan)
    extT_matrix_masked = np.where(extT_mask, extT_matrix, np.nan)
    pm10_matrix_masked = np.where(pm10_mask, pm10_matrix, np.nan)
    pm25_matrix_masked = np.where(pm25_mask, pm25_matrix, np.nan)
    rh_matrix_masked = np.where(rh_mask, rh_matrix, np.nan)

    # Calcola la media di ciascuna colonna della matrice, escludendo i valori nulli (zero)
    co2_mean = np.nanmean(co2_matrix_masked, axis=0)
    extT_mean = np.nanmean(extT_matrix_masked, axis=0)
    pm10_mean = np.nanmean(pm10_matrix_masked, axis=0)
    pm25_mean = np.nanmean(pm25_matrix_masked, axis=0)
    rh_mean = np.nanmean(rh_matrix_masked, axis=0)

    # Trova le posizioni nelle matrici dove i valori sono nulli (zero)
    co2_pos_zero = np.where(co2_matrix == 0)
    extT_pos_zero = np.where(extT_matrix == 0)
    pm10_pos_zero = np.where(pm10_matrix == 0)
    pm25_pos_zero = np.where(pm25_matrix == 0)
    rh_pos_zero = np.where(rh_matrix == 0)

    # Aggiunge la media delle colonne solo alle posizioni dove sono presenti valori nulli
    co2_matrix[co2_pos_zero] += co2_mean[co2_pos_zero[1]]
    extT_matrix[extT_pos_zero] += extT_mean[extT_pos_zero[1]]
    pm10_matrix[pm10_pos_zero] += pm10_mean[pm10_pos_zero[1]]
    pm25_matrix[pm25_pos_zero] += pm25_mean[pm25_pos_zero[1]]
    rh_matrix[rh_pos_zero] += rh_mean[rh_pos_zero[1]]

    rows_to_add = []

    for station_name, station_index in station.items():
        for ora in range(24):
            timestamp = f"{prev_day} {ora:02d}:00:00"
            co2_value = co2_matrix[station_index, ora]
            extT_value = extT_matrix[station_index, ora]
            pm10_value = pm10_matrix[station_index, ora]
            pm25_value = pm25_matrix[station_index, ora]
            rh_value = rh_matrix[station_index, ora]

            # Creazione del dizionario rappresentante la riga da aggiungere
            row_dict = {
                'station_name': station_name,
                'bucket_start_timestamp': timestamp,
                'co2': co2_value,
                'extT': extT_value,
                'pm10': pm10_value,
                'pm25': pm25_value,
                'rh': rh_value,
                'day_of_week': datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S").weekday()
            }

            # Aggiunta del dizionario alla lista delle righe da aggiungere
            rows_to_add.append(row_dict)

    # Creazione di un DataFrame dalle righe da aggiungere
    new_rows_df = pd.DataFrame(rows_to_add)

    # Concatenazione del DataFrame esistente con il nuovo DataFrame delle righe aggiunte
    df_update = pd.concat([df_update, new_rows_df], ignore_index=True)
    if os.path.exists('data/complete_dataframe.csv'):
        os.remove('data/complete_dataframe.csv')
    df_update.to_csv('data/complete_dataframe.csv', index=False, sep=';')


def complete_dataframe(file):
    df = pd.read_csv(file, sep=';')

    # Seleziona le ultime 6 colonne (features)
    df_features = df.iloc[:, -6:]

    # Calcola la media delle ultime 6 colonne ignorando i valori NaN
    means = df_features.mean(skipna=True)

    for col in df_features.columns:
        means_col = means[col]
        df[col].fillna(means_col, inplace=True)

    if os.path.exists(file):
        os.remove(file)
    df.to_csv(file, index=True, sep=';')

def training_testing(train_dataset, test_dataset):
    ###Training###

    # GPU support
    device = torch.device('cpu')  # cuda

    # Create model and optimizers
    model = TemporalGNN(node_features=NUM_FEATURES, periods=NUM_TIMESTEPS_OUT, out_channels=OUT_CHANNELS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()

    start_time = timer()
    print("Running training...")
    for epoch in range(EPOCH):
        step = 0
        loss = 0
        for snapshot in train_dataset:
            snapshot = snapshot.to(device)
            # Get model predictions
            y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
            # Mean squared error
            loss = loss + torch.mean((y_hat - snapshot.y[:, SENSOR, :]) ** 2)
            step +=1

        loss = loss / (step + 1)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print("Epoch {} train MSE: {:.4f}".format(epoch, loss.item()))

    ###Testing###

    model.eval()
    loss = 0
    step = 0

    # Store for analysis
    predictions = []
    labels = []

    ### Ritorno ai valori non normalizzati ###

    matrices = np.load("data/matrici.npz")

    # Estraiamo le matrici
    means = matrices['mean']
    stds = matrices['std']

    for snapshot in test_dataset:
        snapshot = snapshot.to(device)
        # Get predictions
        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        yhat_reverse = reverse_zscore(y_hat, means[SENSOR], stds[SENSOR])
        snapshot.y_reverse = reverse_zscore(snapshot.y[:, SENSOR, :], means[SENSOR], stds[SENSOR])
        # Mean squared error
        loss = loss + torch.sqrt(torch.mean((yhat_reverse-snapshot.y_reverse)**2))
        # Store for analysis below
        labels.append(snapshot.y_reverse)
        predictions.append(yhat_reverse)
        step += 1

    loss = loss / ((step + 1) * NUM_TIMESTEPS_OUT)
    loss = loss.item()
    print("Test RMSE: {:.4f}".format(loss))
    end_time = timer()
    time = end_time - start_time

    with open('data/log.txt', 'a') as file:
        # Scrivere una stringa nel data
        file.write('Tempo impiegato: ' + str(time) + " Test MSE: {:.4f}".format(loss) + '  NUM_TIMESTEPS_IN: ' + str(
            NUM_TIMESTEPS_IN) +
                   '    NUM_TIMESTEPS_OUT: ' + str(NUM_TIMESTEPS_OUT) + '  STEP_SLICE: ' + str(
            STEP_SLICE) + '  EPOCH: ' + str(EPOCH) + '   OUT_CHANNELS: ' + str(OUT_CHANNELS) + '  SENSOR: ' + str(
            SENSOR) + '   THRESHOLD_DISTANCE: ' + str(THRESHOLD_DISTANCE) + '\n')

    path = 'model/model_' + str(NUM_TIMESTEPS_IN) + '_' + str(NUM_TIMESTEPS_OUT) + '_' + str(STEP_SLICE) + '_' + str(
        OUT_CHANNELS) + '_' + str(EPOCH) + '_' + str(SENSOR) + '_' + str(THRESHOLD_DISTANCE) + '.pth'
    if os.path.exists(path):
        os.remove(path)
    torch.save(model.state_dict(), path)

    ### Visualization ###
    for i in range(NUM_STATIONS):
        preds = np.asarray(predictions[0][i].detach().cpu().numpy())
        labs = np.asarray(labels[0][i].cpu().numpy())

        plt.figure(figsize=(20, 5))
        sns.lineplot(data=preds, label="pred", markers='o')
        sns.lineplot(data=labs, label="true", markers='o')
        plt.show()
