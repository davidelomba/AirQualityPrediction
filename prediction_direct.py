import math
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from torch._dynamo.utils import rmse

from AQI import calculate_aqi, pm10_breakpoints, get_aqi_category
from TemporalGNN import TemporalGNN
from request import NUM_FEATURES, NUM_TIMESTEPS_OUT, OUT_CHANNELS, FEATURES_LIST, reverse_zscore, SENSOR, SENSORS_LIST


def prediction_direct(date_start, date_end, data_file, k, network_model, num_timesteps_in, num_timesteps_out):
    # Carica il file CSV
    df = pd.read_csv(data_file, delimiter=';')

    # Converte la colonna 'bucket_start_timestamp' in formato datetime
    df['bucket_start_timestamp'] = pd.to_datetime(df['bucket_start_timestamp'])

    # Filtra le righe in base all'intervallo di date
    df_day = df[(df['bucket_start_timestamp'] >= date_start) & (df['bucket_start_timestamp'] < date_end)]

    date_start = datetime.strptime(date_start, "%Y-%m-%d")
    date_start_true = date_start + timedelta(days=1)
    date_start_true = date_start_true.strftime("%Y-%m-%d")


    # Salvo il giorno successivo per confrontarlo con le predizioni
    df_true = df[(df['bucket_start_timestamp'] >= date_start_true) & (df['bucket_start_timestamp'] < date_end)]

    # Calcolo della media delle colonne specificate
    means = df[SENSORS_LIST].mean()

    # Calcolo della deviazione standard delle colonne specificate
    stds = df_day[SENSORS_LIST].std()

    df_normalized = (df_day[SENSORS_LIST] - means) / stds

    df_day_of_week = (df_day['day_of_week']) / 6

    df_normalized = pd.concat([df_normalized, df_day_of_week], axis=1)

    df_normalized['bucket_start_timestamp'] = df['bucket_start_timestamp']
    df_normalized['station_name'] = df['station_name']

    # Estrae la lista delle stazioni uniche e delle features
    stations = df_normalized['station_name'].unique()
    features = FEATURES_LIST
    num_stations = len(stations)

    # Inizializza una lista per contenere i tensori
    tensor_list = []

    # Itera attraverso gli intervalli di tempo
    start_time = df_normalized['bucket_start_timestamp'].min()
    end_time = df_normalized['bucket_start_timestamp'].max()
    current_time = start_time

    while current_time < end_time:
        # Seleziona i dati per l'intervallo di tempo corrente
        interval_data = df_normalized[(df_normalized['bucket_start_timestamp'] >= current_time) &
                                      (df_normalized['bucket_start_timestamp'] < current_time + pd.Timedelta(hours=num_timesteps_in))]

        # Inizializza un tensore per l'intervallo di tempo corrente
        interval_tensor = np.zeros((len(stations), len(features), num_timesteps_in))

        # Riempie il tensore con i valori delle features per ogni stazione
        for i, station in enumerate(stations):
            station_data = interval_data[interval_data['station_name'] == station]
            for j, feature in enumerate(features):
                interval_tensor[i, j, :] = station_data[feature].values[:num_timesteps_in]

        # Aggiunge il tensore alla lista
        tensor_list.append(interval_tensor)

        # Passa all'intervallo di tempo successivo
        current_time += pd.Timedelta(hours=num_timesteps_in)

    # Faccio la stessa cosa per il mese successivo
    tensor_list_true = []

    start_time = df_true['bucket_start_timestamp'].min()
    end_time = df_true['bucket_start_timestamp'].max()
    current_time = start_time

    while current_time < end_time:
        # Seleziona i dati per l'intervallo di tempo corrente
        interval_data_true = df_true[(df_true['bucket_start_timestamp'] >= current_time) &
                                     (df_true['bucket_start_timestamp'] < current_time + pd.Timedelta(hours=num_timesteps_in))]

        # Inizializza un tensore per l'intervallo di tempo corrente
        interval_tensor_true = np.zeros((len(stations), len(features), num_timesteps_in))

        # Riempie il tensore con i valori delle features per ogni stazione
        for i, station in enumerate(stations):
            station_data = interval_data_true[interval_data_true['station_name'] == station]
            for j, feature in enumerate(features):
                interval_tensor_true[i, j, :] = station_data[feature].values[:num_timesteps_in]

        # Aggiunge il tensore alla lista
        tensor_list_true.append(interval_tensor_true)

        # Passa all'intervallo di tempo successivo
        current_time += pd.Timedelta(hours=num_timesteps_in)

    matrices = np.load("matrici.npz")
    edge_index = matrices['edge_index']
    edge_weight = matrices['edge_weight']

    edge_index = torch.tensor(edge_index).to(torch.int64)
    edge_weight = torch.tensor(edge_weight).to(torch.float32)

    # Create model
    model = TemporalGNN(node_features=NUM_FEATURES, periods=NUM_TIMESTEPS_OUT, out_channels=OUT_CHANNELS)

    state_dict = torch.load(network_model)

    # Carica gli stati nel modello
    model.load_state_dict(state_dict)

    model.eval()

    predictions = []
    labels = tensor_list_true

    label_array = []

    for array in labels:
        # Estraiamo la terza colonna (quella del sensore considerato)
        label_sensor = array[:, SENSOR, :]

        label_array.append(label_sensor)

    final_label = np.concatenate(label_array, axis=1)

    pred_line = np.zeros((num_stations, 744))

    # Il primo valore è uguale all'osservazione per ogni stazione
    for i in range(0, num_stations):
        pred_line[i, 0] = final_label[i, 0]

    for i in range(0, num_stations):
        for j in range(1, 744):
            pred_line[i, j] = k * final_label[i, j - 1] + (1 - k) * pred_line[i, j - 1]

    list_of_tensors = [torch.tensor(arr).to(torch.float32) for arr in tensor_list]

    timestamps = num_timesteps_in

    for i, tensor in enumerate(list_of_tensors):
        mod_tensor = tensor
        next_tensor_index = (i + 1) % len(list_of_tensors)
        for j in range(0, timestamps, num_timesteps_out):
            y_hat = model(mod_tensor, edge_index, edge_weight)
            yhat_reverse = reverse_zscore(y_hat, means[SENSOR], stds[SENSOR])
            predictions.append(yhat_reverse)  # Aggiungi il valore predetto a predictions

            mod_tensor = np.roll(mod_tensor, -num_timesteps_out, axis=2)  # Sposta a sinistra lungo l'asse del timestamp
            next_tensor = list_of_tensors[next_tensor_index]

            if i < len(list_of_tensors) - 1:
                add_value = next_tensor[:, :, j:j + num_timesteps_out]
            else:
                break

            mod_tensor[:, :, -num_timesteps_out:] = add_value
            mod_tensor = torch.tensor(mod_tensor)

    pred_list = [tensor.detach().numpy() for tensor in predictions]

    pred_list = pred_list[:-1]

    final_pred = np.concatenate(pred_list, axis=1)

    rmse_values = np.zeros(final_pred.shape[0])
    rmse_line = np.zeros(final_pred.shape[0])

    rmse_values_day = np.zeros((num_stations, 31))
    rmse_line_day = np.zeros((num_stations, 31))
    pred_line_avg = np.zeros((num_stations, 31))

    for i in range(0, num_stations):
        count = 0
        err_day = 0
        pred_avg = 0
        for j in range(0, 744):
            count = count+1
            if count == 24:
                rmse_line_day[i, int(j / 24)] = math.sqrt(err_day / 24)
                pred_line_avg[i, int(j / 24)] = pred_avg / 24
                err_day = 0
                count = 0
                pred_avg = 0
            else:
                err_day = err_day + (final_label[i, j] - pred_line[i, j]) ** 2
                pred_avg = pred_avg + pred_line[i, j]

    final_label_avg = np.zeros((final_label.shape[0], int(final_label.shape[1] / 24)))
    final_pred_avg = np.zeros((final_pred.shape[0], int(final_pred.shape[1] / 24)))

    rmse_avg = 0
    baseline_avg = 0
    aqi_avg = 0

    for i in range(final_pred.shape[0]):
        labels_aqi = []
        pred_aqi = []
        accuracy = 0
        eqm = 0
        count = 0
        label_avg = 0
        pred_avg = 0
        days = 0
        err_day = 0
        for j in range(final_pred.shape[1]):
            count += 1
            if count == 24:
                label_avg = label_avg / 24
                pred_avg = pred_avg / 24

                final_label_avg[i, int(j / 24)] = label_avg
                final_pred_avg[i, int(j / 24)] = pred_avg

                rmse_values_day[i, int(j / 24)] = math.sqrt(err_day/24)
                # category_l, color_l, ind_l = aqi_pm10(label_avg)
                aqi_pm10_l = calculate_aqi(label_avg, pm10_breakpoints)
                category_l, color_l, ind_l = get_aqi_category(aqi_pm10_l)
                labels_aqi.append(category_l)

                # category_p, color_p, ind_p = aqi_pm10(pred_avg)
                aqi_pm10_p = calculate_aqi(pred_avg, pm10_breakpoints)
                category_p, color_p, ind_p = get_aqi_category(aqi_pm10_p)
                pred_aqi.append(category_p)

                if category_p == category_l:
                    accuracy += 1

                eqm = eqm + (ind_p - ind_l) ** 2

                count = 0
                label_avg = 0
                pred_avg = 0
                days += 1
                err_day = 0
            else:
                label_avg += final_label[i, j]
                pred_avg += final_pred[i, j]
                err_day = err_day + (final_label[i, j] - final_pred[i, j])**2
            # aqi_pm10_l = calculate_aqi(final_label[i, j], pm10_breakpoints)
            # category_l, color_l, ind_l = get_aqi_category(aqi_pm10_l)
            # category_l, color_l, ind_l = aqi_pm10(final_label[i, j])
            # labels_aqi.append(category_l)

            # aqi_pm10_p = calculate_aqi(final_pred[i, j], pm10_breakpoints)
            # category_p, color_p, ind_p = get_aqi_category(aqi_pm10_p)
            # category_p, color_p, ind_p = aqi_pm10(final_pred[i, j])
            # pred_aqi.append(category_p)

        print("Sono giuste " + str(accuracy) + " previsioni su " + str(days) + " con un errore di " + str(
            math.sqrt(eqm) / 31))


        #plt.figure()
        #plt.plot(final_label[i], label='true')
        #plt.plot(final_pred[i], label='pred')
        #plt.legend()
        #plt.grid(True)
        #plt.show()
        rmse_values[i] = np.sqrt(mean_squared_error(final_pred[i], final_label[i]))
        rmse_line[i] = np.sqrt(mean_squared_error(pred_line[i], final_label[i]))
        print("RMSE Predizione {}: {:.6f}".format(i, rmse_values[i]))
        print("RMSE Baseline {}: {:.6f}".format(i, rmse_line[i]))

        aqi_avg += accuracy
        baseline_avg += rmse_line[i]
        rmse_avg += rmse_values[i]


        plt.figure()
        plt.plot(final_label_avg[i], label='Valori reali')
        plt.plot(final_pred_avg[i], label='Valori predetti')
        plt.xlabel('Giorno')
        plt.ylabel('Concentrazione inquinante')
        #plt.ylim(0, 90)
        #plt.plot(pred_line_avg[i], label='pred baseline')
        plt.title("Concentrazione PM10 in un mese")
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure()
        plt.plot(rmse_line_day[i], label='Baseline')
        plt.plot(rmse_values_day[i], label='Predizioni')
        plt.xlabel('Giorno')
        plt.ylabel('Valore RMSE')
        #plt.ylim(0, 50)
        plt.title("RMSE del PM10 in un mese")
        plt.legend()
        plt.grid(True)
        plt.show()


    aqi_avg = (aqi_avg/(31*num_stations))*100
    rmse_avg = rmse_avg/num_stations
    baseline_avg = baseline_avg/num_stations
    print("\n\n\n")
    print("Percentuale AQI corretti: ", aqi_avg)
    print("RMSE su tutte le stazioni: ", rmse_avg)
    print("RMSE baseline su tutte le stazioni: ", baseline_avg)
if __name__ == "__main__":
    prediction_direct('2024-01-25', '2024-02-26', 'complete_dataframe.csv', 0.05, 'model/model_24_4_1_256_100_2_100.pth', 24, 4)
