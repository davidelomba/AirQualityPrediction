import os

import cdsapi
import numpy as np
import xarray as xr
import pandas as pd

from request import complete_dataframe


def request_cds():
    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': [
                '10m_u_component_of_wind', '10m_v_component_of_wind', 'total_precipitation',
            ],
            'year': '2024',
            'month': ['01', '02', '03', '04', ],
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',
            ],
            'time': [
                '00:00', '01:00', '02:00',
                '03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00',
                '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00',
                '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00',
                '21:00', '22:00', '23:00',
            ],
            'area': [
                44.38, 10.59, 43.38,
                11.59,
            ],
            'format': 'netcdf',
        },
        'wind_precipitation.nc')


def create_dataframe_wind_precipitation(file_path):
    dataset = xr.open_dataset(file_path)

    print(dataset)

    # Estrai le variabili specifiche
    u10 = dataset['u10']
    v10 = dataset['v10']
    tp = dataset['tp']

    # Converti ciascuna DataArray in un DataFrame pandas
    df_u10 = u10.to_dataframe().reset_index()
    df_v10 = v10.to_dataframe().reset_index()
    df_tp = tp.to_dataframe().reset_index()

    df_merged = pd.merge(df_u10, df_v10, on=['time', 'expver', 'latitude', 'longitude'])
    df_final = pd.merge(df_merged, df_tp, on=['time', 'expver', 'latitude', 'longitude'])

    df_final = df_final[(df_final['latitude'] == 43.8800) & (df_final['longitude'] == 11.0900)]
    df_final = df_final.groupby('time').first().reset_index()

    # Calcolo il modulo della velocità del vento
    df_final['wind'] = np.sqrt(np.square(df_final['u10']) + np.square(df_final['v10']))
    df_final = df_final.drop(columns=['u10', 'v10'])

    # Converto le precipitazioni totali in millimetri
    df_final['tp'] = df_final['tp'] * 1000
    print(df_final)

    return df_final


def merge_dataframe(file_path, df_w_p):
    df = pd.read_csv(file_path, sep=';')
    df_w_p['time'] = pd.to_datetime(df_w_p['time'])
    df['bucket_start_timestamp'] = pd.to_datetime(df['bucket_start_timestamp'])
    merged_df = pd.merge(df_w_p, df, left_on='time', right_on='bucket_start_timestamp', how='inner')
    merged_df = merged_df[
        ['bucket_start_timestamp', 'station_name', 'co2', 'extT', 'pm10', 'pm25', 'rh', 'wind', 'tp',
         'day_of_week']]
    print(merged_df)
    if os.path.exists('complete_dataframe.csv'):
        os.remove('complete_dataframe.csv')
    merged_df.to_csv('complete_dataframe.csv', index=False, sep=';')


if __name__ == "__main__":
    # request_cds()
    df_w_p = create_dataframe_wind_precipitation('wind_precipitation.nc')
    merge_dataframe('complete_dataframe.csv', df_w_p)
