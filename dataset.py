import os

from request import create_dataframe, get_station_data, create_dataframe_graph, add_data, DATE_START, DATE_END, \
    complete_dataframe

if __name__ == "__main__":
    #################### Creazione dataset ######################

    get_station_data(DATE_START, DATE_END)

    dataframe = create_dataframe()

    df_pivot = create_dataframe_graph(dataframe)
    if os.path.exists('data/pivot_dataframe.csv'):
        os.remove('data/pivot_dataframe.csv')
    df_pivot.to_csv('pivot_dataframe.csv', index=True, sep=';')

    add_data('pivot_dataframe.csv')

    print('Sto sostituendo i valori NaN')

    complete_dataframe('data/complete_dataframe.csv')
