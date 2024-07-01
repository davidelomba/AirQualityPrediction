from torch_geometric_temporal import temporal_signal_split
from request import training_testing, create_dataframe, lon_lat_dataframe, build_graph, THRESHOLD_DISTANCE

if __name__ == "__main__":
    dataframe = create_dataframe()

    df_lon_lat = lon_lat_dataframe(dataframe)

    dataset, node_index_to_station_name = build_graph(df_lon_lat, THRESHOLD_DISTANCE, 'data/complete_dataframe.csv')

    print(dataset)

    print("Edge Index:", dataset.edge_index)
    print("Edge Attributes:", dataset.edge_weight)
    print("Node Features:", dataset.features)
    print("Node targets:", dataset.targets)
    print(node_index_to_station_name)

    train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)

    training_testing(train_dataset, test_dataset)
