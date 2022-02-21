# 1801042103
# Ozan GECKIN
import pandas as pd

from visualization import plot2d_data
from chameleon import chameleonCluster

if __name__ == "__main__":
    dataFrame = pd.read_csv('./datasets/smileface.csv', sep=',')
    result = chameleonCluster(dataFrame,k=10, k_neighbor_number=30, subCluster=50)
    plot2d_data(result)
