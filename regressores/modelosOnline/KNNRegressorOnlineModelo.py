from river import neighbors
from regressores.ModeloBase import ModeloPassivo

class KNNRegressorOnlineModelo(ModeloPassivo):
    def __init__(self, n_neighbors=10, aggregation_method="mean"):
        super().__init__()
        self.modelo = neighbors.KNNRegressor(n_neighbors=n_neighbors, aggregation_method=aggregation_method)
        self.name = "KNN_Online"
