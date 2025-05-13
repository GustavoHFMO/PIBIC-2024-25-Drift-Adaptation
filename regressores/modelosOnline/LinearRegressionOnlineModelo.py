from river import linear_model
from regressores.ModeloBase import ModeloPassivo

class LinearRegressionOnlineModelo(ModeloPassivo):
    def __init__(self):
        super().__init__()
        self.modelo = linear_model.LinearRegression()
        self.name = "LR_Online"
