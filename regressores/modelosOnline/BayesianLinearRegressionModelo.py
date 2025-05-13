from river import linear_model
from regressores.ModeloBase import ModeloPassivo

class BayesianLinearRegressionModelo(ModeloPassivo):
    def __init__(self):
        super().__init__()
        self.modelo = linear_model.BayesianLinearRegression()
        self.name = "Bayesian_Linear_Online"

