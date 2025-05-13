from river import neural_net
from river import optim
from regressores.ModeloBase import ModeloPassivo

class MLPRegressorOnlineModelo(ModeloPassivo):
    def __init__(self, hidden_dims=(20,), activations=("ReLU",), optimizer=optim.Adam(0.001)):
        super().__init__()
        self.modelo = neural_net.MLPRegressor(hidden_dims=hidden_dims, activations=activations, optimizer=optimizer)
        self.name = "MLP_Online"
