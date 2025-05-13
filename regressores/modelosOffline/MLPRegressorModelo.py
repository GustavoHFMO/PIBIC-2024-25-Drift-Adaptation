from sklearn.neural_network import MLPRegressor
from regressores.ModeloBase import ModeloAtivo
from utils.FileManager import FileManager

class MLPRegressorModelo(ModeloAtivo):
    def __init__(self, **kwargs):
        super().__init__()
        
        self.name = "MLPRegressor"  # Define antes de carregar os parâmetros

        # Se não houver parâmetros passados, carrega os parâmetros do JSON
        if not kwargs:
            kwargs = FileManager.carregar_parametros_do_json(self.name)  # Usando o método da classe FileManager

        self.modelo = MLPRegressor(**kwargs)
    
    def treinar(self, X, y):
        self.modelo.fit(X, y)

    def prever(self, X):
        return self.modelo.predict(X)