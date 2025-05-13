from sklearn.neighbors import KNeighborsRegressor
from regressores.ModeloBase import ModeloAtivo
from utils.FileManager import FileManager


class KneighborsRegressorModelo(ModeloAtivo):
    def __init__(self, **kwargs):
        super().__init__()
        self.name = "KNeighborsRegressor"  # Define o nome para carregar parâmetros

        # Se não houver parâmetros passados, carrega os parâmetros do JSON
        if not kwargs:
            kwargs = FileManager.carregar_parametros_do_json(self.name)  # Usando o método da classe FileManager

        # Cria o modelo com os parâmetros (se fornecidos ou do JSON)
        self.modelo = KNeighborsRegressor(**kwargs)
