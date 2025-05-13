from sklearn.svm import SVR
from regressores.ModeloBase import ModeloAtivo
from utils.FileManager import FileManager

class SVRModelo(ModeloAtivo):
    def __init__(self, **kwargs):
        super().__init__()
        
        self.name = "SVR"  # Define antes de carregar os parâmetros

        # Se não houver parâmetros passados, carrega os parâmetros do JSON
        if not kwargs:
            kwargs = FileManager.carregar_parametros_do_json(self.name)  # Usando o método da classe FileManager

        self.modelo = SVR(**kwargs)
