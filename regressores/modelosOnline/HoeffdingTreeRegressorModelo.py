from river import tree
from regressores.ModeloBase import ModeloPassivo

class HoeffdingTreeRegressorModelo(ModeloPassivo):
    def __init__(self, grace_period=100, leaf_prediction="adaptive"):
        super().__init__()
        self.modelo = tree.HoeffdingTreeRegressor()
        self.name = "Hoff_Tree"
