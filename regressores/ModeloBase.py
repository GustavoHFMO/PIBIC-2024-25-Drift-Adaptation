from abc import ABC, abstractmethod

class ModeloBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def treinar(self, X, y):
        pass

    @abstractmethod
    def prever(self, X):
        pass
    
    
class ModeloPassivo(ModeloBase):
    def __init__(self):
        pass

    def treinar(self, X, y):
        for i in range(len(X)):
            X_dict = {f"t{j+1}": value for j, value in enumerate(X[i])}
            self.modelo.learn_one(X_dict, y[i][0])

    def prever(self, X):
        predicoes = []
        for i in range(len(X)):
            X_dict = {f"t{j+1}": value for j, value in enumerate(X[i])}
            predicao = self.modelo.predict_one(X_dict)
            predicoes.append(predicao)
        return predicoes
    

class ModeloAtivo(ModeloBase):
    def __init__(self):
        pass

    def treinar(self, X, y):
        self.modelo.fit(X, y)

    def prever(self, X):
        return self.modelo.predict(X)