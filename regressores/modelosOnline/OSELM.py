import numpy as np
from regressores.ModeloBase import ModeloPassivo

class OSELMModelo(ModeloPassivo):
    def __init__(self, n_hidden=20, activation='sigmoid', input_dim=None):
        super().__init__()
        self.n_hidden = n_hidden
        self.activation = activation
        self.input_dim = input_dim
        self.W = None  # Pesos de entrada
        self.b = None  # Bias
        self.beta = None  # Pesos de saída
        self.P = None  # Matriz de covariância
        self.name = "OS_ELM_Online"

    def _init_weights(self, input_dim):
        self.W = np.random.randn(self.n_hidden, input_dim)
        self.b = np.random.randn(self.n_hidden, 1)

    def _activation(self, X):
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-(X)))
        elif self.activation == 'tanh':
            return np.tanh(X)
        elif self.activation == 'relu':
            return np.maximum(0, X)
        else:
            raise ValueError("Função de ativação não suportada.")

    def treinar(self, X, y):
        for i in range(len(X)):
            x_i = np.array(X[i]).reshape(-1, 1)
            y_i = np.array([[y[i][0]]])

            if self.W is None:
                self._init_weights(x_i.shape[0])

            H_i = self._activation(self.W @ x_i + self.b)  # n_hidden x 1

            if self.P is None:
                self.P = np.linalg.inv(H_i @ H_i.T + np.eye(self.n_hidden) * 1e-3)
                self.beta = self.P @ H_i * y_i
            else:
                P_H = self.P @ H_i
                denom = 1 + H_i.T @ P_H
                self.P = self.P - (P_H @ P_H.T) / denom
                self.beta = self.beta + self.P @ H_i * (y_i - H_i.T @ self.beta)

    def prever(self, X):
        predicoes = []
        for i in range(len(X)):
            x_i = np.array(X[i]).reshape(-1, 1)
            H_i = self._activation(self.W @ x_i + self.b)
            pred = float(H_i.T @ self.beta) if self.beta is not None else 0.0
            predicoes.append(pred)
        return predicoes
