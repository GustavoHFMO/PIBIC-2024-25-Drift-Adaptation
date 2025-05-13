import json
import os
import numpy as np
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.model_selection import PredefinedSplit

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from utils.FileManager import FileManager


class OtimizadorModelosSklearn:
    def __init__(self, n_iter=25, scoring='neg_mean_squared_error', random_state=42, test_size=0.2):
        """
        Inicializa o otimizador com parâmetros padrão e configura os espaços de busca 
        para cada modelo.
        """
        self.n_iter = n_iter
        self.scoring = scoring
        self.random_state = random_state
        self.test_size = test_size
        self.espacos = self._definir_espacos()
        self.modelos = self._definir_modelos()

    def _definir_modelos(self):
        """
        Retorna um dicionário com os modelos do scikit-learn que serão otimizados.
        """
        return {
            "LinearRegression": LinearRegression,
            "KNeighborsRegressor": KNeighborsRegressor,
            "Lasso": Lasso,
            "MLPRegressor": MLPRegressor,
            "RandomForestRegressor": RandomForestRegressor,
            "Ridge": Ridge,
            "SVR": SVR
        }

    def _definir_espacos(self):
        """
        Define os espaços de busca para os hiperparâmetros de cada modelo.
        """
        return {
            "LinearRegression": {
                'fit_intercept': Categorical([True, False]),
                'copy_X': Categorical([True, False]),
                'positive': Categorical([True, False])
            },
            "KNeighborsRegressor": {
                "n_neighbors": Integer(1, 20),
                "weights": Categorical(['uniform', 'distance']),
                "p": Integer(1, 2)
            },
            "Lasso": {
                "alpha": Real(0.001, 10.0, prior='log-uniform'),
                "max_iter": Integer(100, 2000),
                "tol": Real(1e-6, 1e-2, prior='log-uniform')
            },
            "MLPRegressor": {
                "hidden_layer_sizes": Categorical([(50,), (100,)]),  # 1 camada escondida
                "activation": Categorical(['relu', 'tanh']),
                "solver": Categorical(['adam', 'sgd']),
                "alpha": Real(1e-5, 1e-2, prior='log-uniform'),
                "learning_rate_init": Real(1e-4, 1e-1, prior='log-uniform')
            },
            "RandomForestRegressor": {
                "n_estimators": Integer(50, 200),
                "max_depth": Integer(5, 30),
                "min_samples_split": Integer(2, 10),
                "min_samples_leaf": Integer(1, 4)
            },
            "Ridge": {
                "alpha": Real(0.01, 10.0, prior='log-uniform'),
                "solver": Categorical(['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg'])
            },
            "SVR": {
                "C": Real(0.1, 100, prior='log-uniform'),
                "epsilon": Real(0.01, 1.0, prior='log-uniform'),
                "kernel": Categorical(['linear', 'rbf', 'poly']),
                "gamma": Real(1e-3, 1, prior='log-uniform')
            },
        }

    def _holdout_split(self, X, y):
        """
        Realiza a divisão dos dados respeitando a ordem temporal, sem embaralhamento.
        Retorna a parte de treinamento e a parte de validação.
        """
        split_index = int(len(X) * (1 - self.test_size))
        return X[:split_index], X[split_index:], y[:split_index], y[split_index:]

    def _executar_otimizacao(self, modelo, espaco, X_total, y_total, ps):
        """
        Executa a otimização bayesiana para um modelo específico.
        Retorna os melhores parâmetros encontrados.
        """
        search = BayesSearchCV(
            estimator=modelo,
            search_spaces=espaco,
            n_iter=self.n_iter,
            scoring=self.scoring,
            random_state=self.random_state,
            cv=ps,  # Validação temporal fixa
            n_jobs=-1,
            verbose=0
        )
        search.fit(X_total, y_total)
        return search.best_params_

    def _sugerir_parametros_mlp(self):
        """
        Método para sugerir parâmetros padrão para o MLPRegressor caso ocorra erro.
        """
        print("Ajustando parâmetros padrão para o MLPRegressor devido a erro.")
        # Parâmetros sugeridos caso ocorra erro
        parametros_padrao = {
            "hidden_layer_sizes": (100,),  # 1 camada escondida com 100 neurônios
            "activation": 'relu',  # Função de ativação ReLU
            "solver": 'adam',  # Solver adam é estável para a maioria dos casos
            "alpha": 1e-5,  # Regularização muito pequena
            "learning_rate_init": 0.04  # Taxa de aprendizado inicial
        }
        return parametros_padrao
    
    def otimizar(self, X, y, salvar_em="melhores_parametros.json"):
        """
        Executa a otimização bayesiana para todos os modelos definidos e salva os 
        melhores parâmetros encontrados em um arquivo JSON, apenas se o arquivo não existir.
        """
    
        # Verifica se já existe um arquivo com os melhores parâmetros salvos
        parametros_existentes = FileManager.carregar_se_existir(salvar_em)
        if parametros_existentes is not None:
            return parametros_existentes

        # Divide os dados em treino e validação usando holdout
        X_train, X_val, y_train, y_val = self._holdout_split(X, y)

        # Junta os dados novamente para uso com PredefinedSplit
        X_total = np.vstack([X_train, X_val])
        y_total = np.concatenate([y_train, y_val])

        # Cria vetor de índices para o PredefinedSplit (validação definida manualmente)
        split_index = [-1] * len(X_train) + [0] * len(X_val)
        ps = PredefinedSplit(test_fold=split_index)

        # Dicionário para armazenar os melhores parâmetros por modelo
        melhores_parametros = {}

        # Executa a otimização para cada modelo e seu respectivo espaço de busca
        for nome, espaco in self.espacos.items():
            try:
                modelo = self.modelos[nome]()  # Instancia o modelo
                # Executa a otimização bayesiana
                melhores_parametros[nome] = self._executar_otimizacao(modelo, espaco, X_total, y_total, ps)
                print(f"Melhor para {nome}: {melhores_parametros[nome]}")
            except Exception as e:
                print(f"Erro com {nome}: {e}")
                # Garante que ao menos a MLP tenha parâmetros default sugeridos
                if nome == "MLPRegressor":
                    melhores_parametros[nome] = self._sugerir_parametros_mlp()

        # Salva os melhores parâmetros encontrados em um arquivo JSON
        FileManager.salvar_json(salvar_em, melhores_parametros)

        return melhores_parametros





