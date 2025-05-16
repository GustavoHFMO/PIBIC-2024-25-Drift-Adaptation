from avaliacao.AvaliadorDriftBase import AvaliadorDriftBase
from sklearn.metrics import mean_absolute_error
from scipy.stats import ks_2samp
from river import metrics
import numpy as np
import copy

class LOR(AvaliadorDriftBase):
    def __init__(self, modelo_classe, detector_classe, incremental=False):
        self.modelo_classe = modelo_classe
        self.detector_classe = detector_classe
        self.incremental = incremental
                    
    def inicializar_modelos(self, X, y, seed):
                
        ######################## inicializando o regressor ###########################
        # Instancia o modelo com os parâmetros fornecidos
        self.modelo_atual = copy.copy(self.modelo_classe())
        
        # Treinamento do modelo usando o método 'treinar' da subclasse
        self.modelo_atual.treinar(X, y)

        # Cálculo do erro médio (adapte para modelos online, se necessário)
        erro_medio = mean_absolute_error(y, self.modelo_atual.prever(X))
        ###############################################################################
        
        
        ######################## inicializando o detector #############################
        # Instancia o detector com os parâmetros fornecidos
        self.detector_atual = copy.copy(self.detector_classe(seed=seed))
        
        # atualizando o detector
        self.detector_atual.atualizar(erro_medio)
        ###############################################################################
        
    def inicializar_modelo_rapido(self, X, y):
                
        ######################## inicializando o regressor ###########################
        # Instancia o modelo com os parâmetros fornecidos
        self.modelo_atual = copy.copy(self.modelo_online_classe())
        
        # Treinamento do modelo usando o método 'treinar' da subclasse
        self.modelo_atual.treinar(X, y)
        ###############################################################################
            
    def incrementar_janela(self, x, y):
        self.increment_window_X.append(x)
        self.increment_window_y.append(y)

    def prequential(self, X, Y, tamanho_batch, model_classe=None, detect_classe=None, seed=None):
        """
        Realiza a previsão de valores continuamente, detectando mudanças nos dados (drift)
        e retreinando o modelo quando necessário.

        Args:
            X: Dados de entrada.
            Y: Dados de saída.
            tamanho_batch: Tamanho do batch para treinamento inicial e retreinamento.
            modelo_classe: Classe do modelo a ser usado (subclasse de ModeloBase).
            detector_classe: Classe do detector de drift a ser usado (subclasse de DetectorDriftBase).

        Returns:
            predicoes: Lista de previsões.
            deteccoes: Lista de índices onde o drift foi detectado.
        """
        ### variaveis de retorno
        predicoes, erros, deteccoes = [], [], []
        mae = metrics.MAE()

        # inicializacao do modelo
        self.inicializar_modelos(X[:tamanho_batch], Y[:tamanho_batch], seed)
        self.increment_window_X = []
        self.increment_window_y = []
        
        ### variavel de controle
        drift_ativo = False

        ### processamento da stream
        for i in range(tamanho_batch, len(X)):
            
            # recebimento do dado de entrada e computacao da previsao
            entrada = X[i].reshape(1, -1)
            y_pred = self.modelo_atual.prever(entrada)
            erro = mean_absolute_error(Y[i], y_pred)

            # salvando os resultados
            predicoes.append(float(np.array(y_pred).flatten()[0]))
            erros.append(erro)
            mae.update(Y[i][0], y_pred[0])

            # atualizando o detector
            if not drift_ativo:
                # atualizando o detector para cada nova predicao
                self.detector_atual.atualizar(erro)

            # verificando se tem drift
            if self.detector_atual.drift_detectado() and not drift_ativo:
                deteccoes.append(i)                
                drift_ativo = True
        
            # ativando a estrategia de adaptacao ao drift
            if drift_ativo:
                        
                self.incrementar_janela(X[i], Y[i])
                
                if(self.incremental):
                    self.modelo_atual.treinar(self.increment_window_X, self.increment_window_y)
                else:
                    self.modelo_atual.treinar([X[i]], [Y[i]])    
                
                # resetando o modelo apos o preenchimento do batch
                if(len(self.increment_window_X) >= tamanho_batch):
                    self.inicializar_modelos(self.increment_window_X, self.increment_window_y, seed)
                    self.increment_window_X = []
                    self.increment_window_y = []
                    drift_ativo = False            
                              
        return predicoes, deteccoes, mae.get()
    