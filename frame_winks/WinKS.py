from avaliacao.AvaliadorDriftBase import AvaliadorDriftBase
from sklearn.metrics import mean_absolute_error
from scipy.stats import ks_2samp
from river import metrics
import numpy as np
import copy

class WinKS(AvaliadorDriftBase):
    def __init__(self, modelo_classe, modelo_online_classe, detector_classe, n_janelas=4, alpha=0.05):
        self.modelo_classe = modelo_classe
        self.modelo_online_classe = modelo_online_classe
        self.detector_classe = detector_classe
        self.n_janelas = n_janelas
        self.alpha = alpha
                    
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
           
    def inicializar_janelas(self, X, y):
        self.fixed_window_X = copy.copy(X)
        self.fixed_window_y = copy.copy(y)
        
        self.sliding_window_X = copy.copy(X)
        self.sliding_window_y = copy.copy(y)
        
        self.increment_window_X = []
        self.increment_window_y = []
    
    def atualizar_janela_incremental(self, X, y):
        self.increment_window_X = copy.copy(X)
        self.increment_window_y = copy.copy(y)
        
    def deslizar_janela(self, x, y):
        self.sliding_window_X = np.delete(self.sliding_window_X, 0, axis=0)
        self.sliding_window_y = np.delete(self.sliding_window_y, 0, axis=0)
        
        self.sliding_window_X = np.append(self.sliding_window_X, [x], axis=0)
        self.sliding_window_y = np.append(self.sliding_window_y, [y], axis=0)
            
    def incrementar_janela(self, x, y):
        self.increment_window_X = np.append(self.increment_window_X, [x], axis=0)
        self.increment_window_y = np.append(self.increment_window_y, [y], axis=0)
        #self.increment_window_X.append(x)
        #self.increment_window_y.append(y)

    def comparar_janelas_temporais(self, X, y):
        
        # definindo a quantidade de subjanelas
        tamanho = len(y)
        parte = tamanho // self.n_janelas
        
        # separando os dados em subjanelas
        janelas_y = [y[i*parte:(i+1)*parte] for i in range(self.n_janelas)]
        janelas_X = [X[i*parte:(i+1)*parte] for i in range(self.n_janelas)]
        
        # definindo a última janela como parâmetro 
        ultima_y = janelas_y[-1]
        ultima_X = janelas_X[-1]
        
        # variável para salvar os dados similares
        similares_y = [ultima_y]
        similares_X = [ultima_X]
        
        # comparacao da ultima_y janela com as mais antigas
        for i in reversed(range(self.n_janelas - 1)):
            
            # comparacao
            _, p = ks_2samp(janelas_y[i], ultima_y)
            
            if p >= self.alpha:
                similares_y.insert(0, janelas_y[i])
                similares_X.insert(0, janelas_X[i])
            else:
                break
        return np.concatenate(similares_X), np.concatenate(similares_y)

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
        self.inicializar_janelas(X[:tamanho_batch], Y[:tamanho_batch])
                
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
                # deslizando a janela sobre os dados
                self.deslizar_janela(X[i], Y[i])

            # verificando se tem drift
            if self.detector_atual.drift_detectado() and not drift_ativo:
                                
                deteccoes.append(i)
                
                X_new, y_new = self.comparar_janelas_temporais(self.sliding_window_X, self.sliding_window_y)           
                self.atualizar_janela_incremental(X_new, y_new)
                self.inicializar_modelo_rapido(self.sliding_window_X, self.sliding_window_y)
                
                print(i, "-", len(X_new))
                
                drift_ativo = True
        
            # ativando a estrategia de adaptacao ao drift
            if drift_ativo:
                        
                self.incrementar_janela(X[i], Y[i])
                self.modelo_atual.treinar([X[i]], [Y[i]])
                
                # resetando o modelo apos o preenchimento do batch
                if(len(self.increment_window_X) >= tamanho_batch):
                    self.inicializar_modelos(self.increment_window_X, self.increment_window_y, seed)
                    self.inicializar_janelas(self.increment_window_X, self.increment_window_y)
                    drift_ativo = False            
                              
        return predicoes, deteccoes, mae.get()
    