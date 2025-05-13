from avaliacao.AvaliadorDriftBase import AvaliadorDriftBase
from sklearn.metrics import mean_absolute_error
from dtaidistance import dtw
from river import metrics
import numpy as np
import copy

class WinDTW(AvaliadorDriftBase):
    def __init__(self, modelo_classe, detector_classe, n_std, n_subsamples=5, n_sliding_size=50):
        self.modelo_classe = modelo_classe
        self.detector_classe = detector_classe
        self.n_subsamples = n_subsamples
        self.n_sliding_size = n_sliding_size
        self.n_std = n_std
    
    def inicializar_modelos(self, X, y):
                
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
        self.detector_atual = copy.copy(self.detector_classe())
        
        # atualizando o detector
        self.detector_atual.atualizar(erro_medio)
        ###############################################################################
    
    def inicializar_janelas(self, X, y):
        self.fixed_window_X = copy.copy(X)
        self.fixed_window_y = copy.copy(y)
        
        self.sliding_window_X = copy.copy(X)
        self.sliding_window_y = copy.copy(y)
        
        self.increment_window_X = []
        self.increment_window_y = []
        
        self.definir_limiar()
    
    def definir_limiar_old(self):
        
        # definindo o indice máximo que a janela deslizante pode chegar
        max_start = len(self.fixed_window_y) - self.n_sliding_size
        
        # var para salvar os exemplos de series
        samples = []
        
        # gerando varias sub series para definir o limiar
        for _ in range(self.n_subsamples):
            
            # definindo o inicio aleatorio
            start = np.random.randint(0, max_start)
            
            # gerando a subserie
            subwindow = self.fixed_window_y[start:start + self.n_sliding_size]
            
            # calculando a distancia da subserie para a serie fixa
            dist = dtw.distance(self.fixed_window_y, subwindow)
            
            # salvando as series
            samples.append(dist)

        # gerando o limiar
        self.distances = np.array(samples)
        self.min_distances = np.min(self.distances)
        self.mean_distances = np.mean(self.distances)
        self.std_distances = np.std(self.distances)
        
    def definir_limiar(self):
        
        # definindo o passo para deslizar a janela
        window_length = len(self.fixed_window_y)
        step = (window_length - self.n_sliding_size) // (self.n_subsamples - 1)
        
        # var para salvar os exemplos de series
        samples = []

        for i in range(self.n_subsamples):
            
            # definindo o inicio e fim
            start = i * step
            end = start + self.n_sliding_size
            
            # Evita janelas incompletas
            if end > window_length:
                break  
            
            # definindo a subjanela
            subwindow = self.fixed_window_y[start:end]
            
            # calculando e salvando a distancia
            dist = dtw.distance(self.fixed_window_y, subwindow)
            samples.append(dist)

        # gerando o limiar
        self.distances = np.array(samples)
        self.min_distances = np.min(self.distances)
        self.mean_distances = np.mean(self.distances)
        self.std_distances = np.std(self.distances)
           
    def deslizar_janela(self, x, y):
        self.sliding_window_X = np.delete(self.sliding_window_X, 0, axis=0)
        self.sliding_window_y = np.delete(self.sliding_window_y, 0, axis=0)
        
        self.sliding_window_X = np.append(self.sliding_window_X, [x], axis=0)
        self.sliding_window_y = np.append(self.sliding_window_y, [y], axis=0)
            
    def incrementar_janela(self, x, y):
        self.increment_window_X.append(x)
        self.increment_window_y.append(y)
    
    def distancia_entre_janelas(self):
        # Calcula a distância DTW total entre a janela fixa e a deslizante
        distance, _ = dtw.warping_paths(self.fixed_window_y, self.sliding_window_y)
        
        # Retorna a distância geral (custo total)
        return distance
    
    def comparar_janelas(self):
        
        dist = self.distancia_entre_janelas()

        if dist > self.min_distances + self.n_std * self.std_distances:
            status = 'drift'
        else:
            status = 'normal'
            
        return status
       
    def prequential(self, X, Y, tamanho_batch, model_classe=None, detect_classe=None):
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
        self.inicializar_modelos(X[:tamanho_batch], Y[:tamanho_batch])
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
            predicoes.append(y_pred)
            erros.append(erro)
            mae.update(Y[i][0], y_pred[0])

            # atualizando o detector
            if not drift_ativo:
                self.detector_atual.atualizar(erro)
                
                # deslizando a janela sobre os dados
                self.deslizar_janela(X[i], Y[i])

            # verificando se tem drift
            if self.detector_atual.drift_detectado() and not drift_ativo:
                deteccoes.append(i)
                status = self.comparar_janelas() 
                drift_ativo = True
        
            # ativando a estrategia de adaptacao ao drift
            if drift_ativo:
                
                if status == 'normal':
                    
                    drift_ativo = False
                    self.inicializar_modelos(self.sliding_window_X, self.sliding_window_y)
                    self.inicializar_janelas(self.sliding_window_X, self.sliding_window_y)
                    
                elif status == 'drift':
                    
                    self.incrementar_janela(X[i], Y[i])
                    
                    if(len(self.increment_window_X)  >= tamanho_batch):
                        
                        drift_ativo = False
                        self.inicializar_modelos(self.increment_window_X, self.increment_window_y)
                        self.inicializar_janelas(self.increment_window_X, self.increment_window_y)
                        
        return [float(p.flatten()[0]) for p in predicoes], deteccoes, mae.get()
    