from avaliacao.AvaliadorDriftBase import AvaliadorDriftBase
from sklearn.metrics import mean_absolute_error
from river import metrics
import numpy as np
import copy

class OASIS(AvaliadorDriftBase):
    def __init__(self, modelo_classe, detector_classe, len_pool, len_add, pool_exclusion="performance"):
        self.modelo_classe = modelo_classe
        self.detector_classe = detector_classe
        self.len_pool = len_pool
        self.len_add = len_add
        self.pool_exclusion = pool_exclusion

    def inicializar_modelos(self, X, y, seed):
        """
        Inicializa instâncias de modelo e detector.

        Args:
            modelo_classe: Classe do modelo
            detector_classe: Classe do detector de drift

        Returns:
            tuple: (modelo_instancia, detector_instancia)
        """

        ######################## inicializando o regressor ###########################
        # Instancia o modelo com os parâmetros fornecidos
        self.modelo_atual = copy.copy(self.modelo_classe())

        # Treinamento do modelo usando o método 'treinar' da subclasse
        self.modelo_atual.treinar(X, y)

        # criando um modelo de backup
        self.modelo_backup = copy.copy(self.modelo_atual)

        # Cálculo do erro médio (adapte para modelos online, se necessário)
        erro_medio = mean_absolute_error(y, self.modelo_atual.prever(X))
        ###############################################################################


        ######################## inicializando o detector #############################
        # Instancia o detector com os parâmetros fornecidos
        self.detector_atual = copy.copy(self.detector_classe(seed=seed))

        # atualizando o detector
        self.detector_atual.atualizar(erro_medio)
        ###############################################################################

    def ativar_modelo_backup(self, X, y):
        """
        Inicializa instâncias de modelo e detector.

        Args:
            modelo_classe: Classe do modelo
            detector_classe: Classe do detector de drift

        Returns:
            tuple: (modelo_instancia, detector_instancia)
        """

        ######################## inicializando o regressor ###########################
        # Instancia o modelo com os parâmetros fornecidos
        self.modelo_atual = copy.copy(self.modelo_backup)

        # Cálculo do erro médio (adapte para modelos online, se necessário)
        erro_medio = mean_absolute_error(y, self.modelo_atual.prever(X))
        ###############################################################################


        ######################## inicializando o detector #############################
        # Instancia o detector com os parâmetros fornecidos
        self.detector_atual = copy.copy(self.detector_classe())

        # atualizando o detector
        self.detector_atual.atualizar(erro_medio)
        ###############################################################################

    def inicializar_janela(self):
        self.janela_X = []
        self.janela_y = []

    def inicializar_pool(self):
        self.pool = []

    def incrementar_janela(self, x, y):
        self.janela_X.append(x)
        self.janela_y.append(y)

    def avaliar_tamanho_janela(self):
        return True if len(self.janela_X) == self.len_add else False

    def treinar_submodelo(self):

        # Instancia o modelo com os parâmetros fornecidos
        modelo_instancia = copy.copy(self.modelo_classe())

        modelo_instancia.treinar(self.janela_X, self.janela_y)

        return modelo_instancia

    # ALTEREI AQUI
    def add_pool(self, modelo):

        # Avalia o erro do novo modelo nesta janela
        erro_novo = mean_absolute_error(self.janela_y, modelo.prever(self.janela_X))

        if len(self.pool) < self.len_pool:
            self.pool.append(modelo)
            
        elif(self.pool_exclusion == "performance"):

            # Avaliar erros dos modelos no pool
            erros_pool = [
                mean_absolute_error(self.janela_y, m.prever(self.janela_X))
                for m in self.pool
            ]
            pior_idx = np.argmax(erros_pool)
            pior_erro = erros_pool[pior_idx]

            # Se o modelo novo for melhor, substitui o pior
            if erro_novo < pior_erro:
                self.pool[pior_idx] = modelo
                
        elif(self.pool_exclusion == "time"):
            self.pool.append(modelo)
            self.pool.pop(0)

    def popular_pool(self, x, y):

        self.incrementar_janela(x, y)

        if self.avaliar_tamanho_janela():
            self.add_pool(self.treinar_submodelo())
            self.inicializar_janela()

    def avaliar_modelos(self, X, Y):

        # avaliando a performance do modelo atual
        self.modelo_atual = copy.copy(self.modelo_backup)
        performance_base = mean_absolute_error(Y, self.modelo_atual.prever(X))

        # avaliando a performance dos modelos do pool
        for modelo in self.pool:

            performance = mean_absolute_error(Y, modelo.prever(X))

            if(performance < performance_base):
                self.modelo_atual = copy.copy(modelo)
                performance_base = performance

    def prequential(self, X, Y, tamanho_batch, model_classe=None, detect_classe=None, seed=None):
        """
        Realiza a previsão de valores continuamente para algoritmos online,
        sem detecção de drift e retreinamento.

        Args:
            X: Dados de entrada.
            Y: Dados de saída.
            tamanho_batch: Tamanho do batch para treinamento inicial.
            modelo_classe: Classe do modelo a ser usado (subclasse de ModeloBase).

        Returns:
            predicoes: Lista de previsões.
        """

        ### variaveis de retorno
        ### variaveis de retorno
        predicoes, erros, deteccoes = [], [], []
        mae = metrics.MAE()

        # inicializacao do modelo
        self.inicializar_modelos(X[:tamanho_batch], Y[:tamanho_batch], seed)
        self.inicializar_pool()
        self.inicializar_janela()

        # Controle de janela de atualização após drift
        drift_ativo = False

        ### processamento da stream
        for i in range(tamanho_batch, len(X)):


            # recebimento do dado de entrada e computacao da previsao
            y_pred = self.modelo_atual.prever([X[i]])
            erro = mean_absolute_error(Y[i], np.array([y_pred]))


            # salvando os resultados
            predicoes.append(y_pred)
            mae.update(Y[i][0], y_pred)
            erros.append(erro)


            # Retreinar um novo modelo a cada len_add
            self.popular_pool(X[i], Y[i])


            # Atualiza o detector
            if not drift_ativo:
                self.detector_atual.atualizar(erro)
                self.modelo_atual.treinar([X[i]], [Y[i]])
            else:
                self.modelo_backup.treinar([X[i]], [Y[i]])


            # Verifica detecção de drift
            if self.detector_atual.drift_detectado() and not drift_ativo:
                deteccoes.append(i)
                drift_ativo = True
                janela_X, janela_y = [], []


            # ativando a estrategia de adaptacao ao drift
            if drift_ativo:
                janela_X.append(X[i])
                janela_y.append(Y[i])

                if len(janela_X) <= tamanho_batch:
                    self.avaliar_modelos(janela_X, janela_y)

                else:
                    drift_ativo = False
                    # ALTEREI AQUI
                    self.modelo_atual.treinar(janela_X, janela_y)
                    self.ativar_modelo_backup(janela_X, janela_y)


        #print("Modelo utilizado:", modelo)
        #print(f"MAE Modelo Online: {mae.get()}")
        return [float(p.flatten()[0]) for p in np.asarray(predicoes)], deteccoes, mae.get()
