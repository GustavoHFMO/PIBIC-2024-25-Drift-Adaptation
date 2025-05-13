from river import preprocessing
import yfinance as yf
import pandas as pd
import numpy as np
import time
import os

class SeriesProcessor:
    """
    Classe para processamento de séries temporais financeiras.
    """

    @staticmethod
    def carregar_serie_csv(nome_serie, pasta="series"):
        """
        Lê um arquivo CSV com dados de uma série temporal e retorna um array de fechamentos sem NaNs.

        Args:
            nome_serie: Nome do arquivo da série (sem extensão .csv)
            pasta: Pasta onde os arquivos estão salvos

        Returns:
            np.ndarray com os valores de fechamento ajustado (float), sem NaNs
        """
        caminho = os.path.join(pasta, f"{nome_serie}.csv")

        # Ignora as duas primeiras linhas e lê os dados
        df = pd.read_csv(caminho, skiprows=2, names=["Date", "Close"])

        # Remove valores NaN e retorna como array
        return df["Close"].dropna().values.reshape(-1, 1)


    @staticmethod
    def baixar_e_salvar_series(series, pasta_destino="series", intervalo="1d"):
        """
        Baixa e salva os dados de fechamento ajustado de várias séries temporais.

        Args:
            series: Lista de símbolos do Yahoo Finance
            pasta_destino: Nome da pasta onde os arquivos CSV serão salvos
            intervalo: Intervalo dos dados ("1d", "1wk", "1mo", etc.)
        """
        os.makedirs(pasta_destino, exist_ok=True)

        for symbol in series:
            print(f"Baixando dados de {symbol}...")
            try:
                data = yf.download(symbol, period="max", interval=intervalo)
                time.sleep(3)  # Espera 3 segundos entre os downloads
                if not data.empty:
                    nome_arquivo = symbol.replace('^', '').replace('=', '') + ".csv"
                    caminho_arquivo = os.path.join(pasta_destino, nome_arquivo)
                    data[["Close"]].to_csv(caminho_arquivo)
                    print(f"Salvo em: {caminho_arquivo}")
                else:
                    print(f"Nenhum dado retornado para {symbol}.")
            except Exception as e:
                print(f"Erro ao baixar {symbol}: {e}")
                
    @staticmethod
    def baixar_dados(symbol, periodo="5y", intervalo="1d"):
        """
        Baixa dados de séries temporais do Yahoo Finance.

        Args:
            symbol: Símbolo da ação ou índice
            periodo: Período para baixar (ex: "5y" para 5 anos)
            intervalo: Intervalo dos dados (ex: "1d" para diário)

        Returns:
            np.ndarray: Valores de fechamento
        """
        data = yf.download(symbol, period=periodo, interval=intervalo)
        return data["Close"].values

    @staticmethod
    def criar_janela_temporal(y, lags):
        """
        Cria padrões de entrada e saída para previsão de séries temporais.

        Args:
            y: Série temporal
            lags: Número de valores anteriores a usar como entrada

        Returns:
            tuple: (X, Y) onde X são os valores de entrada e Y os valores alvo
        """
        X, Y = [], []
        for i in range(len(y) - lags):
            X.append(y[i:i+lags])
            Y.append(y[i+lags])
        return np.array(X).reshape(-1, lags), np.array(Y)

    @staticmethod
    def normalizar_serie(serie_temporal: np.ndarray) -> np.ndarray:
        """
        Normaliza uma série temporal usando StandardScaler do River.

        Args:
            serie_temporal: Série temporal para normalizar

        Returns:
            np.ndarray: Série temporal normalizada
        """
        # Criando o scaler
        scaler = preprocessing.StandardScaler()

        # Aprendendo a escala com os dados
        for x in serie_temporal:
            scaler.learn_one({"valor": x[0]})

        # Transformando a série
        serie_normalizada = np.array([scaler.transform_one({"valor": x[0]})["valor"] for x in serie_temporal])

        # Garantindo que o shape permaneça correto
        return serie_normalizada.reshape(-1, 1)
