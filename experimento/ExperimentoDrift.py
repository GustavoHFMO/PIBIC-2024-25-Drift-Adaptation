import numpy as np
from preprocessamento.SeriesProcessor import SeriesProcessor

class Experimento:
    def __init__(self, series, modelos, tamanho_batch=100, lags=5, repeticoes=3):
        self.series = series
        self.modelos = modelos  # Lista de dicionários com {"nome", "avaliador", "modelo", "detector"}
        self.tamanho_batch = tamanho_batch
        self.lags = lags
        self.repeticoes = repeticoes
        
    def preprocessar_serie(self, nome_serie):
        """
        Baixa, normaliza e transforma uma série temporal em janelas X e Y.
        """
        
        serie = SeriesProcessor.carregar_serie_csv(nome_serie)
        serie = SeriesProcessor.normalizar_serie(serie)
        X, Y = SeriesProcessor.criar_janela_temporal(serie, self.lags)
                
        return X, Y
    
    def executar(self):
        """
        Executa cada modelo N vezes para cada uma das séries
        """
        resultados = []

        for nome_serie in self.series:
            X, Y = self.preprocessar_serie(nome_serie)

            for modelo_cfg in self.modelos:
                nome_modelo = modelo_cfg["nome"]
                avaliador = modelo_cfg["avaliador"]
                modelo = modelo_cfg["modelo"]
                deterministico = modelo_cfg["deterministico"]
                detector = modelo_cfg.get("detector")

                print(f"Executando {nome_modelo} na série: {nome_serie}")

                if deterministico and detector:
                   
                    _, detecs, mae = avaliador.executar_avaliacao(X, Y, self.tamanho_batch, modelo, detector)
                    
                    for repeticao in range(self.repeticoes):
                        resultados.append({
                            "serie": nome_serie,
                            "modelo": nome_modelo,
                            "repeticao": repeticao + 1,
                            "mae": float(np.ravel(mae)[0]),
                            "qtd_deteccoes": len(detecs)
                        })
                
                elif not deterministico and detector:
                  
                    for repeticao in range(self.repeticoes):
                        
                        _, detecs, mae = avaliador.executar_avaliacao(X, Y, self.tamanho_batch, modelo, detector, seed=repeticao)
                        
                        resultados.append({
                            "serie": nome_serie,
                            "modelo": nome_modelo,
                            "repeticao": repeticao + 1,
                            "mae": float(np.ravel(mae)[0]),
                            "qtd_deteccoes": len(detecs)
                        })

                elif deterministico and detector == None:

                    _, mae = avaliador.executar_avaliacao(X, Y, self.tamanho_batch, modelo)

                    for repeticao in range(self.repeticoes):
                        resultados.append({
                            "serie": nome_serie,
                            "modelo": nome_modelo,
                            "repeticao": repeticao + 1,
                            "mae": float(np.ravel(mae)[0]),
                            "qtd_deteccoes": None
                        })
                
                elif not deterministico and detector == None:

                    for repeticao in range(self.repeticoes):
                        
                        _, mae = avaliador.executar_avaliacao(X, Y, self.tamanho_batch, modelo)
                        
                        resultados.append({
                            "serie": nome_serie,
                            "modelo": nome_modelo,
                            "repeticao": repeticao + 1,
                            "mae": float(np.ravel(mae)[0]),
                            "qtd_deteccoes": None
                        })

        return resultados


    
