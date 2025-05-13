import matplotlib.pyplot as plt
import numpy as np

class Visualizer:
    """
    Classe para visualização de resultados de detecção de drift.
    """

    @staticmethod
    def plotar_resultados(Y, lista_predicoes, labels_algoritmos, deteccoes_por_modelo, tamanho_batch, titulo_plot):
        """
        Plota os resultados de predição e detecção de drift.

        Args:
            Y: Dados reais
            lista_predicoes: Lista de conjuntos de predições
            labels_algoritmos: Rótulos para cada conjunto de predições
            deteccoes_por_modelo: Lista de índices de detecção para cada modelo
            tamanho_batch: Tamanho do batch usado no treinamento
        """
        plt.figure(figsize=(15, 8))
        indices = range(tamanho_batch, tamanho_batch + len(Y[tamanho_batch:]))

        # Plotar valores verdadeiros
        plt.plot(indices, Y[tamanho_batch:tamanho_batch + len(indices)],
                 label="Verdadeiro", linewidth=1.2, color='black')

        # Plotar cada conjunto de previsões
        for i, predicoes in enumerate(lista_predicoes):
            Y_plot = Y[tamanho_batch:tamanho_batch + len(predicoes)]
            predicoes = predicoes[:len(Y_plot)]  # Garantir mesmo tamanho
            label = labels_algoritmos[i] if i < len(labels_algoritmos) else f"Previsões {i+1}"
            plt.plot(indices[:len(predicoes)], predicoes, label=label, linewidth=1.2)

            # Obter detecções para este modelo (se disponíveis)
            modelo_deteccoes = deteccoes_por_modelo[i] if i < len(deteccoes_por_modelo) else []

            # Aumentar o tamanho dos pontos de detecção para este modelo
            if modelo_deteccoes:
                # Verificar se cada ponto de detecção está nos índices válidos
                valid_deteccoes = [d for d in modelo_deteccoes if d < len(Y)]

                if valid_deteccoes:
                    # Usar uma cor diferente para cada modelo
                    cor = plt.cm.tab10(i / 10) if i < 10 else plt.cm.Set3((i-10) / 10)

                    plt.scatter(valid_deteccoes, [Y[d] for d in valid_deteccoes],
                               color=cor, marker='o',
                               label=f"Drift - {label}", zorder=3, s=80)

                    # Destacar áreas pós-retreino com fundo colorido
                    for idx, d in enumerate(valid_deteccoes):
                        if d + tamanho_batch < len(indices):
                            next_end = min(d + tamanho_batch, indices[-1])
                            plt.axvspan(d, next_end, alpha=0.1, color=cor, label='_nolegend_')

                    # Adicionar anotações para mostrar diferenças
                    for d in valid_deteccoes[:3]:  # Limitar a 3 anotações por modelo para não sobrecarregar
                        if d + 5 < len(indices):
                            plt.annotate(f"Retreino {label}",
                                        xy=(d, Y[d]),
                                        xytext=(d+10, Y[d]+0.1 * (i+1)),  # Deslocar verticalmente
                                        arrowprops=dict(facecolor=cor, shrink=0.05, width=1.5),
                                        fontsize=9,
                                        color=cor)

                    print(f"\nDrift detectado para {label} nos índices:", valid_deteccoes)
                else:
                    print(f"\nNenhum drift válido detectado para {label}.")
            else:
                print(f"\nNenhum drift detectado para {label}.")

        plt.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
        plt.title(f"{titulo_plot}", fontsize=14)
        plt.grid(True, alpha=0.4)
        plt.tight_layout()

        plt.show()

    @staticmethod
    def plotar_resultados_multi(Y, lista_predicoes, labels_algoritmos, deteccoes_dict, tamanho_batch, detector_or_modelo):
        """
        Plota os resultados com múltiplas detecções.

        Args:
            Y: Dados reais
            lista_predicoes: Lista de conjuntos de predições
            labels_algoritmos: Rótulos para cada conjunto de predições
            deteccoes_dict: Dicionário com índices de detecção para cada modelo
            tamanho_batch: Tamanho do batch usado no treinamento
        """
        plt.figure(figsize=(15, 8))
        indices = range(tamanho_batch, tamanho_batch + len(Y[tamanho_batch:]))

        # Plotar valores verdadeiros
        plt.plot(indices, Y[tamanho_batch:tamanho_batch + len(indices)],
                 label="Verdadeiro", linewidth=1.2, color='black')

        # Plotar cada conjunto de previsões
        for i, predicoes in enumerate(lista_predicoes):
            if i >= len(labels_algoritmos):
                continue

            modelo_nome = labels_algoritmos[i]
            Y_plot = Y[tamanho_batch:tamanho_batch + len(predicoes)]
            predicoes = predicoes[:len(Y_plot)]  # Garantir mesmo tamanho

            # Cor para este modelo
            cor = plt.cm.tab10(i / 10) if i < 10 else plt.cm.Set3((i-10) / 10)

            plt.plot(indices[:len(predicoes)], predicoes, label=modelo_nome, linewidth=1.2, color=cor)

            # Obter detecções para este modelo
            if modelo_nome in deteccoes_dict:
                modelo_deteccoes = deteccoes_dict[modelo_nome]

                if modelo_deteccoes:
                    # Filtrar detecções válidas
                    valid_deteccoes = [d for d in modelo_deteccoes if d < len(Y)]

                    if valid_deteccoes:
                        plt.scatter(valid_deteccoes, [Y[d] for d in valid_deteccoes],
                                  color=cor, marker='o', s=60,
                                  label=f"Drift - {modelo_nome}", zorder=3)

                        # Destacar áreas pós-retreino
                        for d in valid_deteccoes:
                            if d + tamanho_batch < len(indices):
                                next_end = min(d + tamanho_batch, indices[-1])
                                plt.axvspan(d, next_end, alpha=0.1, color=cor, label='_nolegend_')

                        print(f"\nDrift detectado para {modelo_nome} nos índices:", valid_deteccoes)
                    else:
                        print(f"\nNenhum drift válido detectado para {modelo_nome}.")
                else:
                    print(f"\nNenhum drift detectado para {modelo_nome}.")

        plt.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
        plt.title(f"Predições e Detecção de Drift com Retreino variando os algoritmos de Detecção fixando {detector_or_modelo}", fontsize=14)
        plt.grid(True, alpha=0.4)
        plt.tight_layout()

        plt.show()
