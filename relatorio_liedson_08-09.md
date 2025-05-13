## Criação e Gestão do Pool

* Novos modelos são adicionados quando a janela atinge tamanho predefinido
* Substituição baseada em desempenho (apenas o pior é substituído)
* Modelos no pool são idênticos em estrutura, diferindo apenas nos dados de treinamento
* Ausência de mecanismos para garantir diversidade entre modelos

## Adaptção a Drift

* Durante drift, coleta uma nova janela de dados
* Avalia gradualmente os modelos existentes na janela parcial
* Seleciona o modelo de melhor desempenho na janela atual
* Reinicia com backup se janela ficar muito grande
* Aprendizado contínuo do modelo atual e backup durante todo o processo

## Previsão e Avaliação

* Utiliza apenas o modelo atual para previsões (sem ensemble)
* MAE atual (0.01723559) próximo mas ainda abaixo do BayesianOnline (0.01753886)
* Apresenta resultados estáveis mas com potencial para melhoria

## Possiveis Melhorias

Eu pensei nessas como possíveis melhorias que podem ter um alto impacto:

* Implementar previsão ensemble: Combinando previsões do modelo atual com modelos do pool
* Treinar modelo completamente novo após drift: Usando dados da janela pós-drift
* Retreinar com mais dados antes de restaurar o detector após drift