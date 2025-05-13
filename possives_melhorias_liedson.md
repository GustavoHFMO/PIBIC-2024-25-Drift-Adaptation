## Analisando o cenário atual pontos de melhorias que podem ser feitos no código:

- No método de treinar submodelo poderia ser implementada uma forma de treinar os modelos do pool com diferentes hiperparâmetros, isso porque modelos sempre treinados com os mesmo parâmetros tendem a cometer erros semelhantes. Modelos com diferentes características podem se adaptar melhor a novos conceitos.
- Talvez mudar a forma de remoção dos modelos do pool passando de mais antigos para modelos com menor desempenho, pois pode causar uma relevância maior. Mas também tem aquele porém de o mais antigo está muito atrasado do conceito atual, porém acho válido se de repente o conceito voltar a se repetir ou ser parecido.
- Talvez um ensemble de modelos pode ser válido visto que em alguns casos mais de um modelo acaba superando um único.
- Criar uma forma de aprendizado adaptativo, por exemplo: se a taxa de erro aumenta o aprendizado também irá aumentar.

---

## Algumas críticas sobre a forma como é selecionado o melhor modelo que atualmente é feita com base na MAE:

- Pode ter ali uma espécie de overffiting, pois de repente um modelo pode performar bem em um determinado espaço da série, mas não necessariamente em dados futuros.
- O modelo é avaliado somente uma vez durante o drift e isso não considera a estabilidade da performance.
- O fato também de utilizar somente o MAE para analisar acaba sendo “simples” para algo mais preciso e rigoroso.

---

## Sugestões de avaliação:

- Uma espécie de avaliação cruzada onde dividimos a janela em múltiplos conjuntos e treinar e testar em diferentes combinações
- Criar uma espécie de histórico de performance e fazer uma comparação verificando tendências e tudo mais.
- Utilizar mais de uma métrica, além do MAE
- Um margem para definir se o modelo novo é melhor que o atual para realizar a troca
