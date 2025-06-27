# Chained Model - Classificação Hierárquica

Este projeto implementa um modelo de classificação hierárquica encadeada (chained model) usando Random Forest para categorizar dados em múltiplos níveis (lvl1 a lvl4).

## Descrição

O modelo é treinado para prever categorias em 4 níveis sequenciais, onde a saída de cada nível alimenta o próximo. Isso para melhorar a classificação.

## Estrutura do Projeto

- `prever_categoria()` - função para realizar previsões encadeadas nos 4 níveis.  
- `limpar_entrada()` - função para pré-processar e limpar textos de entrada.  
- Arquivo Excel `BD_TREINAMENTO.xlsx` com dados para treinamento.
