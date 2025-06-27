# Chained Model - Classificação Hierárquica

Este projeto implementa um modelo de classificação hierárquica encadeada (chained model) usando Random Forest para categorizar dados em múltiplos níveis (lvl1 a lvl4).

## Descrição

O modelo é treinado para prever categorias em 4 níveis sequenciais, onde a saída de cada nível alimenta o próximo. Isso para melhorar a classificação.

## Estrutura do Projeto

- `prever_categoria()` - função para realizar previsões encadeadas nos 4 níveis.  
- `limpar_entrada()` - função para pré-processar e limpar textos de entrada.
- BD_TREINAMENTO.xlsx - base que deve ter os dados de treinamento

### Como usar o projeto

1. Clone o repositório
  
  ``` bash
  git clone https://github.com/thhhhomas/chained_model.git
  cd chained_model
  ```

2. Crie e ative o ambiente virtual

  - Linux/macOS
  
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

  - Windows
  
    ```cmd
    python -m venv venv
    venv\Scripts\activate
    ```

3. Instalar dependências

  ```bash
  pip install -r requirements.txt
  ```
  

## ⚠️ **Observação:**  
> Certifique-se de adaptar o caminho para o arquivo `BD_TREINAMENTO.xlsx` no script.  
> Além disso, revise os nomes das colunas utilizadas na base (`aut_`, `id_`, `op_`, `cen_`, `pro_`, `cod_`, `obs_`, `lvl1`, `lvl2`, `lvl3`, `lvl4`) para garantir que correspondem aos nomes reais presentes no seu arquivo.  
> Qualquer divergência pode causar erros na leitura ou no treinamento do modelo.
