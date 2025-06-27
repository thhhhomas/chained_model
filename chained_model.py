import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import re

def limpar_entrada(entrada):
    # Substituir caracteres indesejados por espaço
    entrada = re.sub(r"[-/._()$%#&*¨@!+,]", " ", entrada)
    
    # Remover números
    entrada = re.sub(r"\d", "", entrada)
    
    # Remover múltiplos espaços
    entrada = re.sub(r"\s+", " ", entrada)
    
    # Remover espaços extras e converter para minúsculas
    return entrada.strip().lower()

# Novas entradas
def prever_categoria(nova_entrada):
    # Transformar a entrada para o formato numérico
    entrada_vectorizada = vectorizer.transform([nova_entrada])

    # Fazer previsão do lvl1
    y1_pred = modelo_nivel1.predict(entrada_vectorizada)

    # Concatenar a previsão do lvl1 com a entrada
    entrada_nivel2 = np.hstack((entrada_vectorizada.toarray(), y1_pred.reshape(-1, 1)))

    # Fazer previsão do lvl2
    y2_pred = modelo_nivel2.predict(entrada_nivel2)

    # Concatenar a previsão do lvl2 com a entrada
    entrada_nivel3 = np.hstack((entrada_nivel2, y2_pred.reshape(-1, 1)))

    # Fazer previsão do lvl3
    y3_pred = modelo_nivel3.predict(entrada_nivel3)

    # Concatenar a previsão do lvl3 com a entrada
    entrada_nivel4 = np.hstack((entrada_nivel3, y3_pred.reshape(-1, 1)))

    # Fazer previsão do lvl3
    y4_pred = modelo_nivel4.predict(entrada_nivel4)

    # Converter os resultados para os rótulos originais
    categoria_nivel1 = encoder_nivel1.inverse_transform(y1_pred)[0]
    categoria_nivel2 = encoder_nivel2.inverse_transform(y2_pred)[0]
    categoria_nivel3 = encoder_nivel3.inverse_transform(y3_pred)[0]
    categoria_nivel4 = encoder_nivel4.inverse_transform(y4_pred)[0]

    return categoria_nivel1, categoria_nivel2, categoria_nivel3, categoria_nivel4

# Lendo a base de dados
df = pd.read_excel(r"BD_TREINAMENTO.xlsx")

# Concatenando os campos relevantes em um único texto
colunas = ["aut_", "id_", "op_", "cen_", "pro_", "cod_", "obs_"]

df["TEXTO_COMPLETO"] = df[colunas].astype(str).apply(lambda x: " ".join(x), axis=1)
df["TEXTO_COMPLETO"] = df["TEXTO_COMPLETO"].apply(limpar_entrada)

# Definindo as variáveis
X = df["TEXTO_COMPLETO"].astype(str)
y1 = df["lvl1"].astype(str)
y2 = df["lvl2"].astype(str)
y3 = df["lvl3"].astype(str)
y4 = df["lvl4"].astype(str)

# Convertendo as classes para números
encoder_nivel1 = LabelEncoder()
encoder_nivel2 = LabelEncoder()
encoder_nivel3 = LabelEncoder()
encoder_nivel4 = LabelEncoder()

y1_encoded = encoder_nivel1.fit_transform(y1)
y2_encoded = encoder_nivel2.fit_transform(y2)
y3_encoded = encoder_nivel3.fit_transform(y3)
y4_encoded = encoder_nivel4.fit_transform(y4)

# Dividindo entre treino e teste
X_train, X_test, y1_train, y1_test, y2_train, y2_test, y3_train, y3_test, y4_train, y4_test = train_test_split(
    X, y1_encoded, y2_encoded, y3_encoded, y4_encoded, test_size=0.2, random_state=42
)

# Definindo stopwords
stop_words = ["de", "da", "do", "em", "para", "com", "sem", "no", "na",
               "nos", "nas", "ao", "à", "aos", "às", "e", "ou", "a", "o",
             "um", "uma", "uns", "umas", "que", "se", "por", "como",
               "mais", "muito", "tão", "tanto", "tanta", "valor", "lojas", "loja",
                "nan", "p/", "c/", "rj", "sp", "emissao", "nf", "entrada", "diferente",
                "mesma", "razao", "social", "socia", "ent", "orcamento", "orçamento", "despesa",
                "despesas", "ml", "f", "ff", "ct", "pp", "tp","a", "b", "c", "d", "e", "f", "g",
                "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w",
                "x", "y", "z", "mult", "viscose", "uso", "consumo", "cons", "insumo",
                "imp", "conta", "grand", "grande", "pequeno", "pequena", "pequenos", "pequenas", "p/comercializacao",
                "comercializacao", "comercialização", "vinil", "pad", "po", "wl", "box", "bops", "af", "profissional",
                "termica", "termico", "transp", "panetone", "laran", "amare", "rol", "prox.", "prox", "reajuste",
                "janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto", "setembro", "outubro",
                "novembro", "dezembro", "jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez",
                "pedido", "req.", "req", "requisicao", "requisição", "requisiçao", "requisiçao", "requisiçao", "requisiçao",
                "laranja", "amarelo", "vermelho", "verde", "preto", "branco", "rosa", "roxo", "nan", "inox", "inoxidavel"]

# Modelo para o lvl1
vectorizer = CountVectorizer(stop_words=stop_words, lowercase=True, max_features=10000)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

modelo_nivel1 = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
modelo_nivel1.fit(X_train_vectorized, y1_train)

# Previsões do lvl1
y1_pred_train = modelo_nivel1.predict(X_train_vectorized)
y1_pred_test = modelo_nivel1.predict(X_test_vectorized)

# Adicionando a previsão do lvl1 como entrada para o modelo de lvl2
X_train_nivel2 = np.hstack((X_train_vectorized.toarray(), y1_pred_train.reshape(-1, 1)))
X_test_nivel2 = np.hstack((X_test_vectorized.toarray(), y1_pred_test.reshape(-1, 1)))

# Modelo para o lvl2
modelo_nivel2 = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
modelo_nivel2.fit(X_train_nivel2, y2_train)

# Previsões do lvl2
y2_pred_train = modelo_nivel2.predict(X_train_nivel2)
y2_pred_test = modelo_nivel2.predict(X_test_nivel2)

# Adicionando a previsão do lvl2 como entrada para o modelo de lvl3
X_train_nivel3 = np.hstack((X_train_nivel2, y2_pred_train.reshape(-1, 1)))
X_test_nivel3 = np.hstack((X_test_nivel2, y2_pred_test.reshape(-1, 1)))

# Modelo para o lvl3
modelo_nivel3 = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
modelo_nivel3.fit(X_train_nivel3, y3_train)

# Previsões do lvl3
y3_pred_train = modelo_nivel3.predict(X_train_nivel3)
y3_pred_test = modelo_nivel3.predict(X_test_nivel3)

# Adicionando a previsão do lvl3 como entrada para o modelo de lvl4
X_train_nivel4 = np.hstack((X_train_nivel3, y3_pred_train.reshape(-1, 1)))
X_test_nivel4 = np.hstack((X_test_nivel3, y3_pred_test.reshape(-1, 1)))

# Modelo para o lvl4
modelo_nivel4 = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
modelo_nivel4.fit(X_train_nivel4, y4_train)
