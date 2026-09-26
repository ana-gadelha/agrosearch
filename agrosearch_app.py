import re
import unicodedata
import pandas as pd
import math
import streamlit as st


DOCUMENTOS = {
    "Doc 1": "A soja requer irrigação constante durante o período de floração para garantir a produtividade.",
    "Doc 2": "O controle biológico de lagartas na soja pode ser feito com a vespa Trichogramma.",
    "Doc 3": "A adubação verde com leguminosas melhora o nitrogênio no solo para o milho.",
    "Doc 4": "Lagartas desfolhadoras causam grande prejuízo na cultura da soja e do algodão.",
    "Doc 5": "A irrigação por gotejamento economiza água e é ideal para o cultivo orgânico.",
}

# Primeira etapa: TOKENIZAÇÃO
def tokenizar(texto):
    return re.findall(r"[0-9A-Za-zÀ-ÿ]+", texto)

# Segunda etapa: NORMALIZAÇÃO
def normalizar(tokens):
    resultado = []
    for token in tokens:
        token = token.lower()
        token = unicodedata.normalize("NFD", token)
        token = "".join(c for c in token if unicodedata.category(c) != "Mn")
        resultado.append(token)
    return resultado

STOPWORDS = {
    "a", "o", "as", "os", "um", "uma", "de", "da", "do", "das", "dos",
    "e", "em", "no", "na", "nos", "nas", "por", "para", "com", "sem",
    "ao", "aos", "que", "se", "ou", "mas", "como", "mais", "pode",
    "ser", "feito", "durante", "e", "sao", "foi", "tem",
}

def remover_stopwords(tokens):
    return [token for token in tokens if token not in STOPWORDS]    

SUFIXOS = [
    ("mente", 4), ("amento", 3), ("imento", 3), ("acao", 3), ("icao", 3),
    ("idade", 4), ("adoras", 3), ("adora", 3), ("ador", 3), ("ismo", 3),
    ("avel", 3), ("ivel", 3), ("oso", 3), ("osa", 3), ("ico", 3), ("ica", 3),
    ("ante", 3), ("ente", 3), ("ura", 3), ("agem", 3),
    ("ando", 2), ("endo", 2), ("indo", 2), ("ado", 2), ("ada", 2), ("ido", 2),
    ("am", 2), ("ar", 2), ("er", 2), ("ir", 2),
    ("as", 3), ("es", 3), ("os", 3), ("s", 3),
    ("a", 3), ("e", 3), ("o", 3),
]

def stem(palavra):
    for sufixo, minimo in SUFIXOS:
        if palavra.endswith(sufixo) and len(palavra) - len(sufixo) >= minimo:
            return palavra[: -len(sufixo)]
    return palavra

def aplicar_stemming(tokens):
    return [stem(token) for token in tokens]


#Pipeline completo (juntas as quatro etapas numa única função)
def preprocessar(texto, usar_stopwords, usar_stemming):
    tokens = normalizar(tokenizar(texto))
    if usar_stopwords:
        tokens = remover_stopwords(tokens)
    if usar_stemming:
        tokens = aplicar_stemming(tokens)
    return tokens

def construir_indice_invertido(docs_tokens):
    indice = {}
    for doc_id, tokens in docs_tokens.items():
        for termo in tokens:
            if termo not in indice:
                indice[termo] = []
            if doc_id not in indice[termo]:
                indice[termo].append(doc_id)
    return dict(sorted(indice.items()))

# Terceira Fase
def calcular_tf(tokens):
    tf = {}
    total = len(tokens)
    for termo in tokens:
        tf[termo] = tf.get(termo, 0) + 1
    for termo in tf:
        tf[termo] = tf[termo] / total
    return tf

def calcular_idf(indice, n_docs):
    idf = {}
    for termo, docs in indice.items():
        idf[termo] = math.log10(n_docs / len(docs))
    return idf

def vetor_tfidf(tokens, idf):
    tf = calcular_tf(tokens)
    return {termo: tf[termo] * idf.get(termo, 0) for termo in tf}

# Similaridade de cosseno entre dois vetores
def cosseno(v1, v2):
    produto = sum(v1[t] * v2.get(t, 0) for t in v1)
    norma1 = math.sqrt(sum(x * x for x in v1.values()))
    norma2 = math.sqrt(sum(x * x for x in v2.values()))
    if norma1 == 0 or norma2 == 0:
        return 0.0
    return produto / (norma1 * norma2)


st.title("AgroSearch")
with st.sidebar:
    st.header("Pré-processamento")
    usar_stopwords = st.checkbox ("Remover stopwords", value=True)
    usar_stemming = st.checkbox ("Aplicar Stemming", value=True)

with st.form("form_documento"):
    doc = st.selectbox("Escolha um documento", list(DOCUMENTOS.keys()))
    st.form_submit_button("Processar Documento")

texto = DOCUMENTOS[doc]
tokens = tokenizar(texto)
normalizados = normalizar(tokens)
sem_stopwords = remover_stopwords(normalizados) if usar_stopwords else normalizados
com_stemming = aplicar_stemming(sem_stopwords) if usar_stemming else sem_stopwords

st.write(f"Documento Selecionado: {doc}")
st.write("Texto Original:", texto)
st.write("1. Tokenização:", tokens)
st.write("2. Normalização:", normalizados)
st.write("3. Sem stopwords", sem_stopwords)
st.write("4. Stemming", com_stemming)


vocabulario = set()
for texto_doc in DOCUMENTOS.values():
    vocabulario.update(preprocessar(texto_doc, usar_stopwords, usar_stemming))

st.metric("Tamanho do vocabulário", len(vocabulario))
st.write(sorted(vocabulario))

st.divider()
st.header("Segunda fase: Índice Invertido")

docs_tokens = {}
for doc_id, texto_doc in DOCUMENTOS.items():
    docs_tokens[doc_id] = preprocessar(texto_doc, usar_stopwords, usar_stemming)

indice = construir_indice_invertido(docs_tokens)

st.subheader("Índice invertido (st.json)")
st.json(indice)

st.subheader("Índice invertido em tabela (st.dataframe)")
tabela_indice = pd.DataFrame({
    "Termo": list(indice.keys()),
    "Documentos": [", ".join(docs) for docs in indice.values()],
    "df (nº de docs)": [len(docs) for docs in indice.values()],
})
st.dataframe(tabela_indice, hide_index=True)

st.divider()
st.header("Fase 3 — Busca e Ranqueamento TF-IDF")
st.latex(r"TF(t,d)=\frac{f(t,d)}{|d|} \qquad IDF(t)=\log_{10}\frac{N}{df(t)} \qquad TF\text{-}IDF = TF \times IDF")

with st.form("form_busca"):
    consulta = st.text_input("Digite sua consulta", value="lagartas na soja")
    st.form_submit_button("🔍 Buscar")

termos_consulta = preprocessar(consulta, usar_stopwords, usar_stemming)
st.write("**Consulta pré-processada:**", termos_consulta)

idf = calcular_idf(indice, len(docs_tokens))
termos_validos = [t for t in dict.fromkeys(termos_consulta) if t in indice]

if not termos_validos:
    st.warning("Nenhum termo da consulta aparece nos documentos. Tente outras palavras.")
else:
    detalhe = []
    for doc_id, toks in docs_tokens.items():
        tf = calcular_tf(toks)
        for termo in termos_validos:
            detalhe.append({
                "Documento": doc_id,
                "Termo": termo,
                "TF": round(tf.get(termo, 0), 4),
                "IDF": round(idf[termo], 4),
                "TF-IDF": round(tf.get(termo, 0) * idf[termo], 4),
            })
    detalhe = pd.DataFrame(detalhe)

    # Ranking: soma do TF-IDF dos termos da consulta (TF-IDF acumulado)
    ranking = detalhe.groupby("Documento", as_index=False)["TF-IDF"].sum()
    ranking = ranking.rename(columns={"TF-IDF": "TF-IDF acumulado"})

    # Similaridade de cosseno entre consulta e documento
    vetor_consulta = vetor_tfidf(termos_consulta, idf)
    ranking["Cosseno (bônus)"] = [
        round(cosseno(vetor_consulta, vetor_tfidf(docs_tokens[d], idf)), 4) for d in ranking["Documento"]
    ]
    ranking["Texto"] = [DOCUMENTOS[d] for d in ranking["Documento"]]
    ranking = ranking.sort_values(["TF-IDF acumulado", "Cosseno (bônus)"], ascending=False).reset_index(drop=True)
    ranking["TF-IDF acumulado"] = ranking["TF-IDF acumulado"].round(4)

    vencedor = ranking.iloc[0]
    st.success(f"🏆 Documento vencedor: {vencedor['Documento']} (TF-IDF acumulado = {vencedor['TF-IDF acumulado']})")

    def destacar_vencedor(linha):
        cor = "background-color: rgba(46, 160, 67, 0.35)" if linha.name == 0 else ""
        return [cor] * len(linha)

    st.subheader("Ranking (maior → menor TF-IDF acumulado)")
    st.dataframe(ranking.style.apply(destacar_vencedor, axis=1), hide_index=True)

    st.subheader("Cálculo termo a termo")
    st.dataframe(detalhe, hide_index=True)