import re
import unicodedata
import pandas as pd
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
