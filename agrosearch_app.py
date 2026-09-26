import re
import unicodedata

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


st.title("AgroSearch")

with st.form("form_documento"):
    doc = st.selectbox("Escolha um documento", list(DOCUMENTOS.keys()))
    st.form_submit_button("Processar Documento")

texto = DOCUMENTOS[doc]
tokens = tokenizar(texto)

st.write("Documento Selecionado: {doc}")
st.write("Texto Original:", texto)
st.write("1. Tokenização:", tokens)
st.write("2. Normalização:", normalizar(tokens))