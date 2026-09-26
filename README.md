# AgroSearch — Motor de Busca Inteligente

Laboratório Prático 04 — Desafio Integrador
Disciplina: Tendências em Ciência da Computação — UNIPÊ
Professor: Me. Ricardo Roberto de Lima
Aluna: [Seu nome completo] — RGM [seu RGM]

## Sobre o projeto
Protótipo de motor de busca textual para os manuais técnicos da AgroTech Solutions.
O técnico digita uma consulta e o sistema devolve os documentos ranqueados por relevância.

Seguindo a restrição do enunciado, **nenhuma biblioteca de alto nível** (scikit-learn,
TfidfVectorizer etc.) foi usada: pré-processamento, stemmer, índice invertido, TF-IDF
e similaridade de cosseno foram implementados do zero em Python.

## Funcionalidades
- **Fase 1 — Pré-processamento:** tokenização, normalização (minúsculas e sem acentos),
  remoção de stopwords e stemming, com checkboxes para ligar/desligar stopwords e stemming.
- **Fase 2 — Índice invertido:** termo → documentos, exibido com `st.json` e `st.dataframe`.
- **Fase 3 — Busca TF-IDF:** cálculo de TF, IDF e TF-IDF, ranking pelo TF-IDF acumulado
  com o documento vencedor destacado.
- **Bônus:** similaridade de cosseno entre a consulta e cada documento.

## Como executar
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run agrosearch_app.py
```