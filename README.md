# Tech Challenge 4 – Modelo Preditivo de Obesidade


Isabela Marim Mayerhoffer Pereira - RM 362023

Lucas Constantino Silva - RM 364620

Pedro Bugui Garcia - RM 360783

Sophia Yeshua Senra - RM 362887


## Introdução

A obesidade é uma condição médica caracterizada pelo acúmulo excessivo de gordura corporal, podendo levar a diversas complicações de saúde. Esse problema tem se tornado cada vez mais prevalente em todo o mundo, afetando pessoas de diferentes idades e classes sociais.  
Suas causas são multifatoriais, envolvendo aspectos **genéticos, ambientais e comportamentais**.

Com base na base de dados disponibilizada em **`obesity.csv`**, este projeto propõe o desenvolvimento de um **modelo preditivo** capaz de **auxiliar profissionais da saúde** na **classificação e diagnóstico de obesidade** de forma automatizada, utilizando técnicas de **Machine Learning** e uma interface interativa construída com **Streamlit**.


## Estrutura do Projeto

```
📁 TechChallenge4/
│
├── 📂 bases/
│   ├── Obesity.csv               # Base de dados original fornecida no desafio
│   └── Obesity_tratado.csv       # Base de dados limpa e tratada para uso no modelo
│
├── 📄 app.py                     # Aplicação principal em Streamlit
│                                 Responsável pela interface interativa e predição do nível de obesidade
│
├── 📓 modelo_tc4.ipynb           # Notebook de análise e modelagem
│                                 Contém:
│                                   - Exploração e tratamento dos dados
│                                   - Teste da pipeline
│                                   - Teste de diferentes modelos de Machine Learning
│                                   - Comparação de métricas e escolha do modelo final (Random Forest)
│
├── 📄 modelo_forest.joblib       # Modelo Random Forest treinado e exportado
│                                 Carregado pelo app Streamlit para realizar previsões em tempo real
│
├── 📄 utils.py                   # Arquivo de funções auxiliares e pipeline
│                                 Inclui:
│                                   - Pré-processamento dos dados de entrada
│                                   - Normalização de variáveis quantitativas
│                                   - Codificação de variáveis categóricas
│
├── 📄 requirements.txt           # Lista de dependências e versões utilizadas no projeto
│                                 Permite recriar o ambiente necessário para execução
│
└── 📄 README.md                  # Documentação do projeto

