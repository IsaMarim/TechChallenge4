import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from utils import DropFeatures, MinMax, OneHotEncodingNames, OrdinalFeature, Oversample
import joblib
from joblib import load

# Import da base já tratada
dados = pd.read_csv('bases/Obesity_tratado.csv')

# Título
st.write("# Tech Challenge 4")

st.markdown("<br>", unsafe_allow_html=True)
st.write("Isabela Marim Mayerhoffer Pereira - RM 362023")
st.write("Lucas Constantino Silva - RM 364620")
st.write("Pedro Bugui Garcia - RM 360783")
st.write("Sophia Yeshua Senra - RM 362887")

st.markdown("<br>", unsafe_allow_html=True)
st.write("## Modelo Preditivo de Obesidade")
st.markdown("<br>", unsafe_allow_html=True)


# Formulário
st.write('### Dados Pessoais')

st.markdown("<br>", unsafe_allow_html=True)
input_genero = st.selectbox("Qual o seu gênero?", dados['Gender'].unique())

st.markdown("<br>", unsafe_allow_html=True)
input_idade = int(st.slider('Selecione sua idade', 14, 61))

st.markdown("<br>", unsafe_allow_html=True)
input_altura = float(st.slider('Selecione sua altura', 1.45, 1.98))

st.markdown("<br>", unsafe_allow_html=True)
input_historico_familiar = st.radio('Na sua família existe histórico de excesso de peso?', ['Não', 'Sim'])


st.markdown("<br>", unsafe_allow_html=True)
st.write('### Hábitos Alimentares')

st.markdown("<br>", unsafe_allow_html=True)
input_FAVC = st.radio('Você costuma consumir alimentos muito calóricos frequentemente?', ['Não', 'Sim'])

st.markdown("<br>", unsafe_allow_html=True)
input_FCVC= st.selectbox('Com que frequência você consome vegetais nas refeições?', ['Raramente', 'Às vezes', 'Sempre'])

st.markdown("<br>", unsafe_allow_html=True)
input_NCP = st.selectbox('Quantas refeições você costuma consumir por dia?', ['Uma Refeição', 'Duas Refeições', 'Três Refeições', 'Quatro ou Mais Refeições'])

st.markdown("<br>", unsafe_allow_html=True)
input_CAEC = st.selectbox('Com que frequência você consome lanches entre refeições principais?', ['Nunca', 'Às vezes', 'Frequentemente', 'Sempre'])

st.markdown("<br>", unsafe_allow_html=True)
input_CH2O = st.selectbox('Quantos litros de água você costuma consumir por dia?', ['< 1 L/dia', '1–2 L/dia', '> 2 L/dia'])


st.markdown("<br>", unsafe_allow_html=True)
st.write('### Estilo de Vida')

st.markdown("<br>", unsafe_allow_html=True)
input_SMOKE = st.radio('Você fuma?', ['Não', 'Sim'])

st.markdown("<br>", unsafe_allow_html=True)
input_SCC = st.radio('Você monitora sua ingestão calórica diária?', ['Não', 'Sim'])

st.markdown("<br>", unsafe_allow_html=True)
input_FAF = st.selectbox('Com que frequência você pratica atividades físicas na semana?', ['Nunca', '~1–2×/sem', '~3–4×/sem', '5×/sem ou mais'])

st.markdown("<br>", unsafe_allow_html=True)
input_TUE = st.selectbox('Quantas horas por dia você utiliza dispositivos eletrônicos?', ['~0–2 h/dia', '~3–5 h/dia', '> 5 h/dia'])

st.markdown("<br>", unsafe_allow_html=True)
input_CALC = st.selectbox('Com que frequência você consome bebidas alcoólicas?', ['Nunca', 'Às vezes', 'Frequentemente', 'Sempre'])

st.markdown("<br>", unsafe_allow_html=True)
input_MTRANS = st.selectbox('Qual o seu meio de transporte habitual?', dados['MTRANS'].unique())

st.markdown("<br>", unsafe_allow_html=True)


# Lista de todas as variáveis: 
novo_registro = [input_genero,
                    input_idade,
                    input_altura,
                    0,
                    input_historico_familiar,
                    input_FAVC,
                    input_FCVC,
                    input_NCP,
                    input_CAEC,
                    input_SMOKE,
                    input_CH2O,
                    input_SCC,
                    input_FAF,
                    input_TUE,
                    input_CALC,
                    input_MTRANS,
                    'Abaixo do peso', # Obesity
                    "Peso normal" # target - Obesity_class
                    ]

# Separando os dados em treino e teste
def data_split(df, coluna_target):

    # Separa X e y
    X = df.drop(columns=[coluna_target])
    y = df[coluna_target]

    # Split dos dados com estratificação
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Junta para aplicar passos da pipeline
    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)

    return df_train, df_test

df_train, df_test = data_split(dados, "Obesity_Class")


# Criando novo registro
registro = pd.DataFrame([novo_registro],columns=df_test.columns)

# Concatenando novo registro ao dataframe dos dados de teste
df_test = pd.concat([df_test,registro],ignore_index=True)


# Função para aplicar pipeline
def preparar_dados(df_train, df_test, target='Obesity_Class'):

    # Pipeline
    pipe = Pipeline([
        ('drop_old_target', DropFeatures()),  # ✅ remove a coluna antiga "Obesity"
        ('OneHotEncoding', OneHotEncodingNames()),
        ('ordinal_feature', OrdinalFeature()),
        ('min_max_scaler', MinMax()),
        ('oversample', Oversample(target=target))  # ✅ SMOTE apenas no treino
    ])

    # Fit + transform apenas no treino
    df_train_transformed = pipe.fit_transform(df_train)

    # Transform apenas no teste — SEM SMOTE
    df_test_transformed = pipe.transform(df_test)

    # Separa X e y novamente
    X_train_final = df_train_transformed.drop(columns=[target])
    y_train_final = df_train_transformed[target]

    X_test_final = df_test_transformed.drop(columns=[target])
    y_test_final = df_test_transformed[target]

    return X_train_final, X_test_final, y_train_final, y_test_final, pipe


X_treino, X_teste, y_treino, y_teste, pipeline = preparar_dados(df_train, df_test)


# Rodar modelo ao apertar o botão de enviar
if st.button('Enviar Formulário'):
    model = joblib.load('modelo_forest.joblib')
    final_pred = model.predict(X_teste)
    st.markdown("<br>", unsafe_allow_html=True)
    st.write("#### Resultado da Previsão: ", final_pred[-1])
