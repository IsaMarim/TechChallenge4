import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from imblearn.over_sampling import SMOTE

class DropFeatures(BaseEstimator, TransformerMixin):
  def __init__(self, feature_to_drop=['Obesity', 'Weight']):
    self.feature_to_drop = feature_to_drop

  def fit(self, df):
    return self

  def transform(self, df):
    if (set(self.feature_to_drop).issubset(df.columns)):
      df.drop(self.feature_to_drop, axis=1, inplace=True)
      return df
    else:
      print('Uma ou mais features não estão no DataFrame 1')
      return df


class MinMax(BaseEstimator, TransformerMixin):
    def __init__(self, min_max_scaler=['Age', 'Height']):
        self.min_max_scaler = min_max_scaler
        self.scaler = MinMaxScaler()

    def fit(self, df):
        if set(self.min_max_scaler).issubset(df.columns):
            self.scaler.fit(df[self.min_max_scaler])
        else:
            print("Uma ou mais features não estão no DataFrame 2")
        return self

    def transform(self, df):
        if set(self.min_max_scaler).issubset(df.columns):
            df[self.min_max_scaler] = self.scaler.transform(df[self.min_max_scaler])
            return df
        else:
            print("Uma ou mais features não estão no DataFrame 2")
            return df
        
class OneHotEncodingNames(BaseEstimator, TransformerMixin):
    def __init__(self, OneHotEncoding=['Gender', 'family_history', 'FAVC', 'SMOKE', 'SCC', 'MTRANS']):
        self.OneHotEncoding = OneHotEncoding
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    def fit(self, df, y=None):
        # Fit apenas nas colunas categóricas
        self.encoder.fit(df[self.OneHotEncoding])
        return self

    def transform(self, df):
        if not set(self.OneHotEncoding).issubset(df.columns):
            print('Uma ou mais features não estão no DataFrame 3')
            return df

        # Transform sem refazer fit
        onehot_array = self.encoder.transform(df[self.OneHotEncoding])
        feature_names = self.encoder.get_feature_names_out(self.OneHotEncoding)

        df_onehot = pd.DataFrame(onehot_array, columns=feature_names, index=df.index)

        # Mantém colunas que não foram one-hot
        outras_features = df.drop(columns=self.OneHotEncoding)

        # Concatena
        df_final = pd.concat([df_onehot, outras_features], axis=1)

        return df_final


class OrdinalFeature(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Definindo explicitamente a ordem das categorias
        self.ordinal_feature = ['FCVC', 'NCP', 'CAEC', 'CH2O', 'FAF', 'TUE', 'CALC']

        self.categories = [
            ['Raramente', 'Às vezes', 'Sempre'],  # FCVC
            ['Uma Refeição', 'Duas Refeições', 'Três Refeições', 'Quatro ou Mais Refeições'],  # NCP
            ['Nunca', 'Às vezes', 'Frequentemente', 'Sempre'],  # CAEC
            ['< 1 L/dia', '1–2 L/dia', '> 2 L/dia'],  # CH2O
            ['Nunca', '~1–2×/sem', '~3–4×/sem', '5×/sem ou mais'],  # FAF
            ['~0–2 h/dia', '~3–5 h/dia', '> 5 h/dia'],  # TUE
            ['Nunca', 'Às vezes', 'Frequentemente', 'Sempre']  # CALC
        ]

        self.encoder = OrdinalEncoder(categories=self.categories)

    def fit(self, df):
        if set(self.ordinal_feature).issubset(df.columns):
            self.encoder.fit(df[self.ordinal_feature])
        else:
            print("Uma ou mais features não estão no DataFrame 4")
        return self

    def transform(self, df):
        if set(self.ordinal_feature).issubset(df.columns):
            df[self.ordinal_feature] = self.encoder.transform(df[self.ordinal_feature])
            return df
        else:
            print("Uma ou mais features não estão no DataFrame 4")
            return df


class Oversample(BaseEstimator, TransformerMixin):
    def __init__(self, target='Obesity_Class'):
        self.target = target
        self.smote = SMOTE(sampling_strategy='minority')

    def fit(self, df, y=None):
        # Fit não deve aplicar SMOTE, apenas retornar self
        self.apply_smote = True
        return self

    def transform(self, df):

        if self.target not in df.columns:
            print(f"A coluna alvo '{self.target}' não está no DataFrame.")
            return df
        
        # Somente aplica SMOTE quando apply_smote=True
        if not self.apply_smote:
            return df

        X = df.drop(columns=[self.target])
        y = df[self.target]

        X_res, y_res = self.smote.fit_resample(X, y)

        df_res = pd.DataFrame(X_res, columns=X.columns)
        df_res[self.target] = y_res

        # Depois do fit(), ao chamar transform() no teste, desativa SMOTE
        self.apply_smote = False

        return df_res