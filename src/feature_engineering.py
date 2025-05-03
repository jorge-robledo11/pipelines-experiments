import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import clone

from feature_engine.encoding import (
    OneHotEncoder, OrdinalEncoder, MeanEncoder,
    CountFrequencyEncoder, WoEEncoder, DecisionTreeEncoder
)
from feature_engine.discretisation import (
    EqualFrequencyDiscretiser, EqualWidthDiscretiser,
    GeometricWidthDiscretiser, DecisionTreeDiscretiser
)


class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    Selecciona un subset de columnas y devuelve un array numpy float.
    """
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns].values.astype(float)


class DropTransformer(BaseEstimator, TransformerMixin):
    """
    Devuelve siempre un array de shape (n_samples, 0).
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.empty((len(X), 0), dtype=float)


class CategoricalTransformer(BaseEstimator, TransformerMixin):
    """
    Castea columnas a 'category' y aplica un encoder categórico.
    """
    def __init__(self, variables, encoding, drop_last=True, encoding_method="ordered"):
        self.variables = variables
        self.encoding = encoding
        self.drop_last = drop_last
        self.encoding_method = encoding_method

        if encoding == "onehot":
            self.enc = OneHotEncoder(variables=variables, drop_last=drop_last)
        elif encoding == "ordinal":
            self.enc = OrdinalEncoder(encoding_method=encoding_method, variables=variables)
        elif encoding == "mean":
            self.enc = MeanEncoder(variables=variables)
        elif encoding == "countfreq":
            self.enc = CountFrequencyEncoder(variables=variables, encoding_method=encoding_method)
        elif encoding == "woe":
            self.enc = WoEEncoder(variables=variables, fill_value=0)
        elif encoding == "dtree":
            self.enc = DecisionTreeEncoder(
                variables=variables,
                encoding_method=encoding_method,
                regression=False
            )
        else:
            raise ValueError(f"Encoding inválido: {encoding!r}")

    def fit(self, X, y=None):
        Xc = X[self.variables].copy()
        for col in self.variables:
            if not isinstance(Xc[col].dtype, pd.CategoricalDtype):
                Xc[col] = Xc[col].astype("category")
        self.enc.fit(Xc, y)
        return self

    def transform(self, X):
        Xc = X[self.variables].copy()
        for col in self.variables:
            if not isinstance(Xc[col].dtype, pd.CategoricalDtype):
                Xc[col] = Xc[col].astype("category")
        Xe = self.enc.transform(Xc)
        return Xe.values.astype(float)


class DiscretizeAndEncode(BaseEstimator, TransformerMixin):
    """
    Discretiza variables continuas y luego aplica un encoder.
    """
    def __init__(
        self,
        variables,
        discretizer,
        q=5,
        max_depth=None,
        encoding="ordinal",
        drop_last=True,
        encoding_method="ordered"
    ):
        self.variables = variables
        self.discretizer_name = discretizer
        self.q = q
        self.max_depth = max_depth
        self.encoding = encoding
        self.drop_last = drop_last
        self.encoding_method = encoding_method

        # discretizadores con return_object=True donde aplica
        if discretizer == "equal_freq":
            self.disc = EqualFrequencyDiscretiser(q=q, variables=variables, return_object=True)
        elif discretizer == "equal_width":
            self.disc = EqualWidthDiscretiser(bins=q, variables=variables, return_object=True)
        elif discretizer == "geometric":
            self.disc = GeometricWidthDiscretiser(bins=q, variables=variables, return_object=True)
        elif discretizer == "dtree":
            param_grid = {"max_depth": [max_depth]} if max_depth is not None else None
            self.disc = DecisionTreeDiscretiser(
                variables=variables,
                param_grid=param_grid,
                regression=False,
                random_state=42
            )
        else:
            raise ValueError(f"Discretizer inválido: {discretizer!r}")

        # selecciona encoder
        if encoding == "onehot":
            self.enc = OneHotEncoder(variables=variables, drop_last=drop_last)
        elif encoding == "ordinal":
            self.enc = OrdinalEncoder(encoding_method=encoding_method, variables=variables)
        elif encoding == "mean":
            self.enc = MeanEncoder(variables=variables)
        elif encoding == "countfreq":
            self.enc = CountFrequencyEncoder(variables=variables, encoding_method=encoding_method)
        elif encoding == "woe":
            self.enc = WoEEncoder(variables=variables, fill_value=0)
        elif encoding == "dtree":
            self.enc = DecisionTreeEncoder(
                variables=variables,
                encoding_method=encoding_method,
                regression=False
            )
        else:
            raise ValueError(f"Encoding inválido: {encoding!r}")

    def fit(self, X, y=None):
        Xnum = X[self.variables].apply(pd.to_numeric, errors="raise")
        self.disc.fit(Xnum, y)
        Xb = self.disc.transform(Xnum)
        for col in Xb.columns:
            Xb[col] = Xb[col].astype("category")
        self.enc.fit(Xb, y)
        return self

    def transform(self, X):
        Xnum = X[self.variables].apply(pd.to_numeric, errors="raise")
        Xb = self.disc.transform(Xnum)
        for col in Xb.columns:
            Xb[col] = Xb[col].astype("category")
        Xe = self.enc.transform(Xb)
        return Xe.values.astype(float)


def build_cat_pipeline(trial, categorical_cols):
    """
    Construye transformador para variables categóricas.
    """
    choice = trial.suggest_categorical(
        'cat_enc',
        ['none','onehot','ordinal','mean','countfreq','woe','dtree']
    )
    if choice == 'none':
        return DropTransformer()
    encoding_method = None
    if choice in ('ordinal','dtree'):
        encoding_method = trial.suggest_categorical(f'{choice}_method', ['ordered','arbitrary'])
    if choice == 'countfreq':
        encoding_method = trial.suggest_categorical('countfreq_method', ['count','frequency'])
    return CategoricalTransformer(
        variables=categorical_cols,
        encoding=choice,
        drop_last=True,
        encoding_method=encoding_method or 'ordered'
    )


def build_disc_pipeline(trial, discrete_cols):
    """
    Construye transformador para variables discretas (ya category).
    """
    choice = trial.suggest_categorical(
        'disc_enc',
        ['none','onehot','ordinal','mean','countfreq','woe','dtree']
    )
    if choice == 'none':
        return ColumnSelector(discrete_cols)
    encoding_method = None
    if choice in ('ordinal','dtree'):
        encoding_method = trial.suggest_categorical(f'disc_{choice}_method', ['ordered','arbitrary'])
    if choice == 'countfreq':
        encoding_method = trial.suggest_categorical('disc_countfreq_method', ['count','frequency'])
    return CategoricalTransformer(
        variables=discrete_cols,
        encoding=choice,
        drop_last=True,
        encoding_method=encoding_method or 'ordered'
    )


def build_cont_pipeline(trial, continuous_cols):
    """
    Construye transformador para variables continuas.
    """
    choice = trial.suggest_categorical(
        'cont_tr',
        ['none','equal_freq','equal_width','geometric','dtree',
         'equal_freq_encode','equal_width_encode','geometric_encode','dtree_encode']
    )
    if choice == 'none':
        return ColumnSelector(continuous_cols)
    q = trial.suggest_int('cont_q', 2, 10)
    max_depth = None
    if 'dtree' in choice:
        max_depth = trial.suggest_int('cont_max_depth', 2, 5)
    # discretizadores puros
    if choice in ['equal_freq','equal_width','geometric','dtree']:
        return DiscretizeAndEncode(
            variables=continuous_cols,
            discretizer=choice,
            q=q,
            max_depth=max_depth,
            encoding='ordinal',
            drop_last=True,
            encoding_method='ordered'
        )
    # discretizador + encode
    enc_opt = trial.suggest_categorical(
        'cont_enc',
        ['onehot','ordinal','mean','countfreq','woe','dtree']
    )
    encoding_method = None
    if enc_opt in ('ordinal','dtree'):
        encoding_method = trial.suggest_categorical(f'cont_{enc_opt}_method', ['ordered','arbitrary'])
    if enc_opt == 'countfreq':
        encoding_method = trial.suggest_categorical('cont_countfreq_method', ['count','frequency'])
    disc_name = choice.replace('_encode','')
    return DiscretizeAndEncode(
        variables=continuous_cols,
        discretizer=disc_name,
        q=q,
        max_depth=max_depth,
        encoding=enc_opt,
        drop_last=True,
        encoding_method=encoding_method or 'ordered'
    )


class FeatureEngineeringPipeline:
    """
    Clase principal que construye la pipeline completa de preprocesamiento.
    """
    def __init__(
        self,
        categorical_cols,
        discrete_cols,
        continuous_cols,
        model
    ):
        self.cat_cols = categorical_cols
        self.disc_cols = discrete_cols
        self.cont_cols = continuous_cols
        self.model = model

    def build_pipeline(self, trial):
        cat_pipe = build_cat_pipeline(trial, self.cat_cols)
        disc_pipe = build_disc_pipeline(trial, self.disc_cols)
        cont_pipe = build_cont_pipeline(trial, self.cont_cols)

        preproc = FeatureUnion([
            ('cat', cat_pipe),
            ('disc', disc_pipe),
            ('cont', cont_pipe)
        ])
        
        clf = clone(self.model)

        return Pipeline([
            ('pre', preproc),
            ('clf', clf)
        ])
