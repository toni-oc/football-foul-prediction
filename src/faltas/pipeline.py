"""Preparación aprendida exclusivamente en fit; esquema previo explícito y clonable."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted


AUSENTE = "__AUSENTE__"
NUEVA = "__NUEVA__"


class PreparadorVariables(TransformerMixin, BaseEstimator):
    """Medianas, indicadores de todos los numéricos y one-hot con dos niveles reservados.

    Las listas de columnas son un contrato de diseño, no una selección estadística.
    Las medianas, columnas vacías, categorías y escala se aprenden de nuevo en cada fit.
    La salida densa conserva IDs; su vocabulario solo es estable dentro de ese ajuste.
    """

    def __init__(self, numericas, categoricas, escalar=False):
        self.numericas = numericas
        self.categoricas = categoricas
        self.escalar = escalar

    def _normalizar(self, X):
        if not isinstance(X, pd.DataFrame) or X.empty or not X.index.is_unique:
            raise ValueError("X debe ser un DataFrame no vacío con IDs únicos.")
        esperadas = list(self.numericas) + list(self.categoricas)
        if (len(set(esperadas)) != len(esperadas) or not esperadas
                or not X.columns.is_unique or set(X.columns) != set(esperadas)):
            raise ValueError("El esquema de X difiere de las variables previas declaradas.")
        columnas = {}
        for c in self.numericas:
            if not pd.api.types.is_numeric_dtype(X[c].dtype):
                raise ValueError(f"La columna numérica {c} no tiene tipo numérico.")
            columnas[c] = X[c].to_numpy(dtype=float, na_value=np.nan)
            if np.isinf(columnas[c]).any():
                raise ValueError(f"La columna {c} contiene infinitos.")
        for c in self.categoricas:
            datos = X[c]
            conocidos = datos.dropna()
            if not conocidos.map(lambda v: isinstance(v, str)).all():
                raise ValueError(f"La categoría {c} debe ser texto o ausente.")
            if conocidos.isin([AUSENTE, NUEVA]).any():
                raise ValueError("Los nombres de categorías reservadas no pueden aparecer en la entrada.")
            columnas[c] = datos.astype(object).where(datos.notna(), AUSENTE)
        return pd.DataFrame(columnas, index=X.index)

    def fit(self, X, y=None):
        # y se admite para componer Pipeline, pero no se consulta.
        normal = self._normalizar(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        self.columnas_vacias_ = tuple(c for c in self.numericas if normal[c].isna().all())
        self.numericas_con_datos_ = tuple(c for c in self.numericas if c not in self.columnas_vacias_)
        self.vocabulario_ = {
            c: tuple(sorted(normal.loc[normal[c].ne(AUSENTE), c].unique()))
            for c in self.categoricas
        }
        ramas = []
        if self.numericas_con_datos_:
            numeros = Pipeline([
                ("imputar", SimpleImputer(strategy="median")),
                ("escalar", StandardScaler() if self.escalar else "passthrough"),
            ])
            ramas.append(("numericas", numeros, list(self.numericas_con_datos_)))
        if self.numericas:
            # También cubre una ausencia que aparezca por primera vez al predecir.
            ramas.append(("ausencias", MissingIndicator(features="all"), list(self.numericas)))
        if self.categoricas:
            categorias = [list(self.vocabulario_[c]) + [AUSENTE, NUEVA] for c in self.categoricas]
            ramas.append(("categorias", OneHotEncoder(
                categories=categorias, handle_unknown="error", sparse_output=False, dtype=float,
            ), list(self.categoricas)))
        self.columnas_ = ColumnTransformer(ramas, remainder="drop", sparse_threshold=0,
                                          verbose_feature_names_out=True)
        self.columnas_.fit(normal)
        self.nombres_salida_ = self.columnas_.get_feature_names_out()
        if len(set(self.nombres_salida_)) != len(self.nombres_salida_):
            raise ValueError("Hay nombres de variables transformadas repetidos.")
        return self

    def transform(self, X):
        check_is_fitted(self, "columnas_")
        normal = self._normalizar(X)
        for c in self.categoricas:
            presente = normal[c].ne(AUSENTE)
            normal.loc[presente & ~normal[c].isin(self.vocabulario_[c]), c] = NUEVA
        valores = np.asarray(self.columnas_.transform(normal), dtype=float)
        if not np.isfinite(valores).all():
            raise ValueError("La preparación no ha producido valores finitos.")
        return pd.DataFrame(valores, index=X.index, columns=self.nombres_salida_)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "columnas_")
        if input_features is not None and list(input_features) != list(self.feature_names_in_):
            raise ValueError("Los nombres solicitados difieren del esquema ajustado.")
        return self.nombres_salida_.copy()

    def diagnostico(self):
        """Decisiones de este entrenamiento; no inspecciona validación ni test."""
        check_is_fitted(self, "columnas_")
        medianas = {}
        if self.numericas_con_datos_:
            estadisticas = self.columnas_.named_transformers_["numericas"].named_steps["imputar"].statistics_
            medianas = dict(zip(self.numericas_con_datos_, estadisticas))
        filas = []
        for c in self.numericas:
            vacia = c in self.columnas_vacias_
            filas.append({"campo": c, "tipo": "numerica", "decision":
                          "Valor excluido; solo indicador de ausencia" if vacia else "Mediana del entrenamiento + indicador",
                          "mediana_train": medianas.get(c, np.nan), "categorias_train": None})
        for c in self.categoricas:
            filas.append({"campo": c, "tipo": "categorica", "decision": "One-hot; ausencia y nueva explícitas",
                          "mediana_train": np.nan, "categorias_train": " | ".join(self.vocabulario_[c])})
        return pd.DataFrame(filas)


def construir_preparacion(catalogo, escalar=False):
    if (catalogo.empty or not catalogo["campo"].is_unique
            or not catalogo["tipo"].isin(["numerica", "categorica"]).all()):
        raise ValueError("Catálogo de variables previas inválido.")
    return PreparadorVariables(
        numericas=tuple(catalogo.loc[catalogo["tipo"].eq("numerica"), "campo"]),
        categoricas=tuple(catalogo.loc[catalogo["tipo"].eq("categorica"), "campo"]),
        escalar=escalar,
    )


def construir_pipeline(estimador, catalogo, escalar=False):
    """Objeto nuevo sin ajustar. En fit, pesos mediante modelo__sample_weight.

    Las medianas son ordinarias. El peso temporal corresponde a la pérdida del
    estimador; no se confunde con una mediana ponderada ni se aplica a validación.
    """
    return Pipeline([
        ("preparar", construir_preparacion(catalogo, escalar)),
        ("modelo", clone(estimador)),
    ])
