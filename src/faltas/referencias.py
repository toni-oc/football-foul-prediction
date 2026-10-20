import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

ESTRATEGIAS = ("media_global_uniforme", "media_global_recencia", "liga_localia_previa", "equipo_rival_previo")


ENTRADAS_REFERENCIAS = (
    "es_local", "liga_historial_ponderado_localia",
    "equipo_historial_ponderado_cometidas", "rival_historial_ponderado_recibidas",
)


def _vector(valores, indice, nombre, no_negativo=True):
    if isinstance(valores, pd.Series) and not valores.index.equals(indice):
        raise ValueError(f"{nombre} está desalineado respecto a X.")
    a = np.asarray(valores, dtype=float)
    if a.ndim != 1 or len(a) != len(indice) or not np.isfinite(a).all() or (no_negativo and (a < 0).any()):
        raise ValueError(f"{nombre} debe ser un vector completo, finito y no negativo.")
    return a


def _validar_X(X):
    if not isinstance(X, pd.DataFrame) or X.empty or not X.index.is_unique or not X.columns.is_unique:
        raise ValueError("Se necesita X no vacía con IDs y columnas únicos.")
    if not set(ENTRADAS_REFERENCIAS).issubset(X.columns):
        raise ValueError("Faltan variables previas para las referencias.")
    if not X["es_local"].isin([0, 1]).all():
        raise ValueError("Localía no válida.")
    for c in ENTRADAS_REFERENCIAS[1:]:
        if not pd.api.types.is_numeric_dtype(X[c].dtype):
            raise ValueError(f"Referencia previa no numérica: {c}")
        v = X[c].to_numpy(dtype=float, na_value=np.nan)
        if np.isinf(v).any() or (v < 0).any():
            raise ValueError("Una media previa es infinita o negativa.")


class ReferenciaFaltas(RegressorMixin, BaseEstimator):
    """Referencias sin acceso al resultado del encuentro solicitado.

    La referencia equipo/rival utiliza media aritmética 50/50, fijada por diseño.
    Una media previa ausente recurre a liga; liga ausente, a localía del fit;
    localía sin ejemplos, a la media global del fit. Cero es un valor real.
    """

    def __init__(self, estrategia="liga_localia_previa"):
        self.estrategia = estrategia

    def fit(self, X, y, sample_weight=None):
        _validar_X(X)
        if self.estrategia not in ESTRATEGIAS:
            raise ValueError("Referencia desconocida.")
        objetivo = _vector(y, X.index, "y")
        if not np.equal(objetivo, np.floor(objetivo)).all():
            raise ValueError("Las etiquetas de faltas deben ser recuentos enteros.")
        pesos = np.ones(len(X)) if sample_weight is None else _vector(sample_weight, X.index, "sample_weight")
        if (pesos <= 0).any():
            raise ValueError("Los pesos deben ser positivos.")
        self.media_uniforme_ = float(objetivo.mean())
        self.media_ponderada_ = float(np.average(objetivo, weights=pesos))
        self.medias_localia_ = {}
        for lado in (0, 1):
            mascara = X["es_local"].eq(lado).to_numpy()
            self.medias_localia_[lado] = (float(np.average(objetivo[mascara], weights=pesos[mascara]))
                                          if mascara.any() else self.media_ponderada_)
        self.n_features_in_ = len(X.columns)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def predecir_con_control(self, X):
        check_is_fitted(self, "media_ponderada_")
        _validar_X(X)
        fallback = X["es_local"].map(self.medias_localia_).astype(float)
        liga = X["liga_historial_ponderado_localia"].fillna(fallback).astype(float)
        fuente_liga = pd.Series(np.where(X["liga_historial_ponderado_localia"].notna(),
                                        "liga_previa_ponderada", "localia_fit"), index=X.index)
        if self.estrategia == "media_global_uniforme":
            pred = pd.Series(self.media_uniforme_, index=X.index)
            fuente_equipo = fuente_rival = pd.Series("media_uniforme_fit", index=X.index)
        elif self.estrategia == "media_global_recencia":
            pred = pd.Series(self.media_ponderada_, index=X.index)
            fuente_equipo = fuente_rival = pd.Series("media_recencia_fit", index=X.index)
        elif self.estrategia == "liga_localia_previa":
            pred = liga
            fuente_equipo = fuente_rival = fuente_liga
        else:
            propio = X["equipo_historial_ponderado_cometidas"]
            rival = X["rival_historial_ponderado_recibidas"]
            pred = (propio.fillna(liga) + rival.fillna(liga)) / 2
            fuente_equipo = fuente_liga.where(propio.isna(), "equipo_previo_ponderado")
            fuente_rival = fuente_liga.where(rival.isna(), "rival_previo_ponderado")
        if not np.isfinite(pred).all() or pred.lt(0).any():
            raise ValueError("Predicción de referencia inválida.")
        return pd.DataFrame({"prediccion": pred, "fuente_equipo": fuente_equipo, "fuente_rival": fuente_rival}, index=X.index)

    def predict(self, X):
        return self.predecir_con_control(X)["prediccion"].to_numpy()
