# Create sample points within the provided AOI rectangle and two depths per location
import csv, random, math, json, os
from datetime import datetime, timedelta
import pandas as pd

# AOI bounds (lon_min, lon_max, lat_min, lat_max) from user's polygon (rectangle)
lon_min = -71.88100053463857
lon_max = -71.87370492611807
lat_min = -16.73354802470063
lat_max = -16.73001355926052

random.seed(42)

# Generate 30 base locations uniformly within AOI
locations = []
for i in range(30):
    lon = random.uniform(lon_min, lon_max)
    lat = random.uniform(lat_min, lat_max)
    locations.append((i+1, lon, lat))

# Expand to 2 depths per location
rows = []
for loc_id, lon, lat in locations:
    for depth in ["0-30", "30-70"]:
        rows.append({
            "id": f"P{loc_id:03d}-{depth.replace('-', '_')}",
            "location_id": f"L{loc_id:03d}",
            "lon": round(lon, 8),
            "lat": round(lat, 8),
            "depth_cm": depth,
            "stratum_hint": "",     # to be filled after stratification in GEE (e.g., SI quantile)
            "planned_date": "",     # YYYY-MM-DD
            "notes": ""
        })

points_csv_path = "./puntos_para_relevar_AOI.csv"
with open(points_csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

# Create a synthetic EC + indices dataset for dry-run testing the pipeline
def clip(val, lo, hi):
    return max(lo, min(hi, val))

synthetic = []
start_date = datetime(2025, 6, 1)
for r in rows:
    # Simulate ECe (dS/m) with a spread from 0–16
    ece = max(0, random.gauss(5.5, 3.0))  # mean 5.5, sd 3.0
    # Simulate indices with expected correlations
    ndvi = clip(0.75 - 0.05*ece + random.gauss(0.0, 0.05), -0.2, 0.9)
    ndmi = clip(0.25 - 0.02*ece + random.gauss(0.0, 0.03), -0.5, 0.5)
    si   = clip(1.2 + 0.25*ece + random.gauss(0.0, 0.2), 0.5, 6.0)
    date = start_date + timedelta(days=random.randint(0, 29))
    synthetic.append({
        **r,
        "date": date.strftime("%Y-%m-%d"),
        "ECe_dSm": round(ece, 3),
        "NDVI": round(ndvi, 3),
        "NDMI": round(ndmi, 3),
        "SI": round(si, 3)
    })

dummy_csv_path = "./dummy_ece_indices.csv"
pd.DataFrame(synthetic).to_csv(dummy_csv_path, index=False)

# Build a ready-to-run Colab notebook
nb_cells = []

def code_cell(src):
    return {
        "cell_type": "code",
        "execution_count": 0,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(True)
    }

def md_cell(src):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src.splitlines(True)
    }

nb_cells.append(md_cell("# Validación de salinidad (ECe) con índices Sentinel-2\n"
                        "\n"
                        "Este cuaderno carga puntos con índices (desde GEE) y mediciones de ECe, entrena modelos,\n"
                        "aplica **validación espacial por bloques**, calcula métricas (R², RMSE, MAE, RPD) y prepara\n"
                        "coeficientes/expresiones para aplicar el modelo en GEE.\n"))

nb_cells.append(code_cell("""
# Si usas Colab, puedes necesitar instalar paquetes. Descomenta si es necesario.
# %pip install -q scikit-learn pandas matplotlib
"""))

nb_cells.append(code_cell(f"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Rutas de ejemplo (sube tus archivos o monta Drive)
indices_path = r"{dummy_csv_path}"  # Reemplaza por 'indices_pts.csv' exportado desde GEE
field_path = r"{dummy_csv_path}"    # Reemplaza por tu 'ece_terreno.csv' real cuando lo tengas

indices_df = pd.read_csv(indices_path)
field_df = pd.read_csv(field_path)

# En tu flujo real, 'indices_df' y 'field_df' serían distintos; aquí usamos el dummy para demo.
# Supón que comparten 'id' o ('lon','lat','date','depth_cm'). Haremos un merge por id para simplicidad.
df = pd.merge(indices_df, field_df[['id','ECe_dSm']], on='id', suffixes=('', '_y'))
df = df.rename(columns={{'ECe_dSm': 'ECe'}})
df = df[['id','lon','lat','depth_cm','date','NDVI','NDMI','SI','ECe']].dropna().copy()

print("Filas:", len(df))
df.head()
"""))

nb_cells.append(code_cell("""
# Asignar bloques espaciales gruesos para validación sin fuga espacial
def make_blocks(lon, lat, size=0.001):  # ~100 m aprox; ajusta según AOI
    bx = np.floor((lon - lon.min()) / size).astype(int)
    by = np.floor((lat - lat.min()) / size).astype(int)
    return (bx * 10_000 + by).astype(int)

df['block_id'] = make_blocks(df['lon'].values, df['lat'].values, size=0.0015)

features = ['NDVI','NDMI','SI']
target = 'ECe'

X = df[features].values
y = df[target].values
groups = df['block_id'].values

# Modelos a probar
models = {
    'Linear': Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())]),
    'Ridge':  Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=1.0))]),
    'RF':     Pipeline([('model', RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1))])
}

def rpd(y_true, y_pred):
    sd = np.std(y_true, ddof=1)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return sd / rmse if rmse > 0 else np.nan

gkf = GroupKFold(n_splits=5)

results = []
preds_store = {{}}
for name, pipe in models.items():
    yhat = cross_val_predict(pipe, X, y, cv=gkf.split(X, y, groups=groups), n_jobs=-1)
    R2 = r2_score(y, yhat)
    RMSE = np.sqrt(mean_squared_error(y, yhat))
    MAE = mean_absolute_error(y, yhat)
    RPD = rpd(y, yhat)
    results.append((name, R2, RMSE, MAE, RPD))
    preds_store[name] = yhat

res_df = pd.DataFrame(results, columns=['Model','R2','RMSE','MAE','RPD']).sort_values('R2', ascending=False)
res_df
"""))

nb_cells.append(code_cell("""
# Gráfico Observado vs Predicho para el mejor modelo
best = res_df.iloc[0]['Model']
yhat = preds_store[best]

plt.figure(figsize=(6,6))
plt.scatter(y, yhat, alpha=0.6)
mn, mx = min(y.min(), yhat.min()), max(y.max(), yhat.max())
plt.plot([mn, mx], [mn, mx])
plt.xlabel('ECe observado (dS/m)')
plt.ylabel('ECe predicho (dS/m)')
plt.title(f'Observed vs Predicted - {best}')
plt.tight_layout()
plt.show()
"""))

nb_cells.append(code_cell("""
# Entrena el mejor modelo en todos los datos y exporta coeficientes si es lineal
best = res_df.iloc[0]['Model']
pipe = models[best]
pipe.fit(X, y)

if best in ('Linear','Ridge'):
    scaler = pipe.named_steps['scaler']
    model = pipe.named_steps['model']
    coefs = dict(zip(['NDVI','NDMI','SI'], model.coef_))
    intercept = model.intercept_
    print("Coeficientes (sobre variables estandarizadas):", coefs)
    print("Intercept:", intercept)

    # Para aplicar en GEE, calcula fórmula explícita en términos originales:
    mu = dict(zip(['NDVI','NDMI','SI'], scaler.mean_))
    sd = dict(zip(['NDVI','NDMI','SI'], scaler.scale_))

    # ECe = b0 + sum(bi * (Xi - mu_i)/sd_i)
    # Reescribe como ECe = A0 + a_NDVI*NDVI + a_NDMI*NDMI + a_SI*SI
    a = {k: (coefs[k]/sd[k]) for k in coefs}
    A0 = intercept - sum((coefs[k]*mu[k]/sd[k]) for k in coefs)
    print("Expresión no estandarizada:")
    print("ECe = {:.6f} + {:.6f}*NDVI + {:.6f}*NDMI + {:.6f}*SI".format(A0, a['NDVI'], a['NDMI'], a['SI']))

else:
    # Para RF, la aplicación por pixeles es más cómoda en Python raster o en GEE con TF/EE (no cubierto aquí).
    if hasattr(pipe.named_steps['model'], 'feature_importances_'):
        fi = dict(zip(['NDVI','NDMI','SI'], pipe.named_steps['model'].feature_importances_))
        print("Importancias (RF):", fi)
"""))

nb_cells.append(code_cell("""
# Clasificación por umbrales de ECe (ajusta a tu cultivo/contexto)
# Ejemplo: <2 (baja), 2-4 (leve), 4-8 (moderada), 8-16 (alta), >16 (extrema)
bins = [-np.inf, 2, 4, 8, 16, np.inf]
labels = ['baja','leve','moderada','alta','extrema']
best = res_df.iloc[0]['Model']
yhat = preds_store[best]
classes = pd.cut(yhat, bins=bins, labels=labels)

pd.DataFrame({'ECe_obs': y, 'ECe_pred': yhat, 'class_pred': classes}).head()
"""))

nb = {
    "cells": nb_cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

nb_path = "./validacion_salinidad_colab.ipynb"
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=2)

# Display small previews
points_df = pd.read_csv(points_csv_path)
dummy_df = pd.read_csv(dummy_csv_path)


(points_csv_path, dummy_csv_path, nb_path)
