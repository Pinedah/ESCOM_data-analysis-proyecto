import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from streamlit_folium import st_folium
import folium

# =====================
# CONFIGURACIÓN GENERAL
# =====================
st.set_page_config(
    page_title="Dashboard Incendios Forestales",
    layout="wide"
)

st.title("Dashboard Interactivo de Incendios Forestales")
st.markdown("Análisis exploratorio, operativo y estratégico")

# =====================
# CARGA DE DATOS
# =====================
@st.cache_data
def load_data():
    return pd.read_csv("dataset_limpio.csv")

df = load_data()

# =====================
# SIDEBAR - FILTROS
# =====================
st.sidebar.header("Filtros")
st.sidebar.markdown("*Deja vacío para seleccionar TODOS*")

years = st.sidebar.slider(
    "Año",
    int(df.anio.min()),
    int(df.anio.max()),
    (int(df.anio.min()), int(df.anio.max()))
)

estado = st.sidebar.multiselect(
    "Estado",
    options=sorted(df.Estado.unique()),
    default=None,
    placeholder="TODOS"
)

vegetacion = st.sidebar.multiselect(
    "Tipo de Vegetación",
    options=sorted(df.Tipo_Vegetacion.unique()),
    default=None,
    placeholder="TODOS"
)

causa = st.sidebar.multiselect(
    "Causa",
    options=sorted(df.Causa.unique()) if 'Causa' in df.columns else [],
    default=None,
    placeholder="TODOS"
)

mes = st.sidebar.multiselect(
    "Mes",
    options=sorted(df.mes.unique()) if 'mes' in df.columns else [],
    default=None,
    placeholder="TODOS"
)

# Rango de hectáreas
st.sidebar.markdown("### Rango de Hectáreas")
hectareas_min = st.sidebar.number_input(
    "Mínimo",
    min_value=0.0,
    max_value=float(df.Total_hectareas.max()),
    value=0.0
)

hectareas_max = st.sidebar.number_input(
    "Máximo",
    min_value=0.0,
    max_value=float(df.Total_hectareas.max()),
    value=float(df.Total_hectareas.max())
)

# Aplicación de filtros
df_f = df[
    (df.anio.between(years[0], years[1])) &
    (df.Total_hectareas.between(hectareas_min, hectareas_max))
]

if estado:
    df_f = df_f[df_f.Estado.isin(estado)]

if vegetacion:
    df_f = df_f[df_f.Tipo_Vegetacion.isin(vegetacion)]

if causa and 'Causa' in df.columns:
    df_f = df_f[df_f.Causa.isin(causa)]

if mes and 'mes' in df.columns:
    df_f = df_f[df_f.mes.isin(mes)]

# Mostrar filtros activos
st.sidebar.markdown("---")
st.sidebar.markdown("### Filtros Activos")
if not estado:
    st.sidebar.info("Estados: **TODOS**")
else:
    st.sidebar.success(f"Estados: {len(estado)} seleccionado(s)")

if not vegetacion:
    st.sidebar.info("Vegetación: **TODOS**")
else:
    st.sidebar.success(f"Vegetación: {len(vegetacion)} seleccionado(s)")

if not causa:
    st.sidebar.info("Causas: **TODOS**")
else:
    st.sidebar.success(f"Causas: {len(causa)} seleccionado(s)")

if not mes:
    st.sidebar.info("Meses: **TODOS**")
else:
    st.sidebar.success(f"Meses: {len(mes)} seleccionado(s)")

# =====================
# KPIs PRINCIPALES
# =====================
st.subheader("Indicadores Clave")

c1, c2, c3, c4, c5, c6 = st.columns(6)

c1.metric("Total de incendios", f"{len(df_f):,}")
c2.metric("Hectáreas quemadas", f"{df_f.Total_hectareas.sum():,.0f}")
c3.metric("Duración promedio", f"{df_f.Duracion.mean():.2f} hrs")
c4.metric("Tiempo de llegada", f"{df_f.Llegada.mean():.2f}")
c5.metric("Estados afectados", f"{df_f.Estado.nunique()}")
c6.metric("Vegetación más afectada", df_f.Tipo_Vegetacion.value_counts().index[0] if len(df_f) > 0 else "N/A")

# =====================
# ANÁLISIS TEMPORAL
# =====================
st.subheader("Evolución Temporal")

col1, col2 = st.columns(2)

with col1:
    fig = px.line(
        df_f.groupby("anio").size().reset_index(name="Incendios"),
        x="anio",
        y="Incendios",
        title="Número de Incendios por Año"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.line(
        df_f.groupby("anio")["Total_hectareas"].sum().reset_index(),
        x="anio",
        y="Total_hectareas",
        title="Hectáreas Quemadas por Año"
    )
    st.plotly_chart(fig, use_container_width=True)

# =====================
# ANÁLISIS POR VEGETACIÓN
# =====================
st.subheader("Incendios por Tipo de Vegetación")

col1, col2 = st.columns(2)

with col1:
    fig = px.bar(
        df_f.groupby("Tipo_Vegetacion").size().reset_index(name="Incendios"),
        x="Tipo_Vegetacion",
        y="Incendios",
        title="Incendios por Tipo de Vegetación"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.box(
        df_f,
        x="Tipo_Vegetacion",
        y="Total_hectareas",
        title="Distribución de Hectáreas Quemadas"
    )
    st.plotly_chart(fig, use_container_width=True)

# =====================
# MAPA INTERACTIVO
# =====================
st.subheader("Mapa de Incendios")

mapa = folium.Map(
    location=[df_f.latitud.mean(), df_f.longitud.mean()],
    zoom_start=5
)

# Usar una muestra determinística para evitar re-renderizado constante
df_map = df_f.sample(min(2000, len(df_f)), random_state=42)

for _, row in df_map.iterrows():
    tooltip_text = f"""
    <b>Estado:</b> {row.Estado}<br>
    <b>Causa:</b> {row.Causa if 'Causa' in row else 'N/A'}<br>
    <b>Hectáreas:</b> {row.Total_hectareas:.2f}<br>
    <b>Latitud:</b> {row.latitud:.4f}<br>
    <b>Longitud:</b> {row.longitud:.4f}
    """
    
    folium.CircleMarker(
        location=[row.latitud, row.longitud],
        radius=3,
        popup=f"""
        Estado: {row.Estado}<br>
        Vegetación: {row.Tipo_Vegetacion}<br>
        Hectáreas: {row.Total_hectareas}
        """,
        tooltip=tooltip_text,
        fill=True
    ).add_to(mapa)

st_folium(mapa, width=None, height=500, returned_objects=[])

# =====================
# EFICIENCIA OPERATIVA
# =====================
st.subheader("Eficiencia de Respuesta")

fig = px.scatter(
    df_f,
    x="Llegada",
    y="Total_hectareas",
    color="Tipo_Vegetacion",
    title="Tiempo de Llegada vs Daño"
)
st.plotly_chart(fig, use_container_width=True)

# =====================
# PCA - ANÁLISIS AVANZADO
# =====================
st.subheader("Análisis Multivariable (PCA)")

vars_pca = [
    "Arbolado_Adulto", "Renuevo", "Arbustivo",
    "Herbaceo", "Hojarasca", "Total_hectareas", "Duracion"
]

df_pca = df_f[vars_pca].dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_pca)

pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)

df_pca_vis = pd.DataFrame(
    components,
    columns=["PC1", "PC2"]
)

df_pca_vis["Tipo_Vegetacion"] = df_f.loc[df_pca.index, "Tipo_Vegetacion"]

fig = px.scatter(
    df_pca_vis,
    x="PC1",
    y="PC2",
    color="Tipo_Vegetacion",
    title="PCA de Incendios"
)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    f"Varianza explicada: PC1 = {pca.explained_variance_ratio_[0]:.2%}, "
    f"PC2 = {pca.explained_variance_ratio_[1]:.2%}"
)

# =====================
# TABLA DE DATOS
# =====================
st.subheader("Explorador de Datos")
st.dataframe(df_f, use_container_width=True)
