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

years = st.sidebar.slider(
    "Año",
    int(df.anio.min()),
    int(df.anio.max()),
    (int(df.anio.min()), int(df.anio.max()))
)

estado = st.sidebar.multiselect(
    "Estado",
    options=sorted(df.Estado.unique())
)

vegetacion = st.sidebar.multiselect(
    "Tipo de Vegetación",
    options=sorted(df.Tipo_Vegetacion.unique())
)

# Aplicación de filtros
df_f = df[
    (df.anio.between(years[0], years[1]))
]

if estado:
    df_f = df_f[df_f.Estado.isin(estado)]

if vegetacion:
    df_f = df_f[df_f.Tipo_Vegetacion.isin(vegetacion)]

# =====================
# KPIs PRINCIPALES
# =====================
st.subheader("Indicadores Clave")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Incendios", len(df_f))
c2.metric("Hectáreas Quemadas", f"{df_f.Total_hectareas.sum():,.0f}")
c3.metric("Duración Promedio (hrs)", f"{df_f.Duracion.mean():.2f}")
c4.metric("Tiempo Llegada Promedio", f"{df_f.Llegada.mean():.2f}")

# =====================
# ANÁLISIS TEMPORAL
# =====================
st.subheader("📈 Evolución Temporal")

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

for _, row in df_f.sample(min(2000, len(df_f))).iterrows():
    folium.CircleMarker(
        location=[row.latitud, row.longitud],
        radius=3,
        popup=f"""
        Estado: {row.Estado}<br>
        Vegetación: {row.Tipo_Vegetacion}<br>
        Hectáreas: {row.Total_hectareas}
        """,
        fill=True
    ).add_to(mapa)

st_folium(mapa, use_container_width=True)

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
