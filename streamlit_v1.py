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
c3.metric("Duración promedio", f"{(df_f.Duracion.mean() / 3600):.2f} hrs")
c4.metric("Tiempo de llegada", f"{(df_f.Llegada.mean() / 3600):.2f} hrs")
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
# RANKINGS DINÁMICOS
# =====================
st.subheader("Rankings de Incendios")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Top 10 Municipios Más Afectados")
    
    if 'Municipio' in df_f.columns:
        top_municipios = df_f.groupby('Municipio').agg({
            'Total_hectareas': 'sum',
            'Estado': 'first'
        }).sort_values('Total_hectareas', ascending=False).head(10).reset_index()
        
        top_municipios.columns = ['Municipio', 'Hectáreas Totales', 'Estado']
        top_municipios['Hectáreas Totales'] = top_municipios['Hectáreas Totales'].apply(lambda x: f"{x:,.0f}")
        
        st.dataframe(top_municipios, use_container_width=True, hide_index=True)
    else:
        st.warning("La columna 'Municipio' no está disponible en los datos.")

with col2:
    st.markdown("### Top 10 Incendios Más Grandes")
    
    top_incendios = df_f.nlargest(10, 'Total_hectareas')[
        ['Estado', 'Municipio', 'Total_hectareas', 'Tipo_Vegetacion', 'anio']
    ].copy() if 'Municipio' in df_f.columns else df_f.nlargest(10, 'Total_hectareas')[
        ['Estado', 'Total_hectareas', 'Tipo_Vegetacion', 'anio']
    ].copy()
    
    top_incendios['Total_hectareas'] = top_incendios['Total_hectareas'].apply(lambda x: f"{x:,.2f}")
    
    if 'Municipio' in top_incendios.columns:
        top_incendios.columns = ['Estado', 'Municipio', 'Hectáreas', 'Vegetación', 'Año']
    else:
        top_incendios.columns = ['Estado', 'Hectáreas', 'Vegetación', 'Año']
    
    st.dataframe(top_incendios, use_container_width=True, hide_index=True)

# =====================
# EFICIENCIA OPERATIVA
# =====================
st.subheader("Eficiencia Operativa")

# Indicadores clave
col1, col2 = st.columns(2)

with col1:
    # % de incendios atendidos rápidamente (menos de 2 horas)
    if len(df_f) > 0:
        rapidos = (df_f['Llegada'] <= 7200).sum()  # 2 horas = 7200 segundos
        pct_rapidos = (rapidos / len(df_f)) * 100
        st.metric("Incendios atendidos rápidamente", f"{pct_rapidos:.1f}%", 
                  help="Porcentaje de incendios con tiempo de llegada ≤ 2 horas")
    else:
        st.metric("Incendios atendidos rápidamente", "N/A")

with col2:
    # Estimación de reducción de daño por respuesta temprana
    if len(df_f) > 0 and 'Llegada' in df_f.columns and 'Total_hectareas' in df_f.columns:
        # Calcular promedio de hectáreas para respuestas rápidas vs lentas
        df_temp = df_f[df_f['Llegada'] > 0].copy()
        rapidos_ha = df_temp[df_temp['Llegada'] <= 7200]['Total_hectareas'].mean()  # 2 horas = 7200 segundos
        lentos_ha = df_temp[df_temp['Llegada'] > 7200]['Total_hectareas'].mean()
        
        if pd.notna(rapidos_ha) and pd.notna(lentos_ha) and lentos_ha > 0:
            reduccion = ((lentos_ha - rapidos_ha) / lentos_ha) * 100
            st.metric("Reducción de daño estimada", f"{reduccion:.1f}%",
                     help="Reducción promedio de hectáreas por respuesta temprana (≤2 hrs)")
        else:
            st.metric("Reducción de daño estimada", "N/A")
    else:
        st.metric("Reducción de daño estimada", "N/A")

st.markdown("---")

# Scatter: tiempo de llegada vs hectáreas
df_temp = df_f.copy()
df_temp['Llegada_hrs'] = df_temp['Llegada'] / 3600
fig = px.scatter(
    df_temp,
    x="Llegada_hrs",
    y="Total_hectareas",
    color="Tipo_Vegetacion",
    title="Tiempo de Llegada vs Hectáreas Afectadas",
    labels={"Llegada_hrs": "Tiempo de Llegada (hrs)", "Total_hectareas": "Hectáreas"}
)
st.plotly_chart(fig, use_container_width=True)

# Scatter: detección vs duración
if 'Deteccion' in df_f.columns and 'Duracion' in df_f.columns:
    df_temp = df_f.copy()
    df_temp['Duracion_hrs'] = df_temp['Duracion'] / 3600
    fig = px.scatter(
        df_temp,
        x="Deteccion",
        y="Duracion_hrs",
        color="Tipo_Vegetacion",
        title="Tiempo de Detección vs Duración del Incendio",
        labels={"Deteccion": "Tiempo de Detección (hrs)", "Duracion_hrs": "Duración (hrs)"}
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Datos de 'Deteccion' no disponibles")

col3, col4 = st.columns(2)

with col3:
    # Boxplot: tiempo de llegada por región/estado
    df_temp = df_f.copy()
    df_temp['Llegada_hrs'] = df_temp['Llegada'] / 3600
    fig = px.box(
        df_temp,
        x="Estado",
        y="Llegada_hrs",
        title="Tiempo de Llegada por Estado",
        labels={"Estado": "Estado", "Llegada_hrs": "Tiempo de Llegada (hrs)"}
    )
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

with col4:
    # Heatmap: región vs duración promedio
    if 'Duracion' in df_f.columns:
        duracion_estado = df_f.groupby('Estado')['Duracion'].mean().reset_index()
        duracion_estado['Duracion'] = duracion_estado['Duracion'] / 3600
        duracion_estado = duracion_estado.sort_values('Duracion', ascending=False)
        
        fig = px.bar(
            duracion_estado.head(15),
            x="Estado",
            y="Duracion",
            title="Duración Promedio por Estado (Top 15)",
            labels={"Estado": "Estado", "Duracion": "Duración Promedio (hrs)"},
            color="Duracion",
            color_continuous_scale="Reds"
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Datos de 'Duracion' no disponibles")

# =====================
# PCA - ANÁLISIS AVANZADO
# =====================
st.subheader("Análisis de Componentes Principales (PCA)")

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

df_pca_vis["Tipo_Vegetacion"] = df_f.loc[df_pca.index, "Tipo_Vegetacion"].values

fig = px.scatter(
    df_pca_vis,
    x="PC1",
    y="PC2",
    color="Tipo_Vegetacion",
    title="PCA de Incendios Forestales"
)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    f"Varianza explicada: PC1 = {pca.explained_variance_ratio_[0]:.2%}, "
    f"PC2 = {pca.explained_variance_ratio_[1]:.2%}"
)

# =====================
# CORRELACIÓN DE PEARSON
# =====================
st.subheader("Matriz de Correlación de Pearson")

vars_corr = [
    "Arbolado_Adulto", "Renuevo", "Arbustivo",
    "Herbaceo", "Hojarasca", "Total_hectareas", 
    "Duracion", "Llegada"
]

# Filtrar solo las columnas que existen en el DataFrame
vars_corr_disponibles = [var for var in vars_corr if var in df_f.columns]

df_corr = df_f[vars_corr_disponibles].dropna()

# Calcular la matriz de correlación de Pearson
corr_matrix = df_corr.corr()

# Crear heatmap con plotly
fig = px.imshow(
    corr_matrix,
    text_auto='.2f',
    color_continuous_scale='RdBu_r',
    aspect="auto",
    title="Correlación de Pearson entre Variables",
    labels=dict(color="Correlación"),
    zmin=-1,
    zmax=1
)

fig.update_xaxes(side="bottom")
fig.update_layout(height=500)

st.plotly_chart(fig, use_container_width=True)

# Mostrar las correlaciones más fuertes
st.markdown("**Top 5 correlaciones más fuertes:**")

# Obtener pares de correlaciones sin duplicados
corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        corr_pairs.append({
            'Variable 1': corr_matrix.columns[i],
            'Variable 2': corr_matrix.columns[j],
            'Correlación': corr_matrix.iloc[i, j]
        })

df_corr_pairs = pd.DataFrame(corr_pairs)
df_corr_pairs = df_corr_pairs.reindex(
    df_corr_pairs['Correlación'].abs().sort_values(ascending=False).index
).head(5)

st.dataframe(df_corr_pairs, use_container_width=True, hide_index=True)

# =====================
# TABLA DE DATOS
# =====================
st.subheader("Explorador de Datos")
st.dataframe(df_f, use_container_width=True)
