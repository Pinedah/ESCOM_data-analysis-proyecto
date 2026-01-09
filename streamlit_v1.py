import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

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
    options=sorted(df.Mes_Nombre.unique()) if 'Mes_Nombre' in df.columns else [],
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

if mes and 'Mes_Nombre' in df.columns:
    df_f = df_f[df_f.Mes_Nombre.isin(mes)]

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
# CAUSAS Y PREVENCIÓN
# =====================
st.subheader("Análisis de Causas y Prevención")

if 'Causa' in df_f.columns:
    # Distribución de causas - Pie chart
    causas_count = df_f['Causa'].value_counts().reset_index()
    causas_count.columns = ['Causa', 'Frecuencia']
    
    fig = px.pie(
        causas_count,
        values='Frecuencia',
        names='Causa',
        title='Distribución de Incendios por Causa'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Evolución anual de causas
    causas_tiempo = df_f.groupby(['anio', 'Causa']).size().reset_index(name='Incendios')
    
    fig = px.line(
        causas_tiempo,
        x='anio',
        y='Incendios',
        color='Causa',
        title='Evolución Temporal de Causas',
        labels={'anio': 'Año', 'Incendios': 'Número de Incendios'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Sankey: Causa → Vegetación → Impacto
    # Preparar datos para Sankey
    sankey_data = df_f.groupby(['Causa', 'Tipo_Vegetacion'])['Total_hectareas'].sum().reset_index()
    sankey_data = sankey_data.nlargest(15, 'Total_hectareas')  # Top 15 para legibilidad
    
    # Crear nodos únicos
    causas_unicas = sankey_data['Causa'].unique().tolist()
    vegetacion_unica = sankey_data['Tipo_Vegetacion'].unique().tolist()
    
    all_nodes = causas_unicas + vegetacion_unica
    
    # Crear índices para source y target
    source = [all_nodes.index(causa) for causa in sankey_data['Causa']]
    target = [all_nodes.index(veg) for veg in sankey_data['Tipo_Vegetacion']]
    value = sankey_data['Total_hectareas'].tolist()
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_nodes
        ),
        link=dict(
            source=source,
            target=target,
            value=value
        )
    )])
    
    fig.update_layout(title_text="Flujo: Causa → Tipo de Vegetación", height=400)
    st.plotly_chart(fig, use_container_width=True)
    
else:
    st.warning("La columna 'Causa' no está disponible en los datos.")

# =====================
# ANÁLISIS DE ESTACIONALIDAD
# =====================
st.subheader("Análisis de Estacionalidad")

if 'Mes_Nombre' in df_f.columns:
    col1, col2 = st.columns(2)
    
    with col1:
        # Heatmap mes vs año
        estacionalidad = df_f.groupby(['anio', 'Mes_Nombre']).size().reset_index(name='Incendios')
        pivot_estacional = estacionalidad.pivot(index='Mes_Nombre', columns='anio', values='Incendios').fillna(0)
        
        fig = px.imshow(
            pivot_estacional,
            labels=dict(x="Año", y="Mes", color="Incendios"),
            x=pivot_estacional.columns,
            y=pivot_estacional.index,
            color_continuous_scale='Reds',
            title='Patrón Estacional de Incendios (Mes vs Año)',
            aspect='auto'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Calendario mensual de incendios
        incendios_mes = df_f.groupby('Mes_Nombre').size().reset_index(name='Total_Incendios')
        
        fig = px.bar(
            incendios_mes,
            x='Mes_Nombre',
            y='Total_Incendios',
            title='Distribución Mensual de Incendios',
            labels={'Mes_Nombre': 'Mes', 'Total_Incendios': 'Número de Incendios'},
            color='Total_Incendios',
            color_continuous_scale='Oranges'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Insight estacional
    mes_critico = df_f.groupby('Mes_Nombre').size().idxmax()
    st.info(f"Mes crítico: **{mes_critico}** - Aumentar recursos preventivos en este período")
else:
    st.warning("Datos de 'Mes_Nombre' no disponibles")

# =====================
# CLASIFICACIÓN POR SEVERIDAD
# =====================
st.subheader("Clasificación por Severidad de Incendios")

# Clasificar incendios por tamaño
def clasificar_severidad(hectareas):
    if hectareas < 10:
        return 'Pequeño (<10 ha)'
    elif hectareas < 100:
        return 'Mediano (10-100 ha)'
    elif hectareas < 1000:
        return 'Grande (100-1000 ha)'
    else:
        return 'Catastrófico (>1000 ha)'

df_f['Severidad'] = df_f['Total_hectareas'].apply(clasificar_severidad)

col1, col2 = st.columns(2)

with col1:
    # Distribución por severidad
    severidad_count = df_f['Severidad'].value_counts().reset_index()
    severidad_count.columns = ['Severidad', 'Frecuencia']
    
    fig = px.pie(
        severidad_count,
        values='Frecuencia',
        names='Severidad',
        title='Distribución de Incendios por Severidad',
        color_discrete_sequence=px.colors.sequential.Reds
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Severidad por estado (top 10)
    severidad_estado = df_f.groupby(['Estado', 'Severidad']).size().reset_index(name='Cantidad')
    top_estados = df_f['Estado'].value_counts().head(10).index
    severidad_top = severidad_estado[severidad_estado['Estado'].isin(top_estados)]
    
    fig = px.bar(
        severidad_top,
        x='Estado',
        y='Cantidad',
        color='Severidad',
        title='Severidad de Incendios por Estado (Top 10)',
        labels={'Cantidad': 'Número de Incendios'},
        barmode='stack'
    )
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

# Estadísticas de severidad
pct_controlables = ((df_f['Total_hectareas'] < 100).sum() / len(df_f)) * 100
pct_catastroficos = ((df_f['Total_hectareas'] >= 1000).sum() / len(df_f)) * 100
st.metric("Incendios controlables (<100 ha)", f"{pct_controlables:.1f}%")
st.caption(f"Incendios catastróficos (≥1000 ha): {pct_catastroficos:.1f}%")

# =====================
# COMPARATIVO AÑO A AÑO
# =====================
st.subheader("Análisis Comparativo Año a Año")

if len(df_f['anio'].unique()) >= 2:
    anios_disponibles = sorted(df_f['anio'].unique(), reverse=True)
    anio_actual = anios_disponibles[0]
    anio_anterior = anios_disponibles[1] if len(anios_disponibles) > 1 else anio_actual - 1
    
    col1, col2, col3, col4 = st.columns(4)
    
    # KPIs comparativos
    actual = df_f[df_f['anio'] == anio_actual]
    anterior = df_f[df_f['anio'] == anio_anterior]
    
    if len(anterior) > 0:
        delta_incendios = len(actual) - len(anterior)
        delta_hectareas = actual['Total_hectareas'].sum() - anterior['Total_hectareas'].sum()
        delta_duracion = (actual['Duracion'].mean() - anterior['Duracion'].mean()) / 3600
        delta_llegada = (actual['Llegada'].mean() - anterior['Llegada'].mean()) / 3600
        
        col1.metric(
            f"Incendios {anio_actual}",
            f"{len(actual):,}",
            f"{delta_incendios:+,}",
            delta_color="inverse"
        )
        
        col2.metric(
            f"Hectáreas {anio_actual}",
            f"{actual['Total_hectareas'].sum():,.0f}",
            f"{delta_hectareas:+,.0f}",
            delta_color="inverse"
        )
        
        col3.metric(
            f"Duración prom {anio_actual}",
            f"{(actual['Duracion'].mean() / 3600):.1f} hrs",
            f"{delta_duracion:+.1f} hrs",
            delta_color="inverse"
        )
        
        col4.metric(
            f"Llegada prom {anio_actual}",
            f"{(actual['Llegada'].mean() / 3600):.1f} hrs",
            f"{delta_llegada:+.1f} hrs",
            delta_color="inverse"
        )
    
    # Variación porcentual YoY por mes
    if 'Mes_Nombre' in df_f.columns:
        yoy_mes = df_f[df_f['anio'].isin([anio_actual, anio_anterior])].groupby(['anio', 'Mes_Nombre']).size().reset_index(name='Incendios')
        
        fig = px.line(
            yoy_mes,
            x='Mes_Nombre',
            y='Incendios',
            color='anio',
            title=f'Comparación Mensual {anio_anterior} vs {anio_actual}',
            labels={'Mes_Nombre': 'Mes', 'Incendios': 'Número de Incendios', 'anio': 'Año'},
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Necesitas datos de al menos 2 años para comparación")

# =====================
# BENCHMARKING ENTRE ESTADOS
# =====================
st.subheader("Benchmarking de Eficiencia entre Estados")

# Calcular métricas de eficiencia por estado
eficiencia_estados = df_f.groupby('Estado').agg({
    'Llegada': lambda x: (x.mean() / 3600),  # Convertir a horas
    'Duracion': lambda x: (x.mean() / 3600),
    'Total_hectareas': 'mean'
}).reset_index()

eficiencia_estados.columns = ['Estado', 'Tiempo_Llegada_hrs', 'Duracion_hrs', 'Hectareas_Promedio']

# Calcular % de respuestas rápidas por estado
respuestas_rapidas = df_f.groupby('Estado').apply(
    lambda x: ((x['Llegada'] <= 7200).sum() / len(x)) * 100
).reset_index(name='Pct_Rapidas')

eficiencia_estados = eficiencia_estados.merge(respuestas_rapidas, on='Estado')

# Top 10 estados más eficientes (menor tiempo llegada y mayor % rápidas)
eficiencia_estados['Score_Eficiencia'] = (100 - eficiencia_estados['Pct_Rapidas']) + eficiencia_estados['Tiempo_Llegada_hrs']
top_eficientes = eficiencia_estados.nsmallest(10, 'Score_Eficiencia')

col1, col2 = st.columns(2)

with col1:
    fig = px.bar(
        top_eficientes,
        x='Pct_Rapidas',
        y='Estado',
        orientation='h',
        title='Top 10 Estados: % Respuestas Rápidas (≤2 hrs)',
        labels={'Pct_Rapidas': '% Respuestas Rápidas', 'Estado': 'Estado'},
        color='Pct_Rapidas',
        color_continuous_scale='Greens'
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.scatter(
        eficiencia_estados,
        x='Tiempo_Llegada_hrs',
        y='Hectareas_Promedio',
        size='Duracion_hrs',
        color='Pct_Rapidas',
        hover_data=['Estado'],
        title='Eficiencia: Tiempo vs Daño por Estado',
        labels={
            'Tiempo_Llegada_hrs': 'Tiempo Llegada (hrs)',
            'Hectareas_Promedio': 'Hectáreas Promedio',
            'Pct_Rapidas': '% Rápidas'
        },
        color_continuous_scale='RdYlGn'
    )
    st.plotly_chart(fig, use_container_width=True)

# Estados con mejores prácticas
mejor_estado = eficiencia_estados.nsmallest(1, 'Score_Eficiencia')['Estado'].values[0]
st.success(f"Estado con mejores prácticas: **{mejor_estado}** - Replicar estrategias")

# =====================
# ANÁLISIS DE RIESGO GEOGRÁFICO
# =====================
st.subheader("Análisis de Riesgo Geográfico")

col1, col2 = st.columns(2)

with col1:
    # Densidad de incendios por estado
    densidad = df_f['Estado'].value_counts().reset_index()
    densidad.columns = ['Estado', 'Frecuencia']
    densidad_top15 = densidad.head(15)
    
    fig = px.bar(
        densidad_top15,
        x='Frecuencia',
        y='Estado',
        orientation='h',
        title='Densidad de Incendios por Estado (Top 15)',
        labels={'Frecuencia': 'Número de Incendios', 'Estado': 'Estado'},
        color='Frecuencia',
        color_continuous_scale='Reds'
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Zonas de alto riesgo recurrente (estados con incendios frecuentes y grandes)
    riesgo_alto = df_f.groupby('Estado').agg({
        'Total_hectareas': ['sum', 'count']
    }).reset_index()
    riesgo_alto.columns = ['Estado', 'Total_Hectareas', 'Num_Incendios']
    riesgo_alto['Riesgo_Score'] = riesgo_alto['Total_Hectareas'] * riesgo_alto['Num_Incendios']
    
    fig = px.scatter(
        riesgo_alto,
        x='Num_Incendios',
        y='Total_Hectareas',
        size='Riesgo_Score',
        color='Riesgo_Score',
        hover_data=['Estado'],
        title='Zonas de Alto Riesgo: Frecuencia vs Impacto',
        labels={
            'Num_Incendios': 'Frecuencia de Incendios',
            'Total_Hectareas': 'Hectáreas Totales Afectadas',
            'Riesgo_Score': 'Score de Riesgo'
        },
        color_continuous_scale='Reds'
    )
    st.plotly_chart(fig, use_container_width=True)

# Recomendación de estaciones
zonas_criticas = riesgo_alto.nlargest(5, 'Riesgo_Score')['Estado'].tolist()
st.warning(f"Zonas prioritarias para estaciones: {', '.join(zonas_criticas)}")

# =====================
# EFICIENCIA DE COMBATE
# =====================
st.subheader("Análisis de Eficiencia de Combate")

# Calcular tasa de propagación
df_f['Tasa_Propagacion'] = df_f.apply(
    lambda row: (row['Total_hectareas'] / (row['Duracion'] / 3600)) if row['Duracion'] > 0 else 0,
    axis=1
)

col1, col2 = st.columns(2)

with col1:
    # Tasa de propagación por tipo de vegetación
    tasa_vegetacion = df_f.groupby('Tipo_Vegetacion')['Tasa_Propagacion'].mean().reset_index()
    tasa_vegetacion = tasa_vegetacion.sort_values('Tasa_Propagacion', ascending=False)
    
    fig = px.bar(
        tasa_vegetacion,
        x='Tasa_Propagacion',
        y='Tipo_Vegetacion',
        orientation='h',
        title='Tasa de Propagación por Vegetación (ha/hora)',
        labels={'Tasa_Propagacion': 'Hectáreas/Hora', 'Tipo_Vegetacion': 'Tipo de Vegetación'},
        color='Tasa_Propagacion',
        color_continuous_scale='Reds'
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Boxplot de eficiencia por vegetación
    fig = px.box(
        df_f[df_f['Tasa_Propagacion'] < df_f['Tasa_Propagacion'].quantile(0.95)],  # Remover outliers
        x='Tipo_Vegetacion',
        y='Tasa_Propagacion',
        title='Distribución de Tasa de Propagación',
        labels={'Tasa_Propagacion': 'Hectáreas/Hora', 'Tipo_Vegetacion': 'Vegetación'}
    )
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

# Vegetación más difícil de controlar
veg_dificil = tasa_vegetacion.nlargest(1, 'Tasa_Propagacion')['Tipo_Vegetacion'].values[0]
st.error(f"Vegetación más difícil de controlar: **{veg_dificil}** - Requiere protocolos especiales")

# =====================
# PROYECCIONES Y TENDENCIAS
# =====================
st.subheader("Proyecciones y Tendencias")

# Tendencia anual con forecast simple
tendencia_anual = df_f.groupby('anio').agg({
    'Total_hectareas': 'sum',
}).reset_index()
tendencia_anual.columns = ['Año', 'Hectareas_Totales']

# Agregar línea de tendencia
from sklearn.linear_model import LinearRegression

X = tendencia_anual[['Año']].values
y = tendencia_anual['Hectareas_Totales'].values

modelo = LinearRegression()
modelo.fit(X, y)

# Proyección próximos 3 años
anios_futuros = np.array([[tendencia_anual['Año'].max() + i] for i in range(1, 4)])
proyeccion = modelo.predict(anios_futuros)

# Crear dataframe para gráfica
proyeccion_df = pd.DataFrame({
    'Año': anios_futuros.flatten(),
    'Hectareas_Totales': proyeccion,
    'Tipo': 'Proyección'
})

tendencia_anual['Tipo'] = 'Histórico'
datos_completos = pd.concat([tendencia_anual, proyeccion_df])

fig = px.line(
    datos_completos,
    x='Año',
    y='Hectareas_Totales',
    color='Tipo',
    title='Tendencia Histórica y Proyección de Hectáreas Quemadas',
    labels={'Hectareas_Totales': 'Hectáreas Totales', 'Año': 'Año'},
    markers=True
)
fig.update_traces(line=dict(dash='dash'), selector=dict(name='Proyección'))
st.plotly_chart(fig, use_container_width=True)

# Tendencia de incidentes
tendencia_incidentes = df_f.groupby('anio').size().reset_index(name='Incendios')
X_inc = tendencia_incidentes[['anio']].values
y_inc = tendencia_incidentes['Incendios'].values

modelo_inc = LinearRegression()
modelo_inc.fit(X_inc, y_inc)
tendencia_inc = modelo_inc.coef_[0]

if tendencia_inc > 0:
    st.warning(f"Tendencia al alza: +{tendencia_inc:.1f} incendios/año promedio")
else:
    st.success(f"Tendencia a la baja: {tendencia_inc:.1f} incendios/año promedio")

# =====================
# ANÁLISIS COSTO-BENEFICIO
# =====================
st.subheader("Análisis de Recursos vs Impacto")

# Scatter 3D: Llegada vs Duración vs Hectáreas
df_temp = df_f.copy()
df_temp['Llegada_hrs'] = df_temp['Llegada'] / 3600
df_temp['Duracion_hrs'] = df_temp['Duracion'] / 3600

# Remover outliers para mejor visualización
df_3d = df_temp[
    (df_temp['Llegada_hrs'] < df_temp['Llegada_hrs'].quantile(0.95)) &
    (df_temp['Duracion_hrs'] < df_temp['Duracion_hrs'].quantile(0.95)) &
    (df_temp['Total_hectareas'] < df_temp['Total_hectareas'].quantile(0.95))
]

fig = px.scatter_3d(
    df_3d.sample(min(1000, len(df_3d))),  # Muestra para rendimiento
    x='Llegada_hrs',
    y='Duracion_hrs',
    z='Total_hectareas',
    color='Tipo_Vegetacion',
    title='Relación 3D: Tiempo Llegada - Duración - Impacto',
    labels={
        'Llegada_hrs': 'Tiempo Llegada (hrs)',
        'Duracion_hrs': 'Duración (hrs)',
        'Total_hectareas': 'Hectáreas'
    }
)
st.plotly_chart(fig, use_container_width=True)

# Costo-beneficio de respuesta rápida
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Impacto de Respuesta Temprana")
    
    # Comparar cuartiles de tiempo de llegada
    df_temp['Cuartil_Llegada'] = pd.qcut(df_temp['Llegada'], q=4, labels=['Q1 (Rápido)', 'Q2', 'Q3', 'Q4 (Lento)'])
    impacto_cuartil = df_temp.groupby('Cuartil_Llegada')['Total_hectareas'].mean().reset_index()
    
    fig = px.bar(
        impacto_cuartil,
        x='Cuartil_Llegada',
        y='Total_hectareas',
        title='Hectáreas Promedio por Velocidad de Respuesta',
        labels={'Cuartil_Llegada': 'Velocidad de Respuesta', 'Total_hectareas': 'Hectáreas Promedio'},
        color='Total_hectareas',
        color_continuous_scale='Reds'
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### ROI de Inversión en Respuesta")
    
    ha_rapido = df_temp[df_temp['Llegada'] <= 7200]['Total_hectareas'].mean()
    ha_lento = df_temp[df_temp['Llegada'] > 7200]['Total_hectareas'].mean()
    
    if pd.notna(ha_rapido) and pd.notna(ha_lento):
        ahorro_ha = ha_lento - ha_rapido
        ahorro_pct = (ahorro_ha / ha_lento) * 100 if ha_lento > 0 else 0
        
        st.metric("Ahorro promedio por respuesta rápida", f"{ahorro_ha:.1f} ha")
        st.metric("Reducción porcentual", f"{ahorro_pct:.1f}%")
        
        # Estimación económica (aproximado)
        costo_ha = 50000  # Costo estimado por hectárea
        ahorro_economico = ahorro_ha * costo_ha
        st.metric("Ahorro económico estimado", f"${ahorro_economico:,.0f} MXN")
        
        st.caption("Basado en costo promedio de $50,000 MXN por hectárea afectada")

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
