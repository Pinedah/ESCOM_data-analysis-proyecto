import streamlit_v1 as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('dataset_limpio.csv')
    return df

df = load_data()

# Title and description
st.title("Dashboard de Análisis de Incendios Forestales")
st.markdown("---")

# Sidebar filters
st.sidebar.header("Filtros")
selected_years = st.sidebar.multiselect("Año", sorted(df['anio'].unique()), default=sorted(df['anio'].unique()))
selected_states = st.sidebar.multiselect("Estado", sorted(df['Estado'].unique()), default=sorted(df['Estado'].unique())[:5] if len(df['Estado'].unique()) > 5 else sorted(df['Estado'].unique()))
selected_causes = st.sidebar.multiselect("Causa", sorted(df['Causa'].unique()), default=sorted(df['Causa'].unique()))

# Filter data
filtered_df = df[
    (df['anio'].isin(selected_years) if selected_years else True) &
    (df['Estado'].isin(selected_states) if selected_states else True) &
    (df['Causa'].isin(selected_causes) if selected_causes else True)
]

# KPIs
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Incendios", f"{len(filtered_df):,}")
with col2:
    st.metric("Hectáreas Afectadas", f"{filtered_df['Total_hectareas'].sum():,.2f}")
with col3:
    st.metric("Promedio Hectáreas/Incendio", f"{filtered_df['Total_hectareas'].mean():,.2f}")
with col4:
    st.metric("Duración Promedio (min)", f"{filtered_df['Duracion'].mean():,.0f}")

st.markdown("---")

# Row 1: Two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Incendios por Año")
    yearly_fires = filtered_df.groupby('anio').size().reset_index(name='count')
    fig1 = px.bar(yearly_fires, x='anio', y='count', 
                  labels={'anio': 'Año', 'count': 'Número de Incendios'},
                  color='count', color_continuous_scale='Reds')
    fig1.update_layout(showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("Hectáreas Afectadas por Año")
    yearly_hectares = filtered_df.groupby('anio')['Total_hectareas'].sum().reset_index()
    fig2 = px.area(yearly_hectares, x='anio', y='Total_hectareas',
                   labels={'anio': 'Año', 'Total_hectareas': 'Hectáreas'},
                   color_discrete_sequence=['#FF6B6B'])
    st.plotly_chart(fig2, use_container_width=True)

# Row 2: Three columns
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Top 10 Estados con Más Incendios")
    top_states = filtered_df['Estado'].value_counts().head(10).reset_index()
    top_states.columns = ['Estado', 'count']
    fig3 = px.bar(top_states, y='Estado', x='count', orientation='h',
                  labels={'count': 'Número de Incendios'},
                  color='count', color_continuous_scale='Oranges')
    fig3.update_layout(showlegend=False, yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig3, use_container_width=True)

with col2:
    st.subheader("Distribución por Causa")
    cause_dist = filtered_df['Causa'].value_counts().reset_index()
    cause_dist.columns = ['Causa', 'count']
    fig4 = px.pie(cause_dist, values='count', names='Causa', hole=0.4,
                  color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(fig4, use_container_width=True)

with col3:
    st.subheader("Distribución por Tipo de Vegetación")
    veg_dist = filtered_df['Tipo_Vegetacion'].value_counts().head(10).reset_index()
    veg_dist.columns = ['Tipo_Vegetacion', 'count']
    fig5 = px.pie(veg_dist, values='count', names='Tipo_Vegetacion', hole=0.4,
                  color_discrete_sequence=px.colors.sequential.Peach)
    st.plotly_chart(fig5, use_container_width=True)

# Row 3: Map
st.subheader("Mapa de Incendios")
map_df = filtered_df[['latitud', 'longitud', 'Estado', 'Total_hectareas', 'Causa']].dropna()
fig6 = px.scatter_mapbox(map_df, lat='latitud', lon='longitud', 
                         color='Total_hectareas', size='Total_hectareas',
                         hover_data=['Estado', 'Causa'],
                         color_continuous_scale='Hot', zoom=4,
                         labels={'Total_hectareas': 'Hectáreas Afectadas'})
fig6.update_layout(mapbox_style="open-street-map", height=500)
st.plotly_chart(fig6, use_container_width=True)

# Row 4: Two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribución de Tamaño de Incendios")
    size_order = ['0 a 5 hectáreas', '6 a 10 hectáreas', '11 a 20 hectáreas', 
                  '21 a 50 hectáreas', '51 a 100 hectáreas', 'Mayor a 100 hectáreas']
    size_dist = filtered_df['Tamano'].value_counts().reindex(size_order, fill_value=0).reset_index()
    size_dist.columns = ['Tamano', 'count']
    fig7 = px.bar(size_dist, x='Tamano', y='count',
                  labels={'Tamano': 'Tamaño', 'count': 'Número de Incendios'},
                  color='count', color_continuous_scale='Reds')
    fig7.update_layout(showlegend=False, xaxis_tickangle=-45)
    st.plotly_chart(fig7, use_container_width=True)

with col2:
    st.subheader("Tipo de Impacto")
    impact_dist = filtered_df['Tipo_impacto'].value_counts().reset_index()
    impact_dist.columns = ['Tipo_impacto', 'count']
    fig8 = px.bar(impact_dist, x='Tipo_impacto', y='count',
                  labels={'Tipo_impacto': 'Tipo de Impacto', 'count': 'Número de Incendios'},
                  color='count', color_continuous_scale='Reds')
    fig8.update_layout(showlegend=False)
    st.plotly_chart(fig8, use_container_width=True)

# Row 5: Correlation and Time Analysis
col1, col2 = st.columns(2)

with col1:
    st.subheader("Correlación: Duración vs Hectáreas Afectadas")
    sample_df = filtered_df.sample(min(1000, len(filtered_df)))
    fig9 = px.scatter(sample_df, x='Duracion', y='Total_hectareas',
                     color='Causa', hover_data=['Estado'],
                     labels={'Duracion': 'Duración (min)', 'Total_hectareas': 'Hectáreas Afectadas'},
                     opacity=0.6)
    st.plotly_chart(fig9, use_container_width=True)

with col2:
    st.subheader("Tiempo de Detección vs Llegada")
    detection_df = filtered_df[['Deteccion', 'Llegada']].dropna()
    if len(detection_df) > 0:
        fig10 = go.Figure()
        fig10.add_trace(go.Box(y=detection_df['Deteccion'], name='Detección', marker_color='lightblue'))
        fig10.add_trace(go.Box(y=detection_df['Llegada'], name='Llegada', marker_color='lightcoral'))
        fig10.update_layout(yaxis_title='Tiempo (minutos)', showlegend=True)
        st.plotly_chart(fig10, use_container_width=True)

# Data table
st.markdown("---")
st.subheader("Datos Filtrados")
st.dataframe(filtered_df.head(100), use_container_width=True)

# Download button
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Descargar datos filtrados (CSV)",
    data=csv,
    file_name='incendios_filtrados.csv',
    mime='text/csv',
)