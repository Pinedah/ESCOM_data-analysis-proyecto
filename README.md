# Dashboard de Análisis de Incendios Forestales

Proyecto final de Analítica y Visualización de Datos - ESCOM IPN

**Grupo:** 5AM1  
**Autores:**
- Pineda Hernández Francisco
- Ramírez Aguilar Rodrigo Vidal

## Descripción

Dashboard interactivo para análisis exploratorio, operativo y estratégico de incendios forestales en México (2015-2024). El proyecto incluye análisis temporal, geográfico, de causas, eficiencia operativa, y proyecciones basadas en datos históricos.

## Tecnologías y Dependencias

### Versiones Recomendadas

- **Python:** 3.8 o superior
- **Streamlit:** Última versión estable

### Dependencias Principales

```txt
streamlit          # Framework para aplicaciones web interactivas
pandas             # Manipulación y análisis de datos
numpy              # Operaciones numéricas
plotly             # Visualizaciones interactivas
scikit-learn       # Machine learning (PCA, regresión)
folium             # Mapas interactivos
streamlit-folium   # Integración de Folium con Streamlit
```

### Librerías Python Utilizadas

- `scipy`: Para pruebas estadísticas
- `sklearn.preprocessing.StandardScaler`: Normalización de datos
- `sklearn.decomposition.PCA`: Análisis de componentes principales
- `sklearn.linear_model.LinearRegression`: Regresión lineal para proyecciones

## Instalación y Configuración

### 1. Clonar el Repositorio

```bash
git clone https://github.com/pinedah/ESCOM_data-analysis-proyecto.git
cd ESCOM_data-analysis-proyecto
```

### 2. Crear un Entorno Virtual (Recomendado)

**En Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**En Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar Dependencias

```bash
pip install -r requirements.txt
```

## Ejecución del Proyecto

### Dashboard Interactivo (Streamlit)

```bash
streamlit run streamlit_v1.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`
