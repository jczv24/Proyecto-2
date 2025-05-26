import dash
from dash import dcc, html, exceptions
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import json
import joblib
import numpy as np
from tensorflow.keras.models import load_model

# Inicializar la aplicación Dash
app = dash.Dash(__name__,
                external_stylesheets=[
                    'https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap',
                    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css'
                ])

app.title = "Proyecto 2"
server = app.server

# --- Carga y Preparación Inicial de Datos ---
# Cargar objetos
scaler = joblib.load('scaler.pkl')
columnas_dummies = joblib.load('X_columns.pkl')
modelo = load_model('model.keras')
datos_limpios = pd.read_csv('model_data.csv')

datos_limpios['Ingresa'] = np.where(datos_limpios['punt_global'] >= 300, 1, 0)

# Estandarización de nombres de departamento
datos_limpios['depto_estandarizado'] = datos_limpios['estu_depto_reside'].str.upper().str.strip()
mapeo_departamentos = {
    'BOGOTÁ': 'SANTAFE DE BOGOTA D.C',
    'NORTE SANTANDER': 'NORTE DE SANTANDER',
    'SAN ANDRES': 'ARCHIPIELAGO DE SAN ANDRES PROVIDENCIA Y SANTA CATALINA',
    'VALLE': 'VALLE DEL CAUCA'
}
datos_limpios['depto_estandarizado'] = datos_limpios['depto_estandarizado'].replace(mapeo_departamentos)

# --- Cargar el GeoJSON ---
# Asegúrate de que 'colombia.json' esté en el mismo directorio que este script
try:
    with open('colombia.json', 'r', encoding='utf-8') as f:
        geojson_colombia = json.load(f)
except FileNotFoundError:
    print("ERROR: El archivo 'colombia.json' no se encontró. Asegúrate de que esté en el directorio correcto.")
    geojson_colombia = None
except json.JSONDecodeError:
    print("ERROR: El archivo 'colombia.json' no es un JSON válido.")
    geojson_colombia = None

# --- Funciones para el modelo predictivo ---
def transformar_input(datos_usuario_df):
    # Normalizar
    datos_usuario_df['estu_edad'] = scaler.transform(datos_usuario_df[['estu_edad']])

    # One-hot encoding
    datos_usuario_encoded = pd.get_dummies(datos_usuario_df)
    datos_usuario_encoded = pd.get_dummies(datos_usuario_df).astype(int)

    # Añadir columnas faltantes
    columnas_faltantes = list(set(columnas_dummies) - set(datos_usuario_encoded.columns))
    faltantes_df = pd.DataFrame(0, index=datos_usuario_encoded.index, columns=columnas_faltantes)

    # Concatenar y ordenar columnas
    datos_usuario_encoded = pd.concat([datos_usuario_encoded, faltantes_df], axis=1)
    datos_usuario_encoded = datos_usuario_encoded[columnas_dummies]  # asegurar orden

    return datos_usuario_encoded

nuevo_usuario = pd.DataFrame([{
    'estu_tipodocumento': 'TI',
    'cole_area_ubicacion': 'URBANO',
    'cole_calendario': 'A',
    'cole_genero': 'FEMENINO',
    'cole_jornada': 'Mañana',
    'cole_naturaleza': 'OFICIAL',
    'estu_depto_reside': 'BOGOTA',
    'estu_genero': 'FEMENINO',
    'fami_cuartoshogar': '3',
    'fami_educacionmadre': 'SECUNDARIA',
    'fami_educacionpadre': 'SECUNDARIA',
    'fami_estratovivienda': '5',
    'fami_personashogar': '4',
    'fami_tieneautomovil': 'Si',
    'fami_tienecomputador': 'Si',
    'fami_tieneinternet': 'Si',
    'fami_tienelavadora': 'Si',
    'desemp_ingles': 'B+',
    'estu_edad': 17.0
}])

# Transformar
nuevo_usuario_procesado = transformar_input(nuevo_usuario)

# --- Layout de la Aplicación (sin cambios respecto a tu código original) ---
app.layout = html.Div([
    # Título principal
    html.Div([
        html.H1("Analítica Computacional",
                style={
                    'textAlign': 'center',
                    'color': '#2c3e50',
                    'fontFamily': 'Roboto',
                    'padding': '20px 20px 0px 20px',
                    'marginBottom': '0px',
                    'borderBottom': 'none'
                }),
        html.H2("Proyecto 2",
                style={
                    'textAlign': 'center',
                    'color': '#666',
                    'fontFamily': 'Roboto',
                    'padding': '0px 20px 20px 20px',
                    'marginTop': '0px',
                    'marginBottom': '20px',
                    'borderBottom': '2px solid #eee',
                    'fontWeight': 'normal'
                })
    ]),

    # Contenedor de pestañas
    html.Div([
        dcc.Tabs(id='tabs', value='tab-1', children=[
            dcc.Tab(label='Análisis Descriptivo',
                   value='tab-1',
                   style={
                       'backgroundColor': '#f8f9fa',
                       'color': '#2c3e50',
                       'fontFamily': 'Roboto',
                       'fontWeight': 'bold',
                       'padding': '10px',
                       'border': 'none'
                   },
                   selected_style={
                       'backgroundColor': '#2c3e50',
                       'color': 'white',
                       'fontFamily': 'Roboto',
                       'fontWeight': 'bold',
                       'padding': '10px',
                       'border': 'none'
                   }),
            dcc.Tab(label='Modelo Predictivo',
                   value='tab-2',
                   style={
                       'backgroundColor': '#f8f9fa',
                       'color': '#2c3e50',
                       'fontFamily': 'Roboto',
                       'fontWeight': 'bold',
                       'padding': '10px',
                       'border': 'none'
                   },
                   selected_style={
                       'backgroundColor': '#2c3e50',
                       'color': 'white',
                       'fontFamily': 'Roboto',
                       'fontWeight': 'bold',
                       'padding': '10px',
                       'border': 'none'
                   })
        ], style={
            'fontFamily': 'Roboto',
            'marginBottom': '20px'
        }),

        # Contenido de las pestañas
        html.Div(id='tabs-content')
    ], style={
        'maxWidth': '1200px',
        'margin': '0 auto',
        'padding': '20px',
        'backgroundColor': 'white',
        'borderRadius': '10px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
    })
], style={
    'backgroundColor': '#f0f2f5',
    'minHeight': '100vh',
    'padding': '20px'
})

# Callback para actualizar el contenido de las pestañas
@app.callback(
    Output('tabs-content', 'children'),
    Input('tabs', 'value')
)
def render_content(tab):
    if tab == 'tab-1':
        # --- Cálculo para el nuevo indicador "Ingresa" ---
        puntaje_corte = 300
        count_ingresa = datos_limpios['Ingresa'].value_counts()
        porcentaje_ingresa = (count_ingresa[1] / len(datos_limpios)) * 100 if 1 in count_ingresa else 0
        porcentaje_no_ingresa = (count_ingresa[0] / len(datos_limpios)) * 100 if 0 in count_ingresa else 0
        
        # Histograma para puntaje global con marcador de corte
        fig_performance = px.histogram(
            datos_limpios, 
            x='punt_global',
            color='Ingresa',
            title=f'Distribución de Puntajes Globales - Corte de Admisión: {puntaje_corte}',
            labels={'punt_global': 'Puntaje Global', 'count': 'Cantidad de Estudiantes', 'Ingresa': 'Estado de Admisión'},
            color_discrete_map={0: '#e74c3c', 1: '#2ecc71'},
            nbins=30,
            opacity=0.8
        )
        
        # Añadir línea vertical para el puntaje de corte
        fig_performance.add_vline(x=puntaje_corte, line_width=2, line_dash="dash", line_color="#34495e")
        fig_performance.add_annotation(x=puntaje_corte, y=0.95, yref="paper", text=f"Puntaje de Corte: {puntaje_corte}",
                                   showarrow=True, arrowhead=1, ax=-40, ay=-30, font=dict(color="#34495e"))
        
        # Actualizar leyenda
        fig_performance.update_traces(
            legendgroup="No ingresa", name="No ingresa", selector={"name": "0"}
        )
        fig_performance.update_traces(
            legendgroup="Ingresa", name="Ingresa", selector={"name": "1"}
        )
        
        fig_performance.update_layout(
            title_font_family="Roboto", 
            font_family="Roboto", 
            title_x=0.5, 
            legend_title_text='Estado',
            plot_bgcolor='rgba(0,0,0,0)', 
            paper_bgcolor='rgba(0,0,0,0)',
            bargap=0.1
        )

        # --- NUEVO: Estadísticas y Box Plot para punt_global ---
        summary_stats = datos_limpios['punt_global'].describe()
        mean_score = summary_stats['mean']
        median_score = summary_stats['50%'] # Median is the 50th percentile
        std_score = summary_stats['std']
        min_score = summary_stats['min']
        max_score = summary_stats['max']

        fig_box_punt_global = px.box(datos_limpios, y="punt_global", color="Ingresa",
                                     title="Distribución de Puntaje Global por Estado de Admisión",
                                     color_discrete_map={0: '#e74c3c', 1: '#2ecc71'},
                                     points="outliers") # Muestra outliers
        
        # Actualizar leyenda del boxplot
        fig_box_punt_global.update_traces(
            legendgroup="No ingresa", name="No ingresa", selector={"name": "0"}
        )
        fig_box_punt_global.update_traces(
            legendgroup="Ingresa", name="Ingresa", selector={"name": "1"}
        )
        
        fig_box_punt_global.update_layout(
            title_font_family="Roboto", font_family="Roboto", title_x=0.5,
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            legend_title_text='Estado'
        )
        # --- FIN NUEVO ---

        # Relación entre Estrato Socioeconómico y Admisión
        stratum_performance = datos_limpios.groupby(['fami_estratovivienda', 'Ingresa']).size().reset_index(name='Cantidad')
        stratum_order = sorted(datos_limpios['fami_estratovivienda'].unique(), key=lambda x: int(x.split(' ')[1]) if 'Estrato' in x else 0)
        stratum_performance['fami_estratovivienda'] = pd.Categorical(stratum_performance['fami_estratovivienda'], categories=stratum_order, ordered=True)
        stratum_performance = stratum_performance.sort_values(['fami_estratovivienda', 'Ingresa'])
        
        # Convertir 0 y 1 a etiquetas más legibles
        stratum_performance['Estado de Admisión'] = stratum_performance['Ingresa'].map({0: 'No ingresa', 1: 'Ingresa'})
        
        fig_stratum_performance = px.bar(stratum_performance,
                                         x='fami_estratovivienda', y='Cantidad', color='Estado de Admisión', barmode='group',
                                         title='Estado de Admisión por Estrato Socioeconómico',
                                         labels={'fami_estratovivienda': 'Estrato Socioeconómico'},
                                         color_discrete_map={'No ingresa': '#e74c3c', 'Ingresa': '#2ecc71'},
                                         text_auto=True)
        fig_stratum_performance.update_layout(
            title_font_family="Roboto", font_family="Roboto", title_x=0.5, legend_title_text='Estado',
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
        )
        fig_stratum_performance.update_traces(textposition='outside')

        # Definición del mapa (adaptado para mostrar ingresa vs no ingresa)
        fig_mapa_colombia = None 
        mapa_component_html = None # Para el contenido del mapa o error
        if geojson_colombia:
            df_para_mapa = datos_limpios[datos_limpios['depto_estandarizado'] != 'EXTRANJERO'].copy()
            promedio_por_depto_mapa = df_para_mapa.groupby('depto_estandarizado')['punt_global'].mean().reset_index()
            
            # Calcular porcentaje de admitidos por departamento
            ingresa_counts = df_para_mapa.groupby('depto_estandarizado')['Ingresa'].value_counts().unstack(fill_value=0).reset_index()
            ingresa_counts['total'] = ingresa_counts[0] + ingresa_counts[1]
            ingresa_counts['porcentaje_ingresa'] = (ingresa_counts[1] / ingresa_counts['total'] * 100).round(2)
            
            # Combinar promedios y porcentajes
            map_data_final = pd.merge(promedio_por_depto_mapa, ingresa_counts[['depto_estandarizado', 'porcentaje_ingresa']], on='depto_estandarizado', how='left')
            
            fig_mapa_colombia = px.choropleth(
                map_data_final, geojson=geojson_colombia, locations='depto_estandarizado',
                featureidkey='properties.NOMBRE_DPT', color='porcentaje_ingresa',
                color_continuous_scale="RdYlGn", scope="south america", hover_name='depto_estandarizado',
                hover_data={'punt_global': ':.2f', 'porcentaje_ingresa': ':.2f%'},
                labels={'punt_global':'Puntaje Global Promedio', 'depto_estandarizado': 'Departamento', 'porcentaje_ingresa': 'Porcentaje de Admitidos'},
                title='Porcentaje de Estudiantes Admitidos por Departamento'
            )
            fig_mapa_colombia.update_geos(fitbounds="locations", visible=False)
            fig_mapa_colombia.update_layout(margin={"r":0,"t":40,"l":0,"b":0}, title_x=0.5, title_font_family="Roboto", font_family="Roboto", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            mapa_component_html = html.Div([dcc.Graph(id='mapa-colombia-puntajes', figure=fig_mapa_colombia)], style={'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'backgroundColor': '#f9f9f9', 'marginBottom': '20px'})
        else:
            mapa_component_html = html.P("Error al cargar los datos geográficos para el mapa.", style={'color':'red', 'textAlign':'center', 'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'backgroundColor': '#f9f9f9', 'marginBottom': '20px'})

        # Definición del gráfico de categorías paralelas (adaptado para ingresa vs no ingresa)
        sample_fraction = 0.1
        if len(datos_limpios) * sample_fraction < 1000:
             df_sample_parcats = datos_limpios.sample(n=min(len(datos_limpios), 1000), random_state=1).copy()
        else:
             df_sample_parcats = datos_limpios.sample(frac=sample_fraction, random_state=1).copy()
        
        df_sample_parcats['Estado'] = df_sample_parcats['Ingresa'].map({0: 'No ingresa', 1: 'Ingresa'})
        dimensions_parcats = ['fami_tienecomputador','fami_tieneinternet','Estado']
        custom_labels = {'fami_tienecomputador': 'Familia Tiene Computador','fami_tieneinternet': 'Familia Tiene Internet','Estado': 'Estado de Admisión','Ingresa': 'Estado de Admisión'}
        
        # Usamos el valor numérico (0, 1) para el color
        fig_parallel_categories = px.parallel_categories(
            df_sample_parcats, dimensions=dimensions_parcats, color="Ingresa",
            color_continuous_scale=[(0, "#e74c3c"), (1, "#2ecc71")],
            labels=custom_labels
        )
        fig_parallel_categories.update_layout(title='Relación entre Factores Socioeconómicos y Admisión',title_x=0.5,font_family="Roboto",margin=dict(l=50, r=50, t=80, b=50))
        
        # Ensamblar los hijos de la pestaña 1 en el orden deseado
        children_tab1 = [
            html.H3("Análisis Descriptivo de Datos - Examen Saber Pro ICFES",
                   style={
                       'color': '#34495e', 'fontFamily': 'Roboto', 'textAlign': 'center',
                       'marginBottom': '30px', 'borderBottom': '1px solid #ecf0f1', 'paddingBottom': '10px'
                   }),
            
            # Estadísticas de admisión generales
            html.Div([
                html.Div([
                    html.H4("Estadísticas de Admisión", style={'textAlign': 'center', 'marginBottom': '15px', 'color': '#34495e'}),
                    html.Div([
                        html.Div([
                            html.H1(f"{porcentaje_ingresa:.1f}%", style={'textAlign': 'center', 'color': '#2ecc71', 'margin': '0'}),
                            html.P("Ingresan", style={'textAlign': 'center', 'fontWeight': 'bold'})
                        ], style={'width': '49%', 'display': 'inline-block', 'borderRight': '1px solid #ddd'}),
                        html.Div([
                            html.H1(f"{porcentaje_no_ingresa:.1f}%", style={'textAlign': 'center', 'color': '#e74c3c', 'margin': '0'}),
                            html.P("No ingresan", style={'textAlign': 'center', 'fontWeight': 'bold'})
                        ], style={'width': '49%', 'display': 'inline-block'}),
                    ], style={'marginBottom': '10px'}),
                    html.P(f"El puntaje de corte es {puntaje_corte}. Los estudiantes con puntajes iguales o superiores son admitidos.",
                         style={'textAlign': 'center', 'fontSize': '14px', 'color': '#555'})
                ], style={'width': '100%', 'padding': '15px', 'backgroundColor': '#f9f9f9', 'borderRadius': '5px', 'marginBottom': '20px'})
            ]),
            
            html.Div([
                html.Div([
                    html.H5("Distribución de Puntajes Globales por Estado de Admisión", style={'textAlign': 'center', 'fontFamily': 'Roboto', 'color': '#34495e'}),
                    dcc.Graph(figure=fig_performance)
                ], style={'width': '50%', 'display': 'inline-block', 'padding': '10px', 'verticalAlign': 'top', 'border': '1px solid #ddd', 'borderRadius': '5px', 'backgroundColor': '#f9f9fa', 'marginRight': '10px'}),
                html.Div([
                    html.H5("Estadísticas Clave de Puntaje Global", style={'textAlign': 'center', 'fontFamily': 'Roboto', 'color': '#34495e', 'marginBottom':'15px'}),
                    html.Div([
                        html.P([html.Strong("Promedio: "), f"{mean_score:.2f}"], style={'textAlign': 'center', 'fontFamily': 'Roboto', 'fontSize':'16px', 'marginBottom':'5px'}),
                        html.P([html.Strong("Desviación Estándar: "), f"{std_score:.2f}"], style={'textAlign': 'center', 'fontFamily': 'Roboto', 'fontSize':'16px', 'marginBottom':'20px'}),
                    ]),
                    dcc.Graph(figure=fig_box_punt_global)
                ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px', 'verticalAlign': 'top', 'border': '1px solid #ddd', 'borderRadius': '5px', 'backgroundColor': '#f9f9fa'})
            ], style={'display': 'flex', 'marginBottom': '20px'}),
            
            mapa_component_html, # Mapa en segunda posición de contenido

            html.Div([
                dcc.Graph(figure=fig_stratum_performance)
            ], style={'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'backgroundColor': '#f9f9fa', 'marginBottom': '20px'}),
            
            html.Div([
                dcc.Graph(id='parallel-categories-plot', figure=fig_parallel_categories)
            ], style={'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'backgroundColor': '#f9f9fa', 'marginBottom': '20px'}),
            
            html.P("Universidad de los Andes - Ingenieria industrial",
                  style={
                      'color': '#666', 'fontFamily': 'Roboto', 'fontSize': '16px', 'textAlign': 'center'
                  })
        ]
        return html.Div(children_tab1)

    elif tab == 'tab-2':
        # Predecir usando ejemplo
        prediccion = modelo.predict(nuevo_usuario_procesado)[0][0]
        ingresa = "Ingresa" if prediccion >= 0.7 else "No ingresa"
        color = "#2ecc71" if prediccion >= 0.7 else "#e74c3c"

        return html.Div([
            html.H3("Modelo Predictivo - Simulador de Admisión",
                   style={
                       'color': '#34495e', 'fontFamily': 'Roboto', 'textAlign': 'center',
                       'marginBottom': '30px', 'borderBottom': '1px solid #ecf0f1', 'paddingBottom': '10px'
                   }),
            
            # Contenedor principal con dos columnas
            html.Div([
                # Columna izquierda: formulario
                html.Div([
                    html.H4("Complete el formulario",
                          style={'color': '#2c3e50', 'fontFamily': 'Roboto', 'marginBottom': '20px', 'textAlign': 'center'}),
                    
                    # Datos personales
                    html.Div([
                        html.H5("Datos del Estudiante", style={'textAlign': 'center', 'marginBottom': '20px', 'fontFamily': 'Roboto'}),
                        
                        html.Div([
                            html.Label("Edad:", style={'fontWeight': 'bold', 'fontSize': '14px'}),
                            dcc.Input(
                                id='edad-input',
                                type='number',
                                placeholder='Ej. 17',
                                min=14,
                                max=80,
                                value=17,
                                style={'width': '100%', 'marginBottom': '15px', 'padding': '8px', 'borderRadius': '5px', 'border': '1px solid #ddd'}
                            )
                        ], style={'width': '100%', 'marginBottom': '10px'}),
                        
                        html.Div([
                            html.Label("Tipo de documento:", style={'fontWeight': 'bold', 'fontSize': '14px'}),
                            dcc.Dropdown(
                                id='tipodoc-dropdown',
                                options=[
                                    {'label': 'Cédula de Ciudadanía', 'value': 'CC'},
                                    {'label': 'Tarjeta de Identidad', 'value': 'TI'},
                                    {'label': 'Cédula de Extranjería', 'value': 'CE'},
                                    {'label': 'Permiso Especial de Permanencia', 'value': 'PEP'},
                                    {'label': 'Registro Civil', 'value': 'RC'},
                                    {'label': 'Pasaporte Extranjero', 'value': 'PE'},
                                    {'label': 'Certificado Cedulación', 'value': 'CCB'},
                                    {'label': 'Visa', 'value': 'V'},
                                    {'label': 'Pasaporte Colombiano', 'value': 'PC'},
                                    {'label': 'Número de Identificación Estudiantil', 'value': 'NES'},
                                    {'label': 'Certificado de Residencia', 'value': 'CR'}
                                ],
                                value='TI',
                                style={'width': '100%', 'marginBottom': '15px'}
                            )
                        ], style={'width': '100%', 'marginBottom': '10px'}),
                        
                        html.Div([
                            html.Label("Género:", style={'fontWeight': 'bold', 'fontSize': '14px'}),
                            dcc.RadioItems(
                                id='genero-radio',
                                options=[
                                    {'label': 'Femenino', 'value': 'F'},
                                    {'label': 'Masculino', 'value': 'M'}
                                ],
                                value='F',
                                inputStyle={'marginRight': '5px', 'marginLeft': '10px'}
                            )
                        ], style={'width': '100%', 'marginBottom': '15px'}),
                        
                        html.Div([
                            html.Label("Departamento:", style={'fontWeight': 'bold', 'fontSize': '14px'}),
                            dcc.Dropdown(
                                id='depto-dropdown',
                                options=[
                                    {'label': 'Bogotá', 'value': 'BOGOTÁ'},
                                    {'label': 'Putumayo', 'value': 'PUTUMAYO'},
                                    {'label': 'Valle del Cauca', 'value': 'VALLE'},
                                    {'label': 'Atlántico', 'value': 'ATLANTICO'},
                                    {'label': 'Cesar', 'value': 'CESAR'},
                                    {'label': 'Cundinamarca', 'value': 'CUNDINAMARCA'},
                                    {'label': 'Tolima', 'value': 'TOLIMA'},
                                    {'label': 'Bolívar', 'value': 'BOLIVAR'},
                                    {'label': 'Huila', 'value': 'HUILA'},
                                    {'label': 'Guainía', 'value': 'GUAINIA'},
                                    {'label': 'Risaralda', 'value': 'RISARALDA'},
                                    {'label': 'Antioquia', 'value': 'ANTIOQUIA'},
                                    {'label': 'Caldas', 'value': 'CALDAS'},
                                    {'label': 'Boyacá', 'value': 'BOYACA'},
                                    {'label': 'Meta', 'value': 'META'},
                                    {'label': 'Santander', 'value': 'SANTANDER'},
                                    {'label': 'Magdalena', 'value': 'MAGDALENA'},
                                    {'label': 'La Guajira', 'value': 'LA GUAJIRA'},
                                    {'label': 'Nariño', 'value': 'NARIÑO'},
                                    {'label': 'Chocó', 'value': 'CHOCO'},
                                    {'label': 'Cauca', 'value': 'CAUCA'},
                                    {'label': 'Córdoba', 'value': 'CORDOBA'},
                                    {'label': 'Guaviare', 'value': 'GUAVIARE'},
                                    {'label': 'Sucre', 'value': 'SUCRE'},
                                    {'label': 'Caquetá', 'value': 'CAQUETA'},
                                    {'label': 'Quindío', 'value': 'QUINDIO'},
                                    {'label': 'Norte de Santander', 'value': 'NORTE SANTANDER'},
                                    {'label': 'Arauca', 'value': 'ARAUCA'},
                                    {'label': 'San Andrés', 'value': 'SAN ANDRES'},
                                    {'label': 'Casanare', 'value': 'CASANARE'},
                                    {'label': 'Extranjero', 'value': 'EXTRANJERO'},
                                    {'label': 'Amazonas', 'value': 'AMAZONAS'},
                                    {'label': 'Vichada', 'value': 'VICHADA'},
                                    {'label': 'Vaupés', 'value': 'VAUPES'}
                                ],
                                value='BOGOTÁ',
                                style={'width': '100%', 'marginBottom': '15px'}
                            )
                        ], style={'width': '100%', 'marginBottom': '10px'})
                    ], style={'marginBottom': '15px', 'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '5px', 'boxShadow': '0 1px 3px rgba(0,0,0,0.1)', 'width': '100%'}),
                    
                    # Datos del colegio
                    html.Div([
                        html.H5("Datos del Colegio", style={'textAlign': 'center', 'marginBottom': '20px', 'fontFamily': 'Roboto'}),
                        
                        html.Div([
                            html.Label("Área:", style={'fontWeight': 'bold', 'fontSize': '14px'}),
                            dcc.RadioItems(
                                id='area-radio',
                                options=[
                                    {'label': 'Urbano', 'value': 'URBANO'},
                                    {'label': 'Rural', 'value': 'RURAL'}
                                ],
                                value='URBANO',
                                inputStyle={'marginRight': '5px', 'marginLeft': '10px'}
                            )
                        ], style={'width': '100%', 'marginBottom': '15px'}),
                        
                        html.Div([
                            html.Label("Calendario:", style={'fontWeight': 'bold', 'fontSize': '14px'}),
                            dcc.Dropdown(
                                id='calendario-dropdown',
                                options=[
                                    {'label': 'A', 'value': 'A'},
                                    {'label': 'Otro', 'value': 'OTRO'},
                                    {'label': 'B', 'value': 'B'}
                                ],
                                value='A',
                                style={'width': '100%', 'marginBottom': '15px'}
                            )
                        ], style={'width': '100%', 'marginBottom': '10px'}),
                        
                        html.Div([
                            html.Label("Jornada:", style={'fontWeight': 'bold', 'fontSize': '14px'}),
                            dcc.Dropdown(
                                id='jornada-dropdown',
                                options=[
                                    {'label': 'Tarde', 'value': 'TARDE'},
                                    {'label': 'Mañana', 'value': 'MAÑANA'},
                                    {'label': 'Única', 'value': 'UNICA'},
                                    {'label': 'Completa', 'value': 'COMPLETA'},
                                    {'label': 'Sabatina', 'value': 'SABATINA'},
                                    {'label': 'Noche', 'value': 'NOCHE'}
                                ],
                                value='MAÑANA',
                                style={'width': '100%', 'marginBottom': '15px'}
                            )
                        ], style={'width': '100%', 'marginBottom': '10px'}),
                        
                        html.Div([
                            html.Label("Naturaleza:", style={'fontWeight': 'bold', 'fontSize': '14px'}),
                            dcc.RadioItems(
                                id='naturaleza-radio',
                                options=[
                                    {'label': 'Oficial', 'value': 'OFICIAL'},
                                    {'label': 'No Oficial', 'value': 'NO OFICIAL'}
                                ],
                                value='OFICIAL',
                                inputStyle={'marginRight': '5px', 'marginLeft': '10px'}
                            )
                        ], style={'width': '100%', 'marginBottom': '15px'}),
                        
                        html.Div([
                            html.Label("Tipo de colegio:", style={'fontWeight': 'bold', 'fontSize': '14px'}),
                            dcc.RadioItems(
                                id='cole-genero-radio',
                                options=[
                                    {'label': 'Mixto', 'value': 'MIXTO'},
                                    {'label': 'Femenino', 'value': 'FEMENINO'},
                                    {'label': 'Masculino', 'value': 'MASCULINO'}
                                ],
                                value='MIXTO',
                                inputStyle={'marginRight': '5px', 'marginLeft': '10px'}
                            )
                        ], style={'width': '100%', 'marginBottom': '10px'})
                    ], style={'marginBottom': '15px', 'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '5px', 'boxShadow': '0 1px 3px rgba(0,0,0,0.1)', 'width': '100%'}),
                    
                    # Datos socioeconómicos
                    html.Div([
                        html.H5("Datos Socioeconómicos", style={'textAlign': 'center', 'marginBottom': '20px', 'fontFamily': 'Roboto'}),
                        
                        html.Div([
                            html.Label("Estrato:", style={'fontWeight': 'bold', 'fontSize': '14px'}),
                            dcc.Dropdown(
                                id='estrato-dropdown',
                                options=[
                                    {'label': 'Estrato 1', 'value': 'Estrato 1'},
                                    {'label': 'Estrato 2', 'value': 'Estrato 2'},
                                    {'label': 'Estrato 3', 'value': 'Estrato 3'},
                                    {'label': 'Estrato 4', 'value': 'Estrato 4'},
                                    {'label': 'Estrato 5', 'value': 'Estrato 5'},
                                    {'label': 'Estrato 6', 'value': 'Estrato 6'}
                                ],
                                value='Estrato 3',
                                style={'width': '100%', 'marginBottom': '15px'}
                            )
                        ], style={'width': '100%', 'marginBottom': '10px'}),
                        
                        html.Div([
                            html.Label("Educación de la madre:", style={'fontWeight': 'bold', 'fontSize': '14px'}),
                            dcc.Dropdown(
                                id='educacion-madre-dropdown',
                                options=[
                                    {'label': 'Técnica o tecnológica completa', 'value': 'Técnica o tecnológica completa'},
                                    {'label': 'Técnica o tecnológica incompleta', 'value': 'Técnica o tecnológica incompleta'},
                                    {'label': 'Primaria incompleta', 'value': 'Primaria incompleta'},
                                    {'label': 'Primaria completa', 'value': 'Primaria completa'},
                                    {'label': 'Secundaria (Bachillerato) incompleta', 'value': 'Secundaria (Bachillerato) incompleta'},
                                    {'label': 'Secundaria (Bachillerato) completa', 'value': 'Secundaria (Bachillerato) completa'},
                                    {'label': 'Educación profesional incompleta', 'value': 'Educación profesional incompleta'},
                                    {'label': 'Educación profesional completa', 'value': 'Educación profesional completa'},
                                    {'label': 'Postgrado', 'value': 'Postgrado'},
                                    {'label': 'Ninguno', 'value': 'Ninguno'},
                                    {'label': 'No sabe', 'value': 'No sabe'},
                                    {'label': 'No Aplica', 'value': 'No Aplica'}
                                ],
                                value='Secundaria (Bachillerato) completa',
                                style={'width': '100%', 'marginBottom': '15px'}
                            )
                        ], style={'width': '100%', 'marginBottom': '10px'}),
                        
                        html.Div([
                            html.Label("Educación del padre:", style={'fontWeight': 'bold', 'fontSize': '14px'}),
                            dcc.Dropdown(
                                id='educacion-padre-dropdown',
                                options=[
                                    {'label': 'Técnica o tecnológica completa', 'value': 'Técnica o tecnológica completa'},
                                    {'label': 'Técnica o tecnológica incompleta', 'value': 'Técnica o tecnológica incompleta'},
                                    {'label': 'Primaria incompleta', 'value': 'Primaria incompleta'},
                                    {'label': 'Primaria completa', 'value': 'Primaria completa'},
                                    {'label': 'Secundaria (Bachillerato) incompleta', 'value': 'Secundaria (Bachillerato) incompleta'},
                                    {'label': 'Secundaria (Bachillerato) completa', 'value': 'Secundaria (Bachillerato) completa'},
                                    {'label': 'Educación profesional incompleta', 'value': 'Educación profesional incompleta'},
                                    {'label': 'Educación profesional completa', 'value': 'Educación profesional completa'},
                                    {'label': 'Postgrado', 'value': 'Postgrado'},
                                    {'label': 'Ninguno', 'value': 'Ninguno'},
                                    {'label': 'No sabe', 'value': 'No sabe'},
                                    {'label': 'No Aplica', 'value': 'No Aplica'}
                                ],
                                value='Secundaria (Bachillerato) completa',
                                style={'width': '100%', 'marginBottom': '15px'}
                            )
                        ], style={'width': '100%', 'marginBottom': '10px'}),
                        
                        html.Div([
                            html.Label("Personas en el hogar:", style={'fontWeight': 'bold', 'fontSize': '14px'}),
                            dcc.Dropdown(
                                id='personas-hogar-dropdown',
                                options=[
                                    {'label': '1 a 2', 'value': '1 a 2'},
                                    {'label': '3 a 4', 'value': '3 a 4'},
                                    {'label': '5 a 6', 'value': '5 a 6'},
                                    {'label': '7 a 8', 'value': '7 a 8'},
                                    {'label': '9 o más', 'value': '9 o más'}
                                ],
                                value='3 a 4',
                                style={'width': '100%', 'marginBottom': '15px'}
                            )
                        ], style={'width': '100%', 'marginBottom': '10px'}),
                        
                        html.Div([
                            html.Label("Cuartos en el hogar:", style={'fontWeight': 'bold', 'fontSize': '14px'}),
                            dcc.Dropdown(
                                id='cuartos-hogar-dropdown',
                                options=[
                                    {'label': 'Uno', 'value': 'Uno'},
                                    {'label': 'Dos', 'value': 'Dos'},
                                    {'label': 'Tres', 'value': 'Tres'},
                                    {'label': 'Cuatro', 'value': 'Cuatro'},
                                    {'label': 'Cinco', 'value': 'Cinco'},
                                    {'label': 'Seis o más', 'value': 'Seis o mas'}
                                ],
                                value='Tres',
                                style={'width': '100%', 'marginBottom': '15px'}
                            )
                        ], style={'width': '100%', 'marginBottom': '10px'}),
                    ], style={'marginBottom': '15px', 'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '5px', 'boxShadow': '0 1px 3px rgba(0,0,0,0.1)', 'width': '100%'}),
                    
                    # Recursos tecnológicos e inglés
                    html.Div([
                        html.H5("Recursos y Nivel de Inglés", style={'textAlign': 'center', 'marginBottom': '20px', 'fontFamily': 'Roboto'}),
                        
                        html.Div([
                            html.Div([
                                html.Label("¿Tiene computador?", style={'fontWeight': 'bold', 'fontSize': '14px'}),
                                dcc.RadioItems(
                                    id='computador-radio',
                                    options=[
                                        {'label': 'Sí', 'value': 'Si'},
                                        {'label': 'No', 'value': 'No'}
                                    ],
                                    value='Si',
                                    inputStyle={'marginRight': '5px', 'marginLeft': '10px'}
                                )
                            ], style={'width': '50%', 'display': 'inline-block'}),
                            
                            html.Div([
                                html.Label("¿Tiene internet?", style={'fontWeight': 'bold', 'fontSize': '14px'}),
                                dcc.RadioItems(
                                    id='internet-radio',
                                    options=[
                                        {'label': 'Sí', 'value': 'Si'},
                                        {'label': 'No', 'value': 'No'}
                                    ],
                                    value='Si',
                                    inputStyle={'marginRight': '5px', 'marginLeft': '10px'}
                                )
                            ], style={'width': '50%', 'display': 'inline-block'})
                        ], style={'marginBottom': '15px', 'width': '100%'}),
                        
                        html.Div([
                            html.Div([
                                html.Label("¿Tiene lavadora?", style={'fontWeight': 'bold', 'fontSize': '14px'}),
                                dcc.RadioItems(
                                    id='lavadora-radio',
                                    options=[
                                        {'label': 'Sí', 'value': 'Si'},
                                        {'label': 'No', 'value': 'No'}
                                    ],
                                    value='Si',
                                    inputStyle={'marginRight': '5px', 'marginLeft': '10px'}
                                )
                            ], style={'width': '50%', 'display': 'inline-block'}),
                            
                            html.Div([
                                html.Label("¿Tiene automóvil?", style={'fontWeight': 'bold', 'fontSize': '14px'}),
                                dcc.RadioItems(
                                    id='auto-radio',
                                    options=[
                                        {'label': 'Sí', 'value': 'Si'},
                                        {'label': 'No', 'value': 'No'}
                                    ],
                                    value='No',
                                    inputStyle={'marginRight': '5px', 'marginLeft': '10px'}
                                )
                            ], style={'width': '50%', 'display': 'inline-block'})
                        ], style={'marginBottom': '15px', 'width': '100%'}),
                        
                        html.Div([
                            html.Label("Nivel de inglés:", style={'fontWeight': 'bold', 'fontSize': '14px'}),
                            dcc.Dropdown(
                                id='ingles-dropdown',
                                options=[
                                    {'label': 'A-', 'value': 'A-'},
                                    {'label': 'A1', 'value': 'A1'},
                                    {'label': 'A2', 'value': 'A2'},
                                    {'label': 'B1', 'value': 'B1'},
                                    {'label': 'B+', 'value': 'B+'}
                                ],
                                value='A2',
                                style={'width': '100%', 'marginBottom': '15px'}
                            )
                        ], style={'width': '100%', 'marginBottom': '10px'})
                    ], style={'marginBottom': '15px', 'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '5px', 'boxShadow': '0 1px 3px rgba(0,0,0,0.1)', 'width': '100%'}),
                    
                    # Botón de predicción
                    html.Button(
                        'Predecir Admisión', 
                        id='predict-button',
                        n_clicks=0,
                        style={
                            'backgroundColor': '#2c3e50',
                            'color': 'white',
                            'border': 'none',
                            'padding': '12px 20px',
                            'borderRadius': '5px',
                            'cursor': 'pointer',
                            'fontSize': '18px',
                            'fontWeight': 'bold',
                            'width': '100%',
                            'marginTop': '20px',
                            'boxShadow': '0 2px 4px rgba(0,0,0,0.2)'
                        }
                    ),
                    
                    html.Div(id='prediction-info', children=[
                        html.P("Complete el formulario y haga clic en 'Predecir Admisión' para obtener los resultados.",
                              style={'color': '#666', 'fontSize': '14px', 'fontStyle': 'italic', 'marginTop': '10px', 'textAlign': 'center'})
                    ])
                ], style={
                    'width': '48%',
                    'display': 'flex',
                    'flexDirection': 'column',
                    'alignItems': 'center',
                    'verticalAlign': 'top',
                    'backgroundColor': '#f8f9fa',
                    'padding': '20px',
                    'borderRadius': '10px',
                    'margin': '0 auto',
                    'boxSizing': 'border-box',
                    'minWidth': '500px',
                    'maxWidth': '650px',
                }),
                # Columna derecha: resultados
                html.Div([
                    html.H4("Resultados de la Predicción",
                          style={'color': '#2c3e50', 'fontFamily': 'Roboto', 'marginBottom': '20px', 'textAlign': 'center'}),
                    html.Div(id='prediction-output', style={'width': '100%'})
                ], style={
                    'width': '48%',
                    'display': 'flex',
                    'flexDirection': 'column',
                    'alignItems': 'center',
                    'verticalAlign': 'top',
                    'backgroundColor': '#f8f9fa',
                    'padding': '20px',
                    'borderRadius': '10px',
                    'position': 'sticky',
                    'top': '20px',
                    'alignSelf': 'flex-start',
                    'maxHeight': '80vh',
                    'minWidth': '350px',
                    'maxWidth': '500px',
                    'margin': '0 auto',
                })
            ], style={
                'display': 'flex',
                'flexWrap': 'wrap',
                'justifyContent': 'center',
                'alignItems': 'flex-start',
                'marginBottom': '30px',
                'gap': '32px'
            })
        ], style={'padding': '20px'})

# Callback para procesar la predicción
@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State('edad-input', 'value'),
     State('tipodoc-dropdown', 'value'),
     State('genero-radio', 'value'),
     State('depto-dropdown', 'value'),
     State('area-radio', 'value'),
     State('calendario-dropdown', 'value'),
     State('jornada-dropdown', 'value'),
     State('naturaleza-radio', 'value'),
     State('cole-genero-radio', 'value'),
     State('estrato-dropdown', 'value'),
     State('educacion-madre-dropdown', 'value'),
     State('educacion-padre-dropdown', 'value'),
     State('personas-hogar-dropdown', 'value'),
     State('cuartos-hogar-dropdown', 'value'),
     State('computador-radio', 'value'),
     State('internet-radio', 'value'),
     State('lavadora-radio', 'value'),
     State('auto-radio', 'value'),
     State('ingles-dropdown', 'value')]
)
def predict_admission(n_clicks, edad, tipodoc, genero, depto, area, calendario, jornada, naturaleza, cole_genero,
                   estrato, edu_madre, edu_padre, personas_hogar, cuartos_hogar, computador, internet, lavadora, auto, ingles):
    if n_clicks == 0:
        # Los resultados iniciales ya se muestran, no hace falta hacer nada
        raise exceptions.PreventUpdate
    
    try:
        # Crear un nuevo usuario con los datos del formulario
        usuario_prediccion = pd.DataFrame([{
            'estu_edad': float(edad) if edad is not None else 17.0,
            'estu_tipodocumento': tipodoc,
            'cole_area_ubicacion': area,
            'cole_calendario': calendario,
            'cole_genero': cole_genero,
            'cole_jornada': jornada,
            'cole_naturaleza': naturaleza,
            'estu_depto_reside': depto,
            'estu_genero': genero,
            'fami_cuartoshogar': cuartos_hogar,
            'fami_educacionmadre': edu_madre,
            'fami_educacionpadre': edu_padre,
            'fami_estratovivienda': estrato,
            'fami_personashogar': personas_hogar,
            'fami_tieneautomovil': auto,
            'fami_tienecomputador': computador,
            'fami_tieneinternet': internet,
            'fami_tienelavadora': lavadora,
            'desemp_ingles': ingles
        }])
        
        # Transformar los datos y predecir
        usuario_procesado = transformar_input(usuario_prediccion)
        prediccion = modelo.predict(usuario_procesado)[0][0]
        
        # Determinar si ingresa según la probabilidad
        ingresa = "Ingresa" if prediccion >= 0.7 else "No ingresa"
        color = "#2ecc71" if prediccion >= 0.7 else "#e74c3c"
        
        # Crear el resultado visual
        return html.Div([
            html.Div([
                html.H5("Resultados personalizados:", 
                       style={'textAlign': 'center', 'marginBottom': '15px', 'color': '#2c3e50'})
            ]),
            
            html.Div(
                html.Div([
                    html.H3(f"Probabilidad de ingreso: {prediccion * 100:.2f}%", 
                           style={'color': '#2c3e50', 'textAlign': 'center', 'marginBottom': '15px'}),
                       
                    html.H2(ingresa, style={
                        'fontSize': '36px', 
                        'fontWeight': 'bold',
                        'color': color,
                        'textAlign': 'center',
                        'margin': '10px 0'
                    }),
                    html.Div(
                        html.I(className="fas fa-check" if prediccion >= 0.5 else "fas fa-times", 
                              style={'fontSize': '56px', 'color': color}),
                        style={'textAlign': 'center', 'marginTop': '15px', 'marginBottom': '15px'}
                    ),
                    
                    # Información sobre los factores más influyentes (simulado)
                    html.Div([
                        html.H5("Factores destacados:", style={'marginTop': '15px', 'marginBottom': '10px', 'color': '#2c3e50', 'textAlign': 'center'}),
                        html.Ul([
                            html.Li(f"Nivel de inglés: {ingles}", style={'marginBottom': '5px'}),
                            html.Li(f"Estrato: {estrato}", style={'marginBottom': '5px'}),
                            html.Li(f"Educación de los padres", style={'marginBottom': '5px'})
                        ], style={'textAlign': 'left', 'paddingLeft': '30px', 'paddingRight': '10px'})
                    ], style={'backgroundColor': 'rgba(0,0,0,0.05)', 'borderRadius': '5px', 'padding': '10px', 'marginTop': '10px'})
                ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'justifyContent': 'center', 'width': '100%'}),
                style={
                    'padding': '30px', 
                    'backgroundColor': '#f8f9fa', 
                    'borderRadius': '10px',
                    'border': f'2px solid {color}',
                    'boxShadow': '0 4px 8px rgba(0,0,0,0.2)',
                    'display': 'flex',
                    'justifyContent': 'center',
                    'alignItems': 'center',
                    'marginTop': '20px',
                    'marginBottom': '20px'
                }
            )
        ], style={'padding': '10px'})
    
    except Exception as e:
        print(f"Error en la predicción: {e}")
        return html.Div([
            html.H5("Error en la predicción", style={'color': '#e74c3c', 'textAlign': 'center'}),
            html.P(f"Se produjo un error: {str(e)}",
                  style={'color': '#666', 'textAlign': 'center'})
        ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#f8d7da', 'borderRadius': '5px'})

if __name__ == '__main__':
    app.run(debug=True)