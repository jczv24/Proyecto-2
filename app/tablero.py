import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import json

# Inicializar la aplicación Dash
app = dash.Dash(__name__,
                external_stylesheets=[
                    'https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap',
                    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css'
                ])

app.title = "Proyecto 2"
server = app.server

# --- Carga y Preparación Inicial de Datos ---
datos_limpios = pd.read_csv('model_data.csv')
datos_limpios['desemp'] = pd.qcut(datos_limpios['punt_global'], q=3, labels=['Bajo', 'Medio', 'Alto'])

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
        # --- Cálculo de rangos para 'desemp' ---
        try:
            # El argumento duplicates='drop' es importante si hay muchos valores idénticos en los cuantiles
            _, binedges = pd.qcut(datos_limpios['punt_global'], q=3, labels=False, retbins=True, duplicates='drop')
            range_bajo_txt = f"Bajo: [{binedges[0]:.0f} - {binedges[1]:.0f}]"
            range_medio_txt = f"Medio: ({binedges[1]:.0f} - {binedges[2]:.0f}]"
            range_alto_txt = f"Alto: ({binedges[2]:.0f} - {binedges[3]:.0f}]"
            rangos_desemp_text = f"{range_bajo_txt} | {range_medio_txt} | {range_alto_txt}"
        except Exception as e:
            print(f"Error calculando rangos de desempeño: {e}")
            # Initialize texts to a default error message if calculation fails for any reason
            range_bajo_txt = "Bajo: Rango no disponible"
            range_medio_txt = "Medio: Rango no disponible"
            range_alto_txt = "Alto: Rango no disponible"
            rangos_desemp_text = "Rangos no disponibles" 

        performance_counts = datos_limpios['desemp'].value_counts().reset_index()
        performance_counts.columns = ['Nivel de Desempeño', 'Cantidad']
        fig_performance = px.histogram(
            datos_limpios, 
            x='punt_global',
            color='desemp',
            title='Distribución de Puntajes Globales por Nivel de Desempeño',
            labels={'punt_global': 'Puntaje Global', 'count': 'Cantidad de Estudiantes', 'desemp': 'Nivel de Desempeño'},
            color_discrete_map={'Bajo': '#e74c3c', 'Medio': '#f39c12', 'Alto': '#2ecc71'},
            nbins=30,
            opacity=0.8
        )
        fig_performance.update_layout(
            title_font_family="Roboto", 
            font_family="Roboto", 
            title_x=0.5, 
            legend_title_text='Nivel',
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

        fig_box_punt_global = px.box(datos_limpios, y="punt_global",
                                     title="Distribución de Puntaje Global",
                                     points="outliers") # Muestra outliers
        fig_box_punt_global.update_layout(
            title_font_family="Roboto", font_family="Roboto", title_x=0.5,
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
        )
        # --- FIN NUEVO ---

        # Relación entre Estrato Socioeconómico y Desempeño
        stratum_performance = datos_limpios.groupby(['fami_estratovivienda', 'desemp']).size().reset_index(name='Cantidad')
        stratum_order = sorted(datos_limpios['fami_estratovivienda'].unique(), key=lambda x: int(x.split(' ')[1]) if 'Estrato' in x else 0)
        stratum_performance['fami_estratovivienda'] = pd.Categorical(stratum_performance['fami_estratovivienda'], categories=stratum_order, ordered=True)
        stratum_performance = stratum_performance.sort_values(['fami_estratovivienda', 'desemp'])
        fig_stratum_performance = px.bar(stratum_performance,
                                         x='fami_estratovivienda', y='Cantidad', color='desemp', barmode='group',
                                         title='Desempeño por Estrato Socioeconómico',
                                         labels={'fami_estratovivienda': 'Estrato Socioeconómico', 'desemp': 'Nivel de Desempeño'},
                                         color_discrete_map={'Bajo': '#e74c3c', 'Medio': '#f39c12', 'Alto': '#2ecc71'},
                                         text_auto=True)
        fig_stratum_performance.update_layout(
            title_font_family="Roboto", font_family="Roboto", title_x=0.5, legend_title_text='Nivel',
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
        )
        fig_stratum_performance.update_traces(textposition='outside')

        # Definición del mapa (ya existe en tu código)
        fig_mapa_colombia = None 
        mapa_component_html = None # Para el contenido del mapa o error
        if geojson_colombia:
            df_para_mapa = datos_limpios[datos_limpios['depto_estandarizado'] != 'EXTRANJERO'].copy()
            promedio_por_depto_mapa = df_para_mapa.groupby('depto_estandarizado')['punt_global'].mean().reset_index()
            desemp_counts = df_para_mapa.groupby(['depto_estandarizado', 'desemp']).size().reset_index(name='cantidad')
            total_por_depto = df_para_mapa.groupby('depto_estandarizado').size().reset_index(name='total_estudiantes')
            desemp_counts = pd.merge(desemp_counts, total_por_depto, on='depto_estandarizado')
            desemp_counts['porcentaje'] = (desemp_counts['cantidad'] / desemp_counts['total_estudiantes']) * 100
            porcentajes_pivot = desemp_counts.pivot_table(index='depto_estandarizado', columns='desemp', values='porcentaje', fill_value=0).reset_index()
            porcentajes_pivot = porcentajes_pivot.rename(columns={'Alto': 'Alto (%)','Medio': 'Medio (%)','Bajo': 'Bajo (%)'})
            map_data_final = pd.merge(promedio_por_depto_mapa, porcentajes_pivot, on='depto_estandarizado', how='left')
            for col_perc in ['Alto (%)', 'Medio (%)', 'Bajo (%)']:
                if col_perc not in map_data_final.columns:
                    map_data_final[col_perc] = 0.0
            map_data_final = map_data_final.fillna(0)
            map_data_final['Alto (%)'] = map_data_final['Alto (%)'].round(2)
            map_data_final['Medio (%)'] = map_data_final['Medio (%)'].round(2)
            map_data_final['Bajo (%)'] = map_data_final['Bajo (%)'].round(2)
            fig_mapa_colombia = px.choropleth(
                map_data_final, geojson=geojson_colombia, locations='depto_estandarizado',
                featureidkey='properties.NOMBRE_DPT', color='punt_global',
                color_continuous_scale="Viridis", scope="south america", hover_name='depto_estandarizado',
                hover_data={'punt_global': ':.2f','Alto (%)': ':.2f%','Medio (%)': ':.2f%','Bajo (%)': ':.2f%'},
                labels={'punt_global':'Puntaje Global Promedio', 'depto_estandarizado': 'Departamento'},
                title='Puntaje Global Promedio y Distribución de Desempeño por Departamento'
            )
            fig_mapa_colombia.update_geos(fitbounds="locations", visible=False)
            fig_mapa_colombia.update_layout(margin={"r":0,"t":40,"l":0,"b":0}, title_x=0.5, title_font_family="Roboto", font_family="Roboto", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            mapa_component_html = html.Div([dcc.Graph(id='mapa-colombia-puntajes', figure=fig_mapa_colombia)], style={'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'backgroundColor': '#f9f9f9', 'marginBottom': '20px'})
        else:
            mapa_component_html = html.P("Error al cargar los datos geográficos para el mapa.", style={'color':'red', 'textAlign':'center', 'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'backgroundColor': '#f9f9f9', 'marginBottom': '20px'})

        # Definición del gráfico de categorías paralelas (ya existe en tu código)
        sample_fraction = 0.1
        if len(datos_limpios) * sample_fraction < 1000:
             df_sample_parcats = datos_limpios.sample(n=min(len(datos_limpios), 1000), random_state=1).copy()
        else:
             df_sample_parcats = datos_limpios.sample(frac=sample_fraction, random_state=1).copy()
        desemp_mapping = {'Bajo': 0, 'Medio': 1, 'Alto': 2}
        df_sample_parcats['desemp_code'] = df_sample_parcats['desemp'].map(desemp_mapping)
        dimensions_parcats = ['fami_tienecomputador','fami_tieneinternet','desemp']
        custom_labels = {'fami_tienecomputador': 'Familia Tiene Computador','fami_tieneinternet': 'Familia Tiene Internet','desemp': 'Desempeño General','desemp_code': 'Nivel de Desempeño (Color)'}
        fig_parallel_categories = px.parallel_categories(
            df_sample_parcats, dimensions=dimensions_parcats, color="desemp_code",
            color_continuous_scale=[(0, "#e74c3c"), (0.5, "#f39c12"), (1, "#2ecc71")],
            labels=custom_labels
        )
        fig_parallel_categories.update_layout(title='Relación entre Factores Socioeconómicos y Desempeño',title_x=0.5,font_family="Roboto",margin=dict(l=50, r=50, t=80, b=50))
        
        # Ensamblar los hijos de la pestaña 1 en el orden deseado
        children_tab1 = [
            html.H3("Análisis Descriptivo de Datos - Examen Saber Pro ICFES",
                   style={
                       'color': '#34495e', 'fontFamily': 'Roboto', 'textAlign': 'center',
                       'marginBottom': '30px', 'borderBottom': '1px solid #ecf0f1', 'paddingBottom': '10px'
                   }),
            
            html.Div([
                html.Div([
                    html.H5("Distribución de Puntajes Globales por Nivel de Desempeño", style={'textAlign': 'center', 'fontFamily': 'Roboto', 'color': '#34495e'}),
                    html.Div([
                        html.P(range_bajo_txt, style={'textAlign': 'center', 'fontFamily': 'Roboto', 'fontSize': '14px', 'color': '#555', 'marginBottom': '3px'}),
                        html.P(range_medio_txt, style={'textAlign': 'center', 'fontFamily': 'Roboto', 'fontSize': '14px', 'color': '#555', 'marginBottom': '3px'}),
                        html.P(range_alto_txt, style={'textAlign': 'center', 'fontFamily': 'Roboto', 'fontSize': '14px', 'color': '#555', 'marginBottom': '15px'})
                    ], style={'marginTop': '5px'}), # Div para agrupar los rangos
                    dcc.Graph(figure=fig_performance)
                ], style={'width': '50%', 'display': 'inline-block', 'padding': '10px', 'verticalAlign': 'top', 'border': '1px solid #ddd', 'borderRadius': '5px', 'backgroundColor': '#f9f9f9', 'marginRight': '10px'}),
                html.Div([
                    html.H5("Estadísticas Clave de Puntaje Global", style={'textAlign': 'center', 'fontFamily': 'Roboto', 'color': '#34495e', 'marginBottom':'15px'}),
                    html.Div([
                        html.P([html.Strong("Promedio: "), f"{mean_score:.2f}"], style={'textAlign': 'center', 'fontFamily': 'Roboto', 'fontSize':'16px', 'marginBottom':'5px'}),
                        html.P([html.Strong("Desviación Estándar: "), f"{std_score:.2f}"], style={'textAlign': 'center', 'fontFamily': 'Roboto', 'fontSize':'16px', 'marginBottom':'20px'}),
                    ]),
                    dcc.Graph(figure=fig_box_punt_global) # Confirmando que es fig_box_punt_global como en el archivo del usuario
                ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px', 'verticalAlign': 'top', 'border': '1px solid #ddd', 'borderRadius': '5px', 'backgroundColor': '#f9f9f9'})
            ], style={'display': 'flex', 'marginBottom': '20px'}),
            
            mapa_component_html, # Mapa en segunda posición de contenido

            html.Div([
                dcc.Graph(figure=fig_stratum_performance)
            ], style={'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'backgroundColor': '#f9f9f9', 'marginBottom': '20px'}),
            
            html.Div([
                dcc.Graph(id='parallel-categories-plot', figure=fig_parallel_categories)
            ], style={'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'backgroundColor': '#f9f9f9', 'marginBottom': '20px'}),
            
            html.P("Universidad de los Andes - Ingenieria industrial",
                  style={
                      'color': '#666', 'fontFamily': 'Roboto', 'fontSize': '16px', 'textAlign': 'center'
                  })
        ]
        return html.Div(children_tab1)

    elif tab == 'tab-2':
        return html.Div([
            html.H2("Modelo Predictivo",
                   style={
                       'color': '#2c3e50', 'fontFamily': 'Roboto', 'marginBottom': '20px'
                   }),
            html.P("Aquí irá el contenido del modelo predictivo...",
                  style={
                      'color': '#666', 'fontFamily': 'Roboto', 'fontSize': '16px'
                  })
        ])

if __name__ == '__main__':
    app.run(debug=True)