# WebApp de Generación de Respuestas con GPT

Esta es una aplicación web que utiliza la biblioteca `langchain` y `streamlit` para generar respuestas a partir de archivos de texto utilizando modelos de generación de texto. 
La aplicación es una interfaz fácil de usar que permite a los usuarios cargar archivos de texto y obtener respuestas generadas por GPT.

## Características

- Generación de respuestas a partir de archivos de texto (txt, pdf, docx).
- Interfaz de usuario intuitiva y amigable.
- Personalización de parámetros de generación, como la longitud de la respuesta.
- Integración con GPT u otros modelos para respuestas de alta calidad.

## Requisitos

Asegúrate de tener instalado Python, y las bibliotecas necesarias utilizando pip:

```bash
pip install -r requirements.txt
```

## Uso

1. Ejecuta la aplicación web utilizando Streamlit.

```bash
streamlit run app.py
```

2. Carga un archivo de texto y configura los parámetros de generación según tus necesidades.

3. Haz clic en el botón "Generar Respuesta" y obtén respuestas generadas por el modelo.

## Lógica y personalización

El código incluye un modúlo escrito para la API de OpenAI, sin embargo, es posible añadir módulos al archivo text_modules.py con otras herramientas de utilidad.
Además, puedes personalizar la configuración de la generación de respuestas editando el archivo `app.py`.
