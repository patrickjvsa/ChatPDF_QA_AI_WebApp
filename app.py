import streamlit as st
from dotenv import load_dotenv
import os
import text_modules 

def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']

if __name__ == "__main__":

    st.subheader('manoloGPT 🤖')
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')

    with st.sidebar:

        # text_input para la api_key
        api_key = st.text_input('OpenAI API key:', value = api_key ,type='password', on_change=clear_history)
        # si reescribes la clave
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key
        # widget para subir archivo
        uploaded_file = st.file_uploader('Subir un archivo nuevo:', type=['pdf', 'docx', 'txt'])
        # boton de añadir
        add_data_button = st.button('Añadir', on_click=clear_history)
        # si no ha apretado el boton de añadir pero el vector db del archivo si está cargado
        if not add_data_button:
            if  os.path.exists('db'):
                st.success('Ya existe un archivo en memoria. Añadir otro lo sobreescribirá.')

        # se sube un archivo
        if uploaded_file and add_data_button:
            with st.spinner('Procesando...'):
                
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./', uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)

                # extraer la extension del archivo
                print("Extracting extension...")
                extension = text_modules.ExtractDocumentFormat.extract_extension(file_path=file_name)
                # cargar el archivo
                print("Loading document...")
                data = text_modules.LoadDocument.load_document(file_path=file_name, file_extension=extension)
                # separar en chunks
                print("Chunking document...")
                chunks = text_modules.ChunkData.chunk_data(data=data)
                # instanciar embedding model
                print("Creating embedding model...")
                embedding_model = text_modules.OpenAIEmbeddings.create_embedding_model()
                # crear vector store, generar y guardar embeddings
                print("Creating vector store...")
                vector_store = text_modules.VectorStoreChroma.create_vector_store(chunks, embedding_model)
                # guardar el db en la sesion de Streamlit
                print("Saving vector store...")
                st.session_state.vs = vector_store
                st.success('Nuevo archivo cargado.')

        st.divider()
        # k number del retriever
        k = st.slider('Retriever k number:', min_value=3, max_value=15, value=7, on_change=clear_history)
    
    #rol del chatbot
    system_message = st.text_area('Tu rol es..')
    # query del usuario
    query = st.text_area('Inserte pregunta...', height=160)
    # boton de generar respuesta
    is_button_pressed = st.button('Generar respuesta')

    # si el usuario apretó generar respuesta
    if query and is_button_pressed:

        # si los datos del archivo no han sido cargados en la sesion de Streamlit
        if 'vs' not in st.session_state:

            # instanciar embedding model
            embedding_model = text_modules.OpenAIEmbeddings.create_embedding_model()
            # crear vector store, generar y guardar embeddings
            vector_store = text_modules.VectorStoreChroma.create_vector_store(chunks, embedding_model)
            # guardar el db en la sesion de Streamlit
            st.session_state.vs = vector_store

        # si los datos estan cargados
        if 'vs' in st.session_state:
            with st.spinner('Pensando...'):
                
                # recuperar vector store
                vector_store = st.session_state.vs
                # instanciar chatbot
                print("Creating chatbot...")
                chatbot = text_modules.OpenAIChat.create_chat_model(system_message=system_message)
                # crear cadena de respuesta
                print("Creating answer...")
                answer = text_modules.SimpleQuestionAnswer.ask_and_get_answer(query=query, vector_store=vector_store, llm=chatbot, k=k)
                print("Gathering sources...")
                sources = vector_store.similarity_search(query)

                st.text_area('Respuesta: ', value=answer, height=300)

                with st.expander('Fuentes: '):
                        for source in sources:
                            st.write(source.page_content + '\n')

                st.divider()

                # si no hay historial, crearlo
                if 'history' not in st.session_state:
                    st.session_state.history = ''

                # pregunta actual
                value = f'Pregunta: {query} \nRespuesta: {answer}'

                st.session_state.history = f'{value} \n {"-" * 20} \n {st.session_state.history}'
                h = st.session_state.history

                # historial
                with st.expander('Historial'):
                    st.text_area(label='Historial', value=h, key='history', height=400)

# para correr la app en terminal: streamlit run ./model.py