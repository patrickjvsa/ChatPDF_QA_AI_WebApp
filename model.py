import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os
import pandas as pd

# necesario (al menos en mi pc) para conectarse a la api de openai sin errores
os.environ['REQUESTS_CA_BUNDLE'] = 'openai_web_certificate.crt'

# para leer un archivo
def load_document(file):

    import os
    name, extension = os.path.splitext(file) # separa el nombre del archivo en nombre y formato

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print (f'Cargando {file}...')
        pdfloader = PyPDFLoader(file)
        data = pdfloader.load()

    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print (f'Cargando {file}...')
        docxloader = Docx2txtLoader(file)
        data = docxloader.load()

    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        print (f'Cargando {file}...')
        textloader = TextLoader(file)
        data = textloader.load()

    else:
        print('Formato no soportado')

    print ('Proceso finalizado')

    return data

# dividir en chunks
def chunk_data(data, chunk_size=256, chunk_overlap=20):

    from langchain.text_splitter import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)

    return chunks

# calcular costo por token
def calculate_embedding_cost(texts):

    import tiktoken

    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    embedding_cost = (total_tokens / 1000 * 0.0001)

    return total_tokens, embedding_cost

# para crear embeddings a partir de los chunks de texto
def create_embeddings(index_name):

    from langchain.vectorstores import Chroma
    from langchain.embeddings.openai import OpenAIEmbeddings

    persist_directory = 'db'    # la carpeta donde se guarda el vector db

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory)
    vector_store.persist()

    return vector_store

######################################################################################################################

# pregunta y respuesta

# modelos sencillos: askLIGHT detecta el problema en base a los retrieved chunks, inferLIGHT infiere causas posibles para el
#                    problema detectado por el primero.

def ask(vector_store, query, context_for_query, k=3, temperature=1):

    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI
    from langchain.schema import(HumanMessage, SystemMessage)

    # query del usuario

    # query = 'instrucciones' + query

    # definir el modelo
    openai_api_key = api_key
    llm = ChatOpenAI(model='gpt-3.5-turbo',
                     temperature=temperature,
                     openai_api_key=openai_api_key)

    llm(messages = [
        SystemMessage(content=context_for_query),
        HumanMessage(content=query)])


    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})

    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type='stuff')

    return chain.run(query)


def askPRO(vector_store, query, context_for_query, edited_df, k=3, temperature=1):

    from langchain.chains.router import MultiRetrievalQAChain
    from langchain.chat_models import ChatOpenAI
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.schema import(HumanMessage, SystemMessage)

    # obtengo el df que está en la interfaz y lo integro como fuente de información (retriever)

    edited_df_str = edited_df.to_dict()
    edited_df_str = str(edited_df_str)

    metric_data = Chroma.from_texts(edited_df_str, OpenAIEmbeddings()).as_retriever()

    # sumo el db que contiene el pdf como retriever

    vs_retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})

    # query dentro de una plantilla

    #query = "" + query

    openai_api_key = api_key

    # modelo de lenguaje es definido

    llm = ChatOpenAI(model='gpt-3.5-turbo',
                     temperature=temperature,
                     openai_api_key=openai_api_key
                     )

    llm(messages = [
        SystemMessage(content=context_for_query),
        HumanMessage(content=query)])

    #se definen ambas fuentes de info y sus descripciones

    retriever_infos = [
        {
            "name": "Source 1",
            "description": "Good for...",
            "retriever": retriever_1
        },
        {
            "name": "Source 2",
            "description": "Good for ?",
            "retriever": vs_retriever
        },
    ]

    #se une en una cadena y se corre

    chain = MultiRetrievalQAChain.from_retrievers(llm=llm, retriever_infos=retriever_infos, verbose=True)

    return chain.run(query)


# limpia el historial de la webapp de Streamlit
def clear_history():

    if 'history' in st.session_state:
        del st.session_state['history']

    # necesario en mi pc para que se conecte a la api de openai
    os.environ['REQUESTS_CA_BUNDLE'] = 'openai_web_certificate.crt'

    # cargar api_key
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)

    st.subheader('langchain + chromadb app 🤖')

    with st.sidebar:
        # text_input para la api_key

        env_api_key = os.getenv('OPENAI_API_KEY')


        # si reescribes la clave
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key

        # widget para subir archivo
        uploaded_file = st.file_uploader('Subir un archivo nuevo:', type=['pdf', 'docx', 'txt'])

        # chunk size
        chunk_size = st.number_input('Embedding chunk size:', min_value=100, max_value=2048, value=512, on_change=clear_history)

        # añadir
        add_data = st.button('Añadir', on_click=clear_history)

        # si no ha apretado el boton de añadir pero el vector db del archivo si está cargado
        if not add_data:
            if  os.path.exists('db'):
                st.success('Ya existe un archivo en memoria. Añadir otro lo sobreescribirá.')

        # se sube un archivo
        if uploaded_file and add_data:
            with st.spinner('Leyendo...'):
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./', uploaded_file.name)

                with open(file_name, 'wb') as f:
                    f.write(bytes_data)

                data = load_document(file_name)
                chunks = chunk_data(data, chunk_size=chunk_size)

                tokens, embedding_cost = calculate_embedding_cost(chunks)
                st.write(f'Costo de embedding, model ada002: ${embedding_cost:.4f}')

                # crear embedding y guardar en chroma
                vector_store = create_embeddings(chunks)

                # guardar el db en la sesion de Streamlit
                st.session_state.vs = vector_store
                st.success('Nuevo archivo cargado.')

        st.divider()

        # k number del retriever
        k = st.slider('Retriever k number:', min_value=1, max_value=10, value=3, on_change=clear_history)

        # temperatura del llm
        temperature = st.slider('LLM temperature:', min_value=0.0, max_value=2.0, value=0.4, on_change=clear_history)


    context_input = st.text_area('Tu rol es..')

    # query del usuario
    q = st.text_area('Inserte pregunta u observación...',
                                         height=160)


    is_button_pressed = st.button('Generar respuesta')

    # si el usuario apretó generar respuesta
    if q and is_button_pressed:

        # si los datos del archivo no han sido cargados en la sesion de Streamlit
        if 'vs' not in st.session_state:
            # cargar
            embedding = OpenAIEmbeddings(openai_api_key=api_key)
            vector_store = Chroma(persist_directory="db", embedding_function=embedding)
            st.session_state.vs = vector_store

        # si los datos estan cargados
        if 'vs' in st.session_state:
            with st.spinner('Pensando...'):

                vector_store = st.session_state.vs

                # llm1
                answer1 = ask(vector_store, q, context_input, k, temperature)
                result1 = vector_store.similarity_search(q)

                st.text_area('Problema detectado: ', value=answer1, height=300)

                with st.expander('Fuentes (chunks retrieved)'):

                st.divider()

                # si no hay historial, crearlo
                if 'history' not in st.session_state:
                    st.session_state.history = ''

                # pregunta actual
                value = f'Pregunta: {q} \nRespuesta: {answer1}'

                st.session_state.history = f'{value} \n {"-" * 20} \n {st.session_state.history}'
                h = st.session_state.history

                # historial
                with st.expander('Historial'):
                    st.text_area(label='Historial', value=h, key='history', height=400)

# para correr la app en terminal: streamlit run ./model.py

