import os
from flask import Flask, request
from flask_restx import Resource, Api, fields
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain_community.document_transformers.embeddings_redundant_filter import EmbeddingsRedundantFilter

# Load environment variables
load_dotenv()

# variables
AZURE_OPENAI_ENDPOINT = os.environ['AZURE_OPENAI_ENDPOINT']
AZURE_OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_API_VERSION = os.environ["OPENAI_API_VERSION"]
AZURE_OPENAI_EMBED_DEPLOYEMENT = os.environ["AZURE_OPENAI_EMBED_DEPLOYEMENT"]
EMBEDDING_MODEL_NAME = os.environ['EMBEDDING_MODEL_NAME']
CHAT_COMPLETIONS_DEPLOYMENT_NAME = os.environ['CHAT_COMPLETIONS_DEPLOYMENT_NAME']

app = Flask(__name__)
CORS(app)
api = Api(app)


@api.route('/docprocess')
class DocumentProcess(Resource):
    def post(self):
        """
        Process the PDF document
        """
        # file = request.files['file']
        # file_path = os.path.join('.', 'uploaded_file.pdf')
        # file.save(file_path)

        # check if file already exists
        file = request.files['file']
        file_path = os.path.join('data', file.filename)
        if not os.path.exists(file_path):
            file.save(file_path)
        else:
            return {'message': 'Document Already Processed!'}
        print(file_path)

        try:
            # Read the PDF document, handle process document
            loader = PyPDFLoader(file_path)
            documents = loader.load_and_split()

            text_splitter = CharacterTextSplitter(
                separator='\n',
                chunk_size=1000,
                chunk_overlap=200
            )
            texts = text_splitter.split_documents(documents)

            # Create embeddings for the chunks
            embeddings = AzureOpenAIEmbeddings(
                api_key=AZURE_OPENAI_API_KEY,
                model=EMBEDDING_MODEL_NAME,
                azure_deployment=AZURE_OPENAI_EMBED_DEPLOYEMENT
            )

            persist_directory = 'db'

            # Initialize the EmbeddingsRedundantFilter
            redundant_filter = EmbeddingsRedundantFilter(
                embeddings=embeddings,
                similarity_threshold=0.95
            )

            # Apply the filter to the documents
            filtered_texts = redundant_filter.transform_documents(
                documents=texts)

            # Store the embeddings in the database
            vectordb = Chroma.from_documents(
                documents=filtered_texts,
                embedding=embeddings,
                persist_directory=persist_directory
            )
            vectordb.persist()

            return {'message': 'Document Processed Successfully!'}
        except Exception as e:
            return {'message': f"An error occurred: {e}"}, 500


if __name__ == '__main__':
    app.run(debug=True, host="localhost", port=5001)
