import os
from flask import Flask, request
from flask_restx import Resource, Api, fields
from dotenv import load_dotenv
from openai import AzureOpenAI
from flask_cors import CORS
from langchain.vectorstores.chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings

AZURE_OPENAI_ENDPOINT = os.environ['AZURE_OPENAI_ENDPOINT']
AZURE_OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_API_VERSION = os.environ["OPENAI_API_VERSION"]
AZURE_OPENAI_EMBED_DEPLOYEMENT = os.environ["AZURE_OPENAI_EMBED_DEPLOYEMENT"]
EMBEDDING_MODEL_NAME = os.environ['EMBEDDING_MODEL_NAME']
CHAT_COMPLETIONS_DEPLOYMENT_NAME = os.environ['CHAT_COMPLETIONS_DEPLOYMENT_NAME']

load_dotenv()

app = Flask(__name__)
CORS(app)
api = Api(app)

question_model = api.model('Question', {
    'question': fields.String(required=True, description='Ask the question')
})


@api.route('/qna')
class QnA(Resource):

    @api.expect(question_model)
    def post(self):
        """ 
        Answer the question using RAG 
        """
        data = request.json
        question = data.get('question')

        embeddings = AzureOpenAIEmbeddings(
            api_key=AZURE_OPENAI_API_KEY,
            model=EMBEDDING_MODEL_NAME,
            azure_deployment=AZURE_OPENAI_EMBED_DEPLOYEMENT
        )

        persist_directory = 'db'

        client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version="2023-03-15-preview",
            azure_deployment=CHAT_COMPLETIONS_DEPLOYMENT_NAME
        )

        vectordb = Chroma(persist_directory=persist_directory,
                          embedding_function=embeddings)

        results = vectordb.similarity_search(question, k=3)

        # Combine the document chunks into a single context string
        context = "\n".join([doc.page_content for doc in results])

        msg = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"}
        ]

        # Generate the response using the context
        response = client.chat.completions.create(
            model=os.getenv('CHAT_COMPLETIONS_DEPLOYMENT_NAME'),
            messages=msg,
            max_tokens=500,
            temperature=0.7
        )

        answer = response.choices[0].message.content
        return {
            'Question': question,
            'Answer': answer
        }


if __name__ == '__main__':
    app.run(debug=True, host="localhost", port=5000)
