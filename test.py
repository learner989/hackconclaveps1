from openai import AzureOpenAI
import os
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain_community.document_transformers.embeddings_redundant_filter import EmbeddingsRedundantFilter
from langchain.vectorstores.chroma import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings


load_dotenv()

AZURE_OPENAI_ENDPOINT = os.environ['AZURE_OPENAI_ENDPOINT']
AZURE_OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_API_VERSION = os.environ["OPENAI_API_VERSION"]
AZURE_OPENAI_EMBED_DEPLOYEMENT = os.environ["AZURE_OPENAI_EMBED_DEPLOYEMENT"]
EMBEDDING_MODEL_NAME = os.environ['EMBEDDING_MODEL_NAME']
CHAT_COMPLETIONS_DEPLOYMENT_NAME = os.environ['CHAT_COMPLETIONS_DEPLOYMENT_NAME']
if not AZURE_OPENAI_API_KEY:
    raise ValueError("AZURE_OPENAI_API_KEY environment variable is not set")

loader = PyPDFLoader(".\data\insurance.pdf")
documents = loader.load_and_split()

# Split the documents into chunks
text_splitter = CharacterTextSplitter(
    separator='\n',
    chunk_size=1024,
    chunk_overlap=128
)
texts = text_splitter.split_documents(documents)


# Initialize OpenAI embeddings and Define the directory to persist the Chroma database
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
filtered_texts = redundant_filter.transform_documents(documents=texts)


# Create the Chroma vector store and persist the database
try:
    vectordb = Chroma.from_documents(
        documents=filtered_texts,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectordb.persist()
except Exception as e:
    print(f"An error occurred: {e}")


load_dotenv()

client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2023-03-15-preview",
    azure_deployment=CHAT_COMPLETIONS_DEPLOYMENT_NAME
)

vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embeddings)


def answer_question(query, vectordb):
    # Retrieve the most relevant document chunks
    results = vectordb.similarity_search(query, k=3)

    # Combine the document chunks into a single context string
    context = "\n".join([doc.page_content for doc in results])

    msg = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"}
    ]

    # Generate the response using the context
    response = client.chat.completions.create(
        model=os.getenv('CHAT_COMPLETIONS_DEPLOYMENT_NAME'),
        messages=msg,
        max_tokens=500,
        temperature=0.7
    )

    # Extract and return the assistant's response
    return response.choices[0].message.content


# query = "How many days of annual leave are provided?"
# response = answer_question(query, vectordb)
# print(response)

# print('-----------------------------------------------------------------------')
# query = "What does \"Emergency Care\" mean according to the policy definitions?"
# response = answer_question(query, vectordb)
# print('question :'+query)
# print('response :'+response)
# print('-----------------------------------------------------------------------')
# query = "What is the premium for a 35-year-old individual under the Total Health Plan with a sum insured of 5 lakhs on an individual basis?"
# response = answer_question(query, vectordb)
# print('question :'+query)
# print('response :'+response)
# print('-----------------------------------------------------------------------')
# query = "What is the total premium for all family members if they are covered under a single policy?"
# response = answer_question(query, vectordb)
# print('question :'+query)
# print('response :'+response)
# print('-----------------------------------------------------------------------')
