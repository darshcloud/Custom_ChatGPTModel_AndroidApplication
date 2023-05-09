import os

from flask import Flask, request
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, LLMPredictor, ServiceContext, StorageContext, \
    load_index_from_storage
from langchain import OpenAI

app = Flask(__name__)

os.environ["OPENAI_API_KEY"] = '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'


def construct_index(directory_path):
    num_outputs = 512

    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.7, model_name="text-davinci-003", max_tokens=num_outputs))

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

    docs = SimpleDirectoryReader(directory_path).load_data()

    index = GPTVectorStoreIndex.from_documents(docs, service_context=service_context)

    index.storage_context.persist()

    return index


@app.route('/chatbot', methods=['POST'])
def chatbot():
    input_text = request.json['input_text']

    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine()
    response = query_engine.query(input_text)
    return {'response': response.response}


if __name__ == '__main__':
    index = construct_index("docs")
    app.run()
