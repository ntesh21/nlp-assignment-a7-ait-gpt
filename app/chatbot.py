import os
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from langchain import PromptTemplate
from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM
from transformers import BitsAndBytesConfig
# from langchain import HuggingFacePipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain
import torch
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains import LLMChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory


def get_prompt():
    prompt_template = """
        "Welcome to the Friendly and Assistive Informational Bot for Asian Institute of Technology!
        How can I assist you today? Whether you need information about programs, admission procedures, campus facilities,
        or anything else related to AIT, I'm here to help. Just ask away, and I'll provide you with all the necessary details
        to make your experience at AIT as smooth and enjoyable as possible."
            {context}
        Question: {question}
        Answer:
        """.strip()

    PROMPT = PromptTemplate.from_template(
    template = prompt_template
    )
    return PROMPT


def get_embeddings():
    model_name = 'hkunlp/instructor-base'
    embedding_model = HuggingFaceInstructEmbeddings(
        model_name = model_name,
        model_kwargs = {"device" : device}
    )
    return embedding_model


def get_retriever():
    #calling vector from local
    vector_path = '../vector-store'
    db_file_name = 'nlp_stanford'
    embedding_model = get_embeddings()
    vectordb = FAISS.load_local(
        folder_path = os.path.join(vector_path, db_file_name),
        embeddings = embedding_model,
        index_name = 'nlp' #default index
    )
    #ready to use
    retriever = vectordb.as_retriever()  
    return retriever

def get_memory():
    history = ChatMessageHistory()
    history.add_user_message('hi')
    history.add_ai_message('Whats up?')
    history.add_user_message('How are you')
    history.add_ai_message('I\'m quite good. How about you?')
    memory = ConversationBufferMemory()
    memory.save_context({'input':'hi'}, {'output':'What\'s up?'})
    memory.save_context({"input":'How are you?'},{'output': 'I\'m quite good. How about you?'})
    memory.load_memory_variables({})
    memory = ConversationBufferMemory(return_messages = True)
    memory.save_context({'input':'hi'}, {'output':'What\'s up?'})
    memory.save_context({"input":'How are you?'},{'output': 'I\'m quite good. How about you?'})
    memory.load_memory_variables({})
    memory = ConversationBufferWindowMemory(k=1)
    memory.save_context({'input':'hi'}, {'output':'What\'s up?'})
    memory.save_context({"input":'How are you?'},{'output': 'I\'m quite good. How about you?'})
    memory.load_memory_variables({})
    return memory


def get_pipeline():
    model_id = '../models/fastchat-t5-3b-v1.0/'

    tokenizer = AutoTokenizer.from_pretrained(
        model_id)

    tokenizer.pad_token_id = tokenizer.eos_token_id

    bitsandbyte_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_quant_type = "nf4",
        bnb_4bit_compute_dtype = torch.float16,
        bnb_4bit_use_double_quant = True
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        quantization_config = bitsandbyte_config, #caution Nvidia
        device_map = 'auto',
        load_in_8bit = True
    )

    pipe = pipeline(
        task="text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens = 4,
        model_kwargs = {
            "temperature" : 0,
            "repetition_penalty": 1.5
        }   
    )

    llm = HuggingFacePipeline(pipeline = pipe)
    return llm

def get_doc_chain(llm, PROMPT):
    doc_chain = load_qa_chain(
    llm = llm,
    chain_type = 'stuff',
    prompt = PROMPT,
    verbose = True
    )
    return doc_chain


def chatbot(query):
    PROMPT = get_prompt()
    retriever = get_retriever()
    memory = get_memory()
    llm = get_pipeline()    
    question_generator = LLMChain(
    llm = llm,
    prompt = CONDENSE_QUESTION_PROMPT,
    verbose = True
    )
    doc_chain = get_doc_chain(llm, PROMPT)
    
    chain = ConversationalRetrievalChain(
    retriever=retriever,
    question_generator=question_generator,
    combine_docs_chain=doc_chain,
    return_source_documents=True,
    memory=memory,
    verbose=True,
    get_chat_history=lambda h : h
    )
    answer = chain({"question":query})
    answer = answer['answer'].replace('<pad>', '').strip()
    return answer

