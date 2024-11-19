import argparse
import re
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import torch
import transformers
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from logparser import LogParser
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema.runnable import RunnablePassthrough
import warnings
import time


warnings.filterwarnings("ignore")

# Set CUDA
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cpu':
    print("WARNING: CUDA is not available. Using CPU, which may result in slower computation.")

def load_data(file_path):
    try:
        data = LogParser(file_path).get_all_data()
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def initialize_model_and_tokenizer(model_name):
    try:
        model_config = transformers.AutoConfig.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return model_config, tokenizer
    except Exception as e:
        print(f"Error initializing model/tokenizer: {e}")
        return None, None

def load_model(model_name, bnb_config):
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
        )
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def create_bnb_config(use_4bit=True, compute_dtype=torch.float16, quant_type="nf4", nested_quant=False):
    return BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=nested_quant,
    )

def initialize_text_generation_pipeline(model, tokenizer):
    try:
        pipeline = transformers.pipeline(
            model=model,
            tokenizer=tokenizer,
            task="text-generation",
            temperature=0.3,
            repetition_penalty=1.2,
            return_full_text=True,
            max_new_tokens=1000,
        )
        return HuggingFacePipeline(pipeline=pipeline)
    except Exception as e:
        print(f"Error initializing text generation pipeline: {e}")
        return None

def initialize_faiss_index(data):
    try:
        retriever = SentenceTransformer('all-MiniLM-L6-v2')
        document_embeddings = retriever.encode(data, convert_to_tensor=True)
        dimension = document_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(np.array(document_embeddings.cpu()))
        return retriever, index
    except Exception as e:
        print(f"Error initializing FAISS index: {e}")
        return None, None

def query_index(index, retriever, query, data, top_k=7):
    try:
        query_embedding = retriever.encode(query, convert_to_tensor=True).cpu().numpy().reshape(1, -1)
        distances, indices = index.search(query_embedding, top_k)
        retrieved_documents = [data[i] for i in indices[0]]
        return retrieved_documents
    except Exception as e:
        print(f"Error querying index: {e}")
        return []

# Add the re-ranking function
def re_rank_documents(retrieved_docs, query, re_rank_top_k=3):
    re_ranker = CrossEncoder('ms-marco-MiniLM-L-6-v2')  
    query_doc_pairs = [(query, doc) for doc in retrieved_docs]
    re_rank_scores = re_ranker.predict(query_doc_pairs)
    ranked_docs = [doc for _, doc in sorted(zip(re_rank_scores, retrieved_docs), reverse=True)]
    return ranked_docs[:re_rank_top_k]

def extract_answer(response_text):
    match = re.search(r"\[/INST\](.*)", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()

def run_rag_chain(retrieved_documents, question, mistral_llm):
    try:
        prompt_template = """
        ### [INST] 
        Instruction: You are a robot build by Enova Robotics, 
        your mission is patrol and guard therfor your name is 
        P-Guard.
        Answer the question based on your 
        knowledge. Here is context to help:
        
        {context}
        
        QUESTION:
        {question} 
        
        [/INST]
        """
        prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
        llm_chain = LLMChain(llm=mistral_llm, prompt=prompt)
        rag_chain = ({"context": lambda x: retrieved_documents, "question": RunnablePassthrough()} | llm_chain)
        answer = rag_chain.invoke(question)
        return answer["text"]
    except Exception as e:
        print(f"Error running RAG chain: {e}")
        return None

def main(file_path, model_name, question):
    start_time = time.time()
    data = load_data(file_path)
    if data is None:
        return
    
    model_config, tokenizer = initialize_model_and_tokenizer(model_name)
    if model_config is None or tokenizer is None:
        return
    
    bnb_config = create_bnb_config()
    model = load_model(model_name, bnb_config)
    if model is None:
        return
    
    mistral_llm = initialize_text_generation_pipeline(model, tokenizer)
    if mistral_llm is None:
        return
    
    retriever, index = initialize_faiss_index(data)
    if retriever is None or index is None:
        return
    
    retrieved_documents = query_index(index, retriever, question, data)
    
    if retrieved_documents:
        re_ranked_docs = re_rank_documents(retrieved_documents, question)
        answer = run_rag_chain(re_ranked_docs, question, mistral_llm)
        if answer:
            print("Answer:", extract_answer(answer))
        else:
            print("No answer generated.")
    else:
        print("No documents retrieved.")
    end_time = time.time()  # End the timer
    print(f"Execution Time: {end_time - start_time} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run chatbot with RAG.")
    parser.add_argument("--data_path", type=str, required=False, help="Path to the log file.")
    parser.add_argument("--model_name", type=str, default='mistral-7B-instruct-v0.3', help="Model name.")
    parser.add_argument("--question", type=str, required=False, help="Question to ask the model.")
    parser.add_argument("--info", action="store_true", help="Display information about the script and exit.")
    args = parser.parse_args()

    if args.info:
        print("""
            This script is a chatbot system using Retrieval-Augmented Generation (RAG) to answer questions. 
            It performs the following steps:
            1. Loads a log file containing text data.
            2. Initializes a language model and tokenizer from the specified model.
            3. Uses FAISS for similarity search to find relevant information in the log data.
            4. Generates a response based on both retrieved documents and the language model's answer generation.

            Example usage:
            python chatbot_rag.py --data_path path/to/log.json --model_path path/to/model --question "What's the battery level?"
        """)
    else:
        if args.data_path and args.question:
            main(args.data_path, args.model_name, args.question)
        else:
            print("Please provide both --data_path and --question arguments or use --info for more details.")
