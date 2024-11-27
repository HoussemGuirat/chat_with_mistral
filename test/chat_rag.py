import re
import time
import torch
import warnings
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from logparser import LogParser
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema.runnable import RunnablePassthrough
import transformers

warnings.filterwarnings("ignore")

class ChatSystem:
    def __init__(self, model_name="./mistral-7B-instruct-v0.3", device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        if self.device == 'cpu':
            print("WARNING: CUDA is not available. Using CPU, which may result in slower computation.")
        
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.mistral_llm = None
        self.retriever = None
        self.index = None
        self.data = None

    def load_data(self, file_path):
        try:
            self.data = LogParser(file_path).get_all_data()
        except Exception as e:
            print(f"Error loading data: {e}")
            self.data = None

    def initialize_model_and_tokenizer(self):
        try:
            model_config = transformers.AutoConfig.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
            return model_config
        except Exception as e:
            print(f"Error initializing model/tokenizer: {e}")
            return None

    def load_model(self, bnb_config):
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
            )
        except Exception as e:
            print(f"Error loading model: {e}")

    def create_bnb_config(self):
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        )

    def initialize_text_generation_pipeline(self):
        try:
            pipeline = transformers.pipeline(
                model=self.model,
                tokenizer=self.tokenizer,
                task="text-generation",
                temperature=0.2,
                repetition_penalty=1,
                return_full_text=False,
                max_new_tokens=1000,
            )
            self.mistral_llm = HuggingFacePipeline(pipeline=pipeline)
        except Exception as e:
            print(f"Error initializing text generation pipeline: {e}")

    def initialize_faiss_index(self):
        try:
            if self.data is None:
                print("No data loaded for FAISS index.")
                return
            self.retriever = SentenceTransformer('./all-MiniLM-L6-v2')
            document_embeddings = self.retriever.encode(self.data, convert_to_tensor=True)
            dimension = document_embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(np.array(document_embeddings.cpu()))
        except Exception as e:
            print(f"Error initializing FAISS index: {e}")

    def query_index(self, query, top_k=5):
        try:
            if self.index is None:
                print("FAISS index is not initialized.")
                return []
            query_embedding = self.retriever.encode(query, convert_to_tensor=True).cpu().numpy().reshape(1, -1)
            distances, indices = self.index.search(query_embedding, top_k)
            return [self.data[i] for i in indices[0]]
        except Exception as e:
            print(f"Error querying index: {e}")
            return []

    def re_rank_documents(self, retrieved_docs, query, re_rank_top_k=3):
        re_ranker = CrossEncoder('./ms-marco-MiniLM-L-6-v2')
        query_doc_pairs = [(query, doc) for doc in retrieved_docs]
        re_rank_scores = re_ranker.predict(query_doc_pairs)
        ranked_docs = [doc for _, doc in sorted(zip(re_rank_scores, retrieved_docs), reverse=True)]
        return ranked_docs[:re_rank_top_k]

    def run_rag_chain(self, retrieved_documents, question):
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
            llm_chain = LLMChain(llm=self.mistral_llm, prompt=prompt)
            rag_chain = ({"context": lambda x: retrieved_documents, "question": RunnablePassthrough()} | llm_chain)
            answer = rag_chain.invoke(question)
            return answer["text"]
        except Exception as e:
            print(f"Error running RAG chain: {e}")
            return None

    def mount(self, file_path):
        self.load_data(file_path)
        if self.data is None:
            return
        
        model_config = self.initialize_model_and_tokenizer()
        if model_config is None:
            return
        
        bnb_config = self.create_bnb_config()
        self.load_model(bnb_config)
        self.initialize_text_generation_pipeline()
        self.initialize_faiss_index()

    def run(self, question):
        if not self.data or not self.index:
            print("System is not properly mounted. Please call `mount` first.")
            return
        
        retrieved_documents = self.query_index(question)
        if retrieved_documents:
            re_ranked_docs = self.re_rank_documents(retrieved_documents, question)
            if self.run_rag_chain(re_ranked_docs, question):
                return self.run_rag_chain(re_ranked_docs, question)
            else:
                print("No answer generated.")
        else:
            print("No documents retrieved.")


#Usage
# Initialize the system
#chat_system = ChatSystem()

# Mount the system with the log file "log_p-guard1.json"
#log_file_path = "./log_p-guard1.json"
#chat_system.mount(log_file_path)

# Ask a question
#question = "What is the status of the latest error in the log?"
#chat_system.run(question)