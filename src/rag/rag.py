from typing import List, Dict, Any
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import Pinecone, PineconeVectorStore
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

import os 
from dotenv import load_dotenv

load_dotenv()

class Rag:
    def __init__(self):
        try:
            self.llm = init_chat_model(model_name="gpt-4o", model_provider="openai")
            
            # Initialize embeddings
            self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=os.getenv('OPENAI_API_KEY'))
            
            # Initialize vector store
            self.vector_store = PineconeVectorStore(
                embedding=self.embeddings,
                index=os.getenv('PINECONE_INDEX_NAME'),
                pinecone_api_key=os.getenv('PINECONE_API_KEY')
            )
            
        except Exception as e:
            print(f"Failed to initialize RAG system: {str(e)}") 
            raise

    def store(self, data: List[Document]) -> bool:
        """
        Store documents in the vector database.
        
        Args:
            data: List of Document objects to store
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print(f"Processing {len(data)} documents for storage")
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                add_start_index=True,
            )

            splits = text_splitter.split_documents(data)
            print(f"Split into {len(splits)} chunks")

            self.vector_store.add_documents(splits)
            print("Documents successfully added to vector store")
            
            return True
            
        except Exception as e:
            print(f"Error storing documents: {str(e)}")
            return False
            
    def query(self, query_text: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Query the RAG system with a question.
        
        Args:
            query_text: The query or question
            top_k: Number of most relevant documents to retrieve
            
        Returns:
            Dict containing the answer and source documents
        """
        try:
            print(f"Processing query: {query_text}")
            
            # Create retriever
            retriever = self.vector_store.as_retriever(
                search_kwargs={"k": top_k}
            )
            
            # Create prompt template
            template = """
            Answer the question based only on the following context:
            {context}
            
            Question: {question}
            """
            prompt = ChatPromptTemplate.from_template(template)
            
            # Create RAG chain
            rag_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
            )
            
            # Execute query
            response = rag_chain.invoke(query_text)
            
            # Get source documents
            source_docs = retriever.get_relevant_documents(query_text)
            sources = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in source_docs
            ]
            
            return {
                "query": query_text,
                "response": response,
                "sources": sources
            }
            
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            raise