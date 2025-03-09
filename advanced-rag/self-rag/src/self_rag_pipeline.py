"""
Self-RAG Pipeline Implementation

This module implements a Self-RAG (Retrieval-Augmented Generation with Self-Reflection) pipeline
that incorporates self-reflection mechanisms to improve retrieval and generation quality.
"""

import os
import argparse
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


class SelfRAG:
    """
    A Self-RAG pipeline that incorporates self-reflection to improve
    retrieval and generation quality.
    """

    def __init__(
        self,
        vector_store_path: Optional[str] = None,
        embedding_model: str = "text-embedding-ada-002",
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        top_k: int = 4,
        confidence_threshold: float = 0.7,
    ):
        """
        Initialize the Self-RAG pipeline.

        Args:
            vector_store_path: Path to the FAISS vector store. If None, a new one will be created.
            embedding_model: Name of the embedding model to use.
            model_name: Name of the LLM to use.
            temperature: Temperature for the LLM.
            top_k: Number of documents to retrieve.
            confidence_threshold: Threshold for confidence score to determine if retrieval is needed.
        """
        self.embedding_model = OpenAIEmbeddings(model=embedding_model)
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.top_k = top_k
        self.confidence_threshold = confidence_threshold
        
        # Track last decisions for analysis
        self.last_retrieval_decision = ""
        self.last_critique = ""
        self.last_confidence_score = 0.0
        self.last_citations = []
        
        # Load vector store if path is provided
        if vector_store_path and os.path.exists(vector_store_path):
            self.vector_store = FAISS.load_local(vector_store_path, self.embedding_model)
        else:
            self.vector_store = None
            
        # Define the prompt template for retrieval decision
        self.retrieval_decision_template = PromptTemplate(
            input_variables=["question"],
            template="""
            You are an AI assistant that decides whether to retrieve external information to answer a question.
            
            Question: {question}
            
            First, assess if you already have enough knowledge to answer this question without looking up additional information.
            Consider:
            1. Is this asking for factual information?
            2. Is this asking for recent information that might be beyond your knowledge cutoff?
            3. Is this asking for specific details you might not know precisely?
            4. Is this asking for domain-specific knowledge?
            
            Make your decision and explain your reasoning.
            Then provide a confidence score between 0 and 1 (where 0 is completely uncertain, and 1 is completely certain).
            
            Decision: [RETRIEVE or NO_RETRIEVE]
            Reasoning: [Your explanation]
            Confidence: [Score between 0 and 1]
            """,
        )
        
        self.retrieval_decision_chain = LLMChain(llm=self.llm, prompt=self.retrieval_decision_template)
        
        # Define the prompt template for generation
        self.generation_template = PromptTemplate(
            input_variables=["question", "context"],
            template="""
            You are a helpful assistant that answers questions based on the provided context.
            
            Context:
            {context}
            
            Question:
            {question}
            
            Answer the question based on the context provided. If the answer cannot be found in the context, 
            say "I don't have enough information to answer this question." and suggest what additional 
            information would be needed to answer it accurately.
            """,
        )
        
        self.generation_chain = LLMChain(llm=self.llm, prompt=self.generation_template)
        
        # Define the prompt template for self-critique
        self.critique_template = PromptTemplate(
            input_variables=["question", "answer", "context"],
            template="""
            You are a critical evaluator assessing the quality of an AI assistant's response.
            
            Question: {question}
            
            AI Response: {answer}
            
            Context that was available to the AI: 
            {context}
            
            Please evaluate the response on the following criteria:
            1. Factual Accuracy: Is the information correct and supported by the context?
            2. Relevance: Does the response directly address the question?
            3. Comprehensiveness: Does the response cover all important aspects of the question?
            4. Hallucination: Does the response include information not supported by the context?
            
            For each criterion, provide a score between 0 and 1, and briefly explain your reasoning.
            Then provide an overall assessment and suggestions for improvement.
            
            Finally, provide a confidence score between 0 and 1 for the entire response.
            
            Factual Accuracy (0-1): 
            Explanation:
            
            Relevance (0-1):
            Explanation:
            
            Comprehensiveness (0-1):
            Explanation:
            
            Hallucination (0-1): [0 means high hallucination, 1 means no hallucination]
            Explanation:
            
            Overall Assessment:
            
            Improvement Suggestions:
            
            Confidence Score (0-1):
            """,
        )
        
        self.critique_chain = LLMChain(llm=self.llm, prompt=self.critique_template)
        
        # Define the prompt template for revised generation
        self.revised_generation_template = PromptTemplate(
            input_variables=["question", "context", "initial_answer", "critique"],
            template="""
            You are a helpful assistant that improves answers based on critical feedback.
            
            Question: {question}
            
            Context:
            {context}
            
            Your initial answer: {initial_answer}
            
            Critique of your initial answer: {critique}
            
            Please provide an improved answer that addresses the critique while staying grounded in the provided context.
            Be especially careful to avoid any hallucinations or claims not supported by the context.
            Where appropriate, indicate your level of confidence and cite the specific parts of the context that support key points.
            
            Improved answer:
            """,
        )
        
        self.revised_generation_chain = LLMChain(llm=self.llm, prompt=self.revised_generation_template)

    def decide_retrieval(self, query: str) -> Tuple[bool, float, str]:
        """
        Decide whether to retrieve external information based on the query.

        Args:
            query: The user query.

        Returns:
            Tuple of (retrieval_needed, confidence_score, reasoning)
        """
        # Get the retrieval decision from the LLM
        result = self.retrieval_decision_chain.run(question=query)
        
        # Parse the result
        lines = result.strip().split('\n')
        decision = "NO_RETRIEVE"  # Default
        reasoning = ""
        confidence = 0.0  # Default
        
        for line in lines:
            if line.startswith("Decision:"):
                decision = line.replace("Decision:", "").strip()
            elif line.startswith("Reasoning:"):
                reasoning = line.replace("Reasoning:", "").strip()
            elif line.startswith("Confidence:"):
                try:
                    confidence = float(line.replace("Confidence:", "").strip())
                except ValueError:
                    confidence = 0.0
        
        # Store for later reference
        self.last_retrieval_decision = f"Decision: {decision}\nReasoning: {reasoning}\nConfidence: {confidence}"
        
        # Convert decision to boolean
        retrieval_needed = decision.upper() == "RETRIEVE"
        
        return retrieval_needed, confidence, reasoning

    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: The query to retrieve documents for.

        Returns:
            List of retrieved documents.
        """
        if self.vector_store is None:
            raise ValueError("No vector store available. Ingest documents first.")
            
        retrieved_docs = self.vector_store.similarity_search(query, k=self.top_k)
        return retrieved_docs

    def generate_initial(self, query: str, context_docs: List[Document]) -> str:
        """
        Generate an initial response based on the query and retrieved documents.

        Args:
            query: The user query.
            context_docs: The retrieved context documents.

        Returns:
            Generated response.
        """
        # Combine document texts into a single context string
        context = "\n\n".join([doc.page_content for doc in context_docs])
        
        # Generate response
        response = self.generation_chain.run(question=query, context=context)
        return response

    def critique_response(self, query: str, response: str, context_docs: List[Document]) -> Tuple[str, float]:
        """
        Critique the response for factuality, relevance, comprehensiveness, and hallucination.

        Args:
            query: The user query.
            response: The generated response.
            context_docs: The retrieved context documents.

        Returns:
            Tuple of (critique, confidence_score)
        """
        # Combine document texts into a single context string
        context = "\n\n".join([doc.page_content for doc in context_docs])
        
        # Generate critique
        critique = self.critique_chain.run(question=query, answer=response, context=context)
        
        # Parse confidence score
        confidence = 0.0
        for line in critique.strip().split('\n'):
            if line.startswith("Confidence Score:"):
                try:
                    confidence = float(line.replace("Confidence Score:", "").strip())
                except ValueError:
                    confidence = 0.0
        
        # Store for later reference
        self.last_critique = critique
        self.last_confidence_score = confidence
        
        return critique, confidence

    def revise_response(self, query: str, initial_response: str, critique: str, context_docs: List[Document]) -> str:
        """
        Revise the response based on the critique.

        Args:
            query: The user query.
            initial_response: The initial generated response.
            critique: The critique of the initial response.
            context_docs: The retrieved context documents.

        Returns:
            Revised response.
        """
        # Combine document texts into a single context string
        context = "\n\n".join([doc.page_content for doc in context_docs])
        
        # Generate revised response
        revised_response = self.revised_generation_chain.run(
            question=query,
            context=context,
            initial_answer=initial_response,
            critique=critique
        )
        
        return revised_response

    def query(self, query: str) -> Dict[str, Any]:
        """
        End-to-end Self-RAG pipeline: decide, retrieve, generate, critique, and revise.

        Args:
            query: The user query.

        Returns:
            Dictionary containing the query, retrieval decision, retrieved documents, 
            initial response, critique, confidence score, and final response.
        """
        # Decide whether to retrieve
        retrieval_needed, confidence, reasoning = self.decide_retrieval(query)
        
        # Initialize context_docs
        context_docs = []
        
        # Retrieve if needed or if confidence is below threshold
        if retrieval_needed or confidence < self.confidence_threshold:
            context_docs = self.retrieve(query)
        
        # Generate initial response
        initial_response = self.generate_initial(query, context_docs)
        
        # Critique the response
        critique, confidence = self.critique_response(query, initial_response, context_docs)
        
        # Revise the response if confidence is below threshold
        final_response = initial_response
        if confidence < self.confidence_threshold:
            final_response = self.revise_response(query, initial_response, critique, context_docs)
        
        return {
            "query": query,
            "retrieval_decision": self.last_retrieval_decision,
            "retrieved_documents": context_docs,
            "initial_response": initial_response,
            "critique": critique,
            "confidence_score": confidence,
            "final_response": final_response,
        }


def main():
    """
    Main function to run the Self-RAG pipeline from the command line.
    """
    parser = argparse.ArgumentParser(description="Run the Self-RAG pipeline")
    parser.add_argument("--query", type=str, required=True, help="The query to process")
    parser.add_argument(
        "--vector_store_path",
        type=str,
        default="../../../basic-rag/data/processed/vector_store",
        help="Path to the vector store",
    )
    args = parser.parse_args()

    # Initialize the Self-RAG pipeline
    self_rag = SelfRAG(vector_store_path=args.vector_store_path)
    
    # Process the query
    result = self_rag.query(args.query)
    
    # Print the response
    print("\nQuery:", result["query"])
    
    print("\nRetrieval Decision:")
    print(result["retrieval_decision"])
    
    if result["retrieved_documents"]:
        print("\nRetrieved Documents:")
        for i, doc in enumerate(result["retrieved_documents"]):
            print(f"\nDocument {i+1}:")
            print(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
    
    print("\nInitial Response:")
    print(result["initial_response"])
    
    print("\nCritique:")
    print(result["critique"])
    
    print("\nConfidence Score:", result["confidence_score"])
    
    print("\nFinal Response:")
    print(result["final_response"])


if __name__ == "__main__":
    main() 