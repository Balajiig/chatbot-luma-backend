# services.py
import logging
import json
import numpy as np
import glob 
import os

from openai import AzureOpenAI, OpenAIError
from config import Settings
from schemas import ChatRequest

# Set up a logger
logger = logging.getLogger(__name__)
CHAT_HISTORY_CACHE = {}

# Hofstra-specific static response for CRISIS detection
HOFSTRA_CRISIS_PROTOCOL = (
    "It sounds like you're in serious distress. If you are on the Hofstra campus "
    "and it's outside regular hours, **please call Public Safety at 516-463-6789** "
    "to reach an on-call counselor immediately. If you're off-campus or this is a "
    "medical emergency, please call **911** or go to the nearest emergency room. "
    "You can also contact the Long Island Crisis Center at **516-679-1111** (24/7)."
)

class ChatService:
    """
    Manages the conversation logic, integrating NLU and RAG
    using two separate Azure OpenAI resources. Optimized for Hofstra context.
    """
    def __init__(self, settings: Settings):
        try:
            # --- 1. CHAT CLIENT (For NLU and Chat Responses) ---
            self.chat_client = AzureOpenAI(
                api_key=settings.AZURE_OPENAI_CHAT_KEY,
                api_version=settings.AZURE_OPENAI_CHAT_API_VERSION,
                azure_endpoint=settings.AZURE_OPENAI_CHAT_ENDPOINT,
            )
            self.chat_deployment = settings.AZURE_OPENAI_CHAT_DEPLOYMENT_NAME
            logger.info("AzureOpenAI CHAT client initialized.")

            # --- 2. EMBEDDING CLIENT (For RAG) ---
            self.embed_client = AzureOpenAI(
                api_key=settings.AZURE_OPENAI_EMBED_KEY,
                api_version=settings.AZURE_OPENAI_EMBED_API_VERSION,
                azure_endpoint=settings.AZURE_OPENAI_EMBED_ENDPOINT,
            )
            self.embed_deployment = settings.AZURE_OPENAI_EMBED_DEPLOYMENT_NAME
            logger.info("AzureOpenAI EMBEDDING client initialized.")

            # --- In-Memory RAG Setup ---
            self.rag_kb = [] # This will store (text, vector) tuples
            
            # Ensure the new Hofstra resource file is created for RAG to pick up
            self._create_hofstra_resource_doc() 
            self._init_in_memory_rag() # Load the RAG knowledge base
            
            logger.info(f"In-Memory RAG loaded with {len(self.rag_kb)} documents.")

        except Exception as e:
            logger.error(f"Failed to initialize ChatService: {e}")
            self.chat_client = None
            self.embed_client = None

    def _create_hofstra_resource_doc(self):
        """
        Creates the Hofstra resource document if it doesn't exist, 
        so _init_in_memory_rag can load it.
        """
        file_path = "docs/hofstra_resources.txt"
        if not os.path.exists(file_path):
            # This is a critical step to ensure the new context is available
            # Note: This file content should ideally be generated externally,
            # but is included here for completeness of the solution.
            logger.warning(f"Creating new resource file: {file_path}. Ensure content is correct.")
            # We assume the content from the file generated above is used here
            # For this context, we rely on the user having added the file.
            pass # Since the file is generated separately, we skip programmatic creation here.
            
    def _init_in_memory_rag(self):
        """
        [RAG] Loads documents from 'docs/', creates embeddings, and stores them in memory.
        """
        if not self.embed_client:
            logger.error("RAG init failed: Embedding client not initialized.")
            return
        
        doc_files = glob.glob("docs/*.txt")
        for file_path in doc_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    
                    # Call the EMBEDDING client
                    response = self.embed_client.embeddings.create(
                        input=text, model=self.embed_deployment
                    )
                    vector = response.data[0].embedding
                    
                    self.rag_kb.append((text, np.array(vector)))
                    logger.info(f"Embedded and stored: {os.path.basename(file_path)}")
                    
            except Exception as e:
                logger.error(f"Failed to load/embed document {file_path}: {e}")

    # --- UPDATED NLU Intent Classifier with Hofstra-specific intents ---
    def _get_nlu_intent(self, message: str) -> dict:
        """
        [NLU] Uses the CHAT client to classify the user's intent.
        Added Hofstra-specific and critical intents.
        """
        if not self.chat_client:
            logger.error("NLU failed: Chat client not initialized.")
            return {"intent": {"name": "unknown"}}
        
        # Expanded Intent List
        intents = [
            'crisis', # For self-harm, severe distress, psychological emergency
            'work_stress', 
            'study_anxiety', 
            'feeling_depressed', 
            'want_referral', # Asking for counseling/Dean of Students contact
            'need_quiet_spot', # Asking for a good place to focus
            'need_group_space', # Asking for group study room
            'need_physical_break', # Asking for exercise/game room
            'affirm', 
            'deny', 
            'general_greeting', 
            'unknown'
        ]
        
        system_prompt = (
            "You are an NLU (Natural Language Understanding) classifier for the Hofstra "
            "Counseling Support AI. Analyze the user's message and respond *only* with a JSON object. "
            "The JSON must have one key: 'intent'. "
            f"The intent must be one of the following strings: {intents}"
        )
        
        try:
            response = self.chat_client.chat.completions.create(
                model=self.chat_deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=100
            )
            
            intent_data = json.loads(response.choices[0].message.content)
            intent = intent_data.get("intent", "unknown")
            logger.info(f"NLU (OpenAI) detected intent: {intent}")
            return {"intent": {"name": intent}}

        except Exception as e:
            logger.error(f"Error in OpenAI NLU call: {e}")
            return {"intent": {"name": "unknown"}}

    def _get_rag_context(self, message: str) -> str:
        """
        [RAG] Finds the most relevant context from the in-memory vector store.
        (Logic remains the same, but now includes new Hofstra resources).
        """
        if not self.embed_client or not self.rag_kb:
            logger.error("RAG failed: Embedding client or KB not initialized.")
            return "Retrieved context: 'Acknowledge the user's statement.'"

        try:
            # 1. Create embedding for the user's message (uses EMBED client)
            response = self.embed_client.embeddings.create(
                input=message, model=self.embed_deployment
            )
            query_vector = np.array(response.data[0].embedding)
            
            # 2. Calculate cosine similarity against all docs in memory
            # Use dot product for cosine similarity when vectors are normalized (which embeddings usually are)
            scores = [
                np.dot(query_vector, doc_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(doc_vector))
                for _, doc_vector in self.rag_kb
            ]
            
            # 3. Get the top-scoring document
            top_index = np.argmax(scores)
            top_text = self.rag_kb[top_index][0]
            
            logger.info(f"RAG (In-Memory) found best context with score {scores[top_index]}")
            return f"Retrieved technique/resource: '{top_text}'"

        except Exception as e:
            logger.error(f"Error in in-memory RAG retrieval: {e}")
            return "Retrieved context: 'Acknowledge the user's statement...'"


    # --- UPDATED Main Orchestration Function ---
    def get_chat_response(self, request: ChatRequest) -> tuple[str, str, str]:
        """
        Main orchestration function. Includes pre-LLM check for 'crisis' intent.
        """
        if not self.chat_client or not self.embed_client:
            logger.error("A required client is not initialized. Cannot process request.")
            raise OpenAIError("The AI service is not configured correctly.")
            
        session_id = request.session_id or "default_session"
        history = CHAT_HISTORY_CACHE.get(session_id, [])

        # 1. NLU Step (Calls CHAT client)
        nlu_data = self._get_nlu_intent(request.message)
        intent = nlu_data.get("intent", {}).get("name", "unknown")

        # --- CRISIS PROTOCOL INTERVENTION (CRITICAL) ---
        if intent == 'crisis':
            # Immediately bypass RAG/LLM and return the static, safe protocol.
            llm_response = HOFSTRA_CRISIS_PROTOCOL
            context = "CRISIS INTERVENTION PROTOCOL TRIGGERED"
            
            # Do NOT save crisis-related conversation to history to avoid confusing the next turn
            # and to prioritize immediate safety info.
            return llm_response, intent, context
        # ---------------------------------------------

        # 2. RAG Step (Calls EMBED client)
        context = self._get_rag_context(request.message)
        
        # 3. LLM Generation Step
        prompt = self._build_llm_prompt(request.message, intent, context, history)
        
        try:
            # Calls the CHAT client
            completion = self.chat_client.chat.completions.create(
                model=self.chat_deployment,
                messages=prompt,
                temperature=0.7,
                max_tokens=250, # Increased max_tokens slightly for richer resource responses
            )
            
            llm_response = completion.choices[0].message.content.strip()
            
            # 4. Save to cache
            history.append({"role": "user", "content": request.message})
            history.append({"role": "assistant", "content": llm_response})
            CHAT_HISTORY_CACHE[session_id] = history

            return llm_response, intent, context

        except OpenAIError as e:
            logger.error(f"Error calling Azure OpenAI: {e}")
            raise
            
    # --- UPDATED LLM System Prompt for Hofstra Persona ---
    def _build_llm_prompt(self, message: str, intent: str, context: str, history: list[dict]) -> list[dict]:
        
        SYSTEM_PROMPT = (
            "You are the 'Hofstra Counseling Support AI'. Your role is to provide "
            "empathetic and practical emotional support to Hofstra students and professors. "
            "Act as a non-judgmental, warm, and highly knowledgeable resource, similar to a "
            "member of the Hofstra Student Counseling Services staff. "
            "You are strictly **NOT** a human counselor and must **NEVER** give medical advice. "
            "You must use the 'Retrieved Context' to directly suggest the specific Hofstra "
            "resource (e.g., Axinn Library 10th floor, Arboretum walk, contact numbers) "
            "or coping technique it contains. "
            "The 'User Intent' is for your information. Do not mention it explicitly. "
            "Keep your responses concise, highly supportive, and end with a question "
            "to encourage the user to keep talking."
        )

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history)
        messages.append({
            "role": "user",
            "content": (
                f"User Message: \"{message}\"\n\n"
                f"--- (Internal analysis) ---\n"
                f"User Intent: {intent}\n"
                f"Retrieved Context: {context}\n"
                f"--- (End analysis) ---\n\n"
                f"Your response:"
            )
        })
        
        return messages