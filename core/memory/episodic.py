import logging
from datetime import datetime
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

class EpisodicMemory:
    """
    Persistent temporal graph. Stores visual entities and threat signatures across
    time bounds so the Agent Swarm can establish context linking previously isolated incidents.
    """
    def __init__(self):
        try:
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            self.db = Chroma(
                collection_name="sentinel_memory",
                embedding_function=embeddings,
                persist_directory="./chroma_db"
            )
        except Exception as e:
            logger.error(f"Failed to mount Episodic Memory: {e}")
            self.db = None
            
    def store_episode(self, entity_description: str, threat_level: str):
        """Commits the visual context into the Vector Store with precise temporal metadata."""
        if not self.db: 
            return
        now = datetime.now().isoformat()
        content = f"[{now}] THREAT LEVEL: {threat_level} | CONTEXT: {entity_description}"
        self.db.add_texts(texts=[content], metadatas=[{"timestamp": now, "type": "episode"}])
        logger.info("EpisodicMemory: Incident committed to long-term storage.")
        
    def recall_history(self, query: str, top_k: int = 3) -> str:
        """Retrieves semantically similar previous incursions to inform the Director."""
        if not self.db: 
            return ""
        try:
            # Simple similarity extraction bridging independent events
            docs = self.db.similarity_search(query, k=top_k)
            if not docs: 
                return ""
            return "\n".join([d.page_content for d in docs])
        except Exception:
            return ""
