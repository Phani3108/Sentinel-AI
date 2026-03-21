import logging
from langchain_community.chat_models import ChatOllama
from core.config import get_settings

logger = logging.getLogger(__name__)

class TranslatorEngine:
    """
    The Babel Matrix.
    Intercepts physical AI Ordnance payloads and natively uses highly isolated smaller
    LLMs to dynamically translate threat outputs into active physical Regional deployments.
    """
    def __init__(self):
        # We can use the primary LLM model for translation
        self.llm = ChatOllama(model=get_settings().ollama_llm_model, temperature=0.1)
        
    def translate_payload(self, text: str, target_region: str) -> str:
        # Standardized mock regions
        region_map = {
            "AP-SOUTH-1": "Hindi",
            "AP-NORTHEAST-1": "Japanese",
            "EU-WEST-2": "French",
            "SA-EAST-1": "Spanish"
        }
        
        target_lang = region_map.get(target_region.upper(), "English")
        if target_lang == "English":
            return text
            
        logger.info(f"Babel Matrix: Intercepting payload strictly routing -> {target_lang}")
        
        prompt = (
            f"You are the Sentinel Babel Security Translator. "
            f"Translate the following high-alert security payload into strict, formal {target_lang}. "
            f"Do not add any conversational filler or introductions. Only output the direct {target_lang} translation.\n\n"
            f"PAYLOAD TO TRANSLATE:\n{text}"
        )
        
        try:
            res = self.llm.invoke(prompt)
            return res.content.strip()
        except Exception as e:
            logger.error(f"Babel Matrix Routing Fault: {e}. Defaulting back to native English vector.")
            return text
            
translator_engine = TranslatorEngine()
