import spacy
import os
from datetime import datetime
from custom_logger import CustomLogger

LOG_LEVEL = os.getenv('LOG_LEVEL')

def test_spacy_model_endpoint(text, dataset_name):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger = None
        # Initialize the custom logger
        logger = CustomLogger(
            name='SpacyTestLogger',
            level=LOG_LEVEL,
            log_file=os.path.join("/workspace/logs", f"spacy_test_{timestamp}.log")
        ).get_logger()
            
        output_model_dir = f"/models/spacy/en_core_web_trf_finetuned_{dataset_name}/"
        
        # Check for GPU availability
        if spacy.prefer_gpu():
            logger.info("Using GPU for testing.")
        else:
            logger.info("No GPU found, using CPU.")
                
        # Load the fine-tuned model
        nlp = spacy.load(output_model_dir)
        #nlp = spacy.load("en_core_web_trf")

        # Process the input text
        doc = nlp(text)
        
        # Get the predicted categories (intents) and their confidence scores
        intents_scores = sorted(doc.cats.items(), key=lambda item: item[1], reverse=True)[:5]

        # Extract entities and their labels
        entities = [(ent.text, ent.label_) for ent in doc.ents]

        # Return top 5 intents with scores and extracted entities
        return {
            "intents": intents_scores,
            "entities": entities,
        }
    except FileNotFoundError as fnf_error:
        if logger:
            logger.error(f"FileNotFoundError: {fnf_error}")
        raise
    except Exception as e:
        if logger:
            logger.error(f"An error occurred: {e}")
        raise