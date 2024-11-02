import os
import json
import random
from datetime import datetime
import spacy
from spacy.training import Example
#from spacy.vocab import Vectors
import numpy as np
from spacy.util import minibatch
from custom_logger import CustomLogger

SPACY_MODEL = os.getenv('SPACY_MODEL')
LOG_LEVEL = os.getenv('LOG_LEVEL')
def train_spacy(dataset_name, num_iterations=30, learning_rate=0.0005, log_dir="/workspace/logs"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = None
    try:
        # Initialize the custom logger
        logger = CustomLogger(
            name='SpacyTrainingLogger',
            level=LOG_LEVEL,
            log_file=os.path.join(log_dir, f"spacy_training_{timestamp}.log")
        ).get_logger()

        # Check for GPU availability
        if spacy.prefer_gpu():
            logger.info("Using GPU for training.")
        else:
            logger.info("No GPU found, using CPU.")
            
        # Load pre-trained spaCy model
        nlp = spacy.load(SPACY_MODEL)
        # Step 1: Check if the model is transformer-based
        if "transformer" in nlp.pipe_names:
            print("This is a transformer-based model. Vectors are not needed.")
        else:
            print("This is not a transformer-based model.")
            # Step 2: Check if 'tok2vec' is in the pipeline and add it if needed
            # We are using vectors from the medium model for this example
            # Load vectors from en_core_web_md
            nlp_md = spacy.load("en_core_web_md")
            tok2vec_component = nlp_md.get_pipe("tok2vec")
            vectors = nlp_md.vocab.vectors
            # Free up memory by deleting the medium model
            del nlp_md  
            
            if "tok2vec" not in nlp.pipe_names:
                # Transfer the tok2vec component
                if "tok2vec" in nlp_md.pipe_names:
                    nlp.add_pipe("tok2vec", source=tok2vec_component, first=True)
                    del tok2vec_component
                    print("tok2vec component transferred from en_core_web_md to en_core_web_sm.")
                else:
                    print("tok2vec component not found in en_core_web_md.")
                    nlp.add_pipe("tok2vec", first=True)
                    print("tok2vec added to the pipeline.")
            else:
                print("tok2vec is already present in the pipeline.")
            
            # Step 3: Check if the model has vectors
            if not nlp.vocab.vectors:
                print("No vectors found in the model. Adding custom vectors...")
                nlp.vocab.vectors = vectors
                del vectors
                ## Add vectors (example with random vectors; replace with actual embeddings)
                #
                ## Random Vectors: 
                ## Using random vectors won't improve the model's semantic capabilities. 
                ## They serve as placeholders and don't provide meaningful embeddings. 
                ## For real-world applications use pre-trained vectors from sources like GloVe, FastText, or custom-trained word embeddings.
                ## Pre-trained Embeddings: 
                ## If possible, use pre-trained embeddings to add to your model's vocabulary. 
                ## These can be loaded from text or binary files (e.g., GloVe, word2vec). 
                # vectors_data = np.random.rand(1000, 300)  
                ## Replace with pre-trained vector data
                # vectors = Vectors(data=vectors_data, keys=np.arange(1000))
                # nlp.vocab.vectors = vectors
                print("Custom vectors added to the model.")
            else:
                print(f"Model already has {len(nlp.vocab.vectors)} vectors.")      
        
        examples, textcat = prepare_spacy_data_and_pipeline(dataset_name, nlp, False, logger)
        validation_examples, validation_textcat = prepare_spacy_data_and_pipeline(dataset_name, nlp, True, logger)
        
        output_model_dir = f"/models/spacy/en_core_web_trf_finetuned_{dataset_name}"
        # Create necessary directories
        os.makedirs(output_model_dir, exist_ok=True)
        
        # Initialize the text categorizer with the examples
        textcat.initialize(lambda: examples)
        # Early stopping parametri
        patience = 2
        best_val_loss = float('inf')
        no_improve_count = 0
        early_stop = False

        # Start training
        optimizer = nlp.resume_training()
        optimizer.learn_rate = learning_rate
        batch_size = 32
        
        logger.info(f"Num iterations: {num_iterations} Learning rate: {learning_rate} Batch size: {batch_size}")
        
        for i in range(num_iterations):
            random.shuffle(examples)
            losses = {}
            train_loss = 0.0

            for batch in minibatch(examples, size=batch_size):
                nlp.update(batch, drop=0.2, losses=losses)
                train_loss += losses.get('textcat', 0)
                
            train_loss /= len(examples)  # Prosečan gubitak po epohi
            logger.info(f"Iteration {i + 1}/{num_iterations} Training Loss: {train_loss:.6f}")
            
            # Provera validacionog gubitka (dodati validacioni skup)
            val_loss = calculate_validation_loss(nlp, validation_examples, batch_size)  # Dodajte funkciju za validaciju
            logger.info(f"Iteration {i + 1}/{num_iterations} Validation Loss: {val_loss:.6f}")
            
            # Early stopping logika
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve_count = 0
                # Čuvanje najboljeg modela
                nlp.to_disk(output_model_dir)
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    logger.info('Early stopping triggered')
                    early_stop = True
                    break

            if early_stop:
                break
        logger.info(f"Fine-tuned model saved to {output_model_dir}")
        return output_model_dir
    except FileNotFoundError as fnf_error:
        if logger:
            logger.error(f"FileNotFoundError: {fnf_error}")
        raise
    except Exception as e:
        if logger:
            logger.error(f"An error occurred: {e}")
        raise

def calculate_validation_loss(nlp, validation_examples, batch_size = 32):
    """Funkcija za izračunavanje validacionog gubitka."""
    losses = {}
    for batch in minibatch(validation_examples, size=batch_size):
        nlp.update(batch, sgd=None, losses=losses, drop=0.0)  # Bez ažuriranja optimizatora
    return losses.get('textcat', 0) / len(validation_examples)

def prepare_spacy_data_and_pipeline(dataset_name, nlp, is_validation = False, logger = None):
    """
    Prepares spaCy training data and sets up the pipeline for fine-tuning based on a given dataset name.
    
    Args:
        dataset_name (str): Name of the dataset used to prepare training data.
        nlp (Language): spaCy language model to be used for training.
        is_validation (bool): Flag to indicate whether the dataset is for validation or not.
        logger (Logger): Custom logger object for logging messages.
    Returns:
        examples (list): List of spaCy Example objects prepared from the dataset.
    """
    # Set up paths
    base_dir = f"/datasets/spacy/{dataset_name}_spacy"
    if is_validation:
        json_file_path = os.path.join(base_dir, f"intent_data_validation_{dataset_name.replace('/', '_')}.json")
    else:
        json_file_path = os.path.join(base_dir, f"intent_data_{dataset_name.replace('/', '_')}.json")
        
    # Verify paths and load the dataset
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Dataset directory not found: {base_dir}")

    with open(json_file_path, "r") as f:
        intent_data = json.load(f)

    # Extract distinct intents
    distinct_intents = {entry[1]["intent"] for entry in intent_data}

    # Add or get the text categorizer
    if "textcat" not in nlp.pipe_names:
        textcat = nlp.add_pipe("textcat", last=True)
        logger.info("Text categorizer added to the pipeline.")
    else:
        textcat = nlp.get_pipe("textcat")
        logger.info("Text categorizer retrieved from the pipeline.")

    # Add labels to the text categorizer
    for intent in distinct_intents:
        textcat.add_label(intent)

    # Prepare training data in spaCy's Example format
    examples = [
        Example.from_dict(nlp.make_doc(text), {"cats": {annotations["intent"]: 1.0}})
        for text, annotations in intent_data
    ]

    logger.info(f"Training examples prepared and text categorizer initialized for {dataset_name}")
    
    return examples, textcat