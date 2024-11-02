import os
import spacy
from datasets import load_dataset
from custom_logger import CustomLogger
from datetime import datetime
import json
from datasets import load_dataset, DatasetDict
from sklearn.model_selection import train_test_split

LOG_LEVEL = os.getenv('LOG_LEVEL')
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logger = None
# Initialize the custom logger (example)
logger = CustomLogger(name='SpacyConversionLogger', level=LOG_LEVEL, log_file=f"/workspace/logs/spacy_conversion_{timestamp}.log").get_logger()
def convert_to_spacy(dataset_name, config_name=None):
    try:
        # Učitavanje dataset-a pomoću check_or_split_dataset
        dataset_dict = check_or_split_dataset(dataset_name, config_name)
        # Pristupanje trening skupu
        dataset = dataset_dict['train']
        validation_dataset = dataset_dict['validation']
        #dataset = load_dataset(dataset_name, config_name, split="train")
        
        if dataset_name == "bkonkle/snips-joint-intent":
            # Parse the dataset into the INTENT_DATA format
            INTENT_DATA = [
                (example["input"], {"intent": example["intent"]})
                for example in dataset
            ]
            VALIDATION_INTENT_DATA = [
                (example["input"], {"intent": example["intent"]})
                for example in validation_dataset
            ]
        else:
            # Extract the list of intent labels from the dataset's "intent" column feature information
            intent_labels = dataset.features["intent"].names
            # Parse the dataset into the INTENT_DATA format
            INTENT_DATA = [
                (example["text"], {"intent": intent_labels[example["intent"]]})
                for example in dataset
            ]
            VALIDATION_INTENT_DATA = [
                (example["text"], {"intent": intent_labels[example["intent"]]})
                for example in validation_dataset
            ]
        # Create the output directory if it doesn't exist
        output_dir = f"/datasets/spacy/{dataset_name}_spacy"
        os.makedirs(output_dir, exist_ok=True)
        # Save the parsed data to a JSON file in the specified directory
        output_file_path = os.path.join(output_dir, f"intent_data_{dataset_name.replace('/', '_')}.json")
        output_validation_file_path = os.path.join(output_dir, f"intent_data_validation_{dataset_name.replace('/', '_')}.json")
        with open(output_file_path, "w") as f:
            json.dump(INTENT_DATA, f, indent=4)
        with open(output_validation_file_path, "w") as f:
            json.dump(VALIDATION_INTENT_DATA, f, indent=4)
            
        logger.info(f"Data successfully converted and saved to {output_file_path}")
        return output_dir
    except FileNotFoundError as fnf_error:
        logger.error(f"FileNotFoundError: {fnf_error}")
        print(f"Error: {fnf_error}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        print(f"An error occurred: {e}")
        raise  # Re-raise the exception after logging

def check_or_split_dataset(dataset_name, config_name=None, test_size=0.2, seed=42):
    """
    Proverava da li učitani dataset ima validacioni skup. Ako ne postoji, deli trening skup na 80/20.

    Args:
        dataset_name (str): Naziv dataset-a koji se učitava.
        config_name (str, optional): Konfiguracija dataset-a, ako postoji.
        test_size (float, optional): Procenat podataka za validacioni skup. Default je 0.2 (20%).
        seed (int, optional): Nasumično seme za replikaciju podele. Default je 42.

    Returns:
        DatasetDict: Dataset koji sadrži trening, validacioni (kreirani ako je potrebno), i test skupove.
    """
    dataset = load_dataset(dataset_name, config_name) if config_name else load_dataset(dataset_name)
    print(f"Broj primera u trening skupu: {len(dataset['train'])}")
    if 'validation' in dataset:
        print(f"Dataset '{dataset_name}' sadrži validacioni skup.")
        return dataset
    else:
        print(f"Dataset '{dataset_name}' ne sadrži validacioni skup. Podela trening skupa na 80/20")

        # Podela trening skupa na trening i validacioni skup
        try:
            print(f"Broj primera u trening skupu: {len(dataset)}")
            train_dataset = dataset['train']
            train_dataset_list = list(train_dataset)
            train_examples, validation_examples = train_test_split(
                train_dataset_list, test_size=test_size, random_state=seed
            )
            # Kreiranje DatasetDict objekta sa podeljenim podacima
            dataset_split = DatasetDict({
                'train': train_examples,
                'validation': validation_examples,
                'test': dataset['test'] if 'test' in dataset else None  # Zadržava test skup ako postoji
            })

            return dataset_split
        except Exception as e:
            logger.error(f"An error! : {e}")
            print(f"An error! : {e}")
            raise  # Re-raise the exception after logging