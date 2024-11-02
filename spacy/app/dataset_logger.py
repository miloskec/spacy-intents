import os
from datasets import load_dataset
from custom_logger import CustomLogger
from datetime import datetime

LOG_LEVEL = os.getenv('LOG_LEVEL')
def inspect_dataset(dataset_name, config_name=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        # Initialize the custom logger (example)
        logger = CustomLogger(name='DatasetInspectorLogger', level=LOG_LEVEL, log_file=f"/workspace/logs/dataset_inspection_{timestamp}.log").get_logger()
        # Log the start of the inspection process
        logger.info(f"Starting dataset inspection for {dataset_name} with config {config_name}")

        # Load the dataset with the specified split ("train" in this case)
        dataset = load_dataset(dataset_name, config_name, split="train")
        logger.info(f"Dataset {dataset_name} loaded successfully with config {config_name}")

        # Log the column names and types
        logger.info("Dataset column information:")
        for column_name, column_type in dataset.features.items():
            logger.info(f"Column '{column_name}': {column_type}")

        # Log some examples (first 5 rows) for manual inspection
        logger.info("Dataset example rows (first 5 rows):")
        for i in range(5):
            logger.info(f"Row {i}: {dataset[i]}")

        # Log number of examples in the dataset
        logger.info(f"Total number of examples in the dataset: {len(dataset)}")

        print(f"Dataset {dataset_name} inspected and details logged.")
        return dataset  # Return the dataset for further manipulation if needed

    except FileNotFoundError as fnf_error:
        logger.error(f"FileNotFoundError: {fnf_error}")
        print(f"Error: {fnf_error}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        print(f"An error occurred: {e}")
        raise  # Re-raise the exception after logging

# Example of using the function
# inspect_dataset("snips-joint-intent", config_name="default")
