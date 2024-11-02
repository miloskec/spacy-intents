
# spaCy Intent Recognition Project

## 1. Introduction
This project demonstrates fine-tuning spaCy models for intent recognition using well-known datasets. It includes an integrated setup for model training, testing, and dataset conversion, with API endpoints accessible through Postman.

## 2. Features
- Fine-tuning with multiple spaCy models (`en_core_web_trf`, `en_core_web_sm`, `en_core_web_md`).
- Support for datasets like [CLINC OOS](https://huggingface.co/datasets/clinc/clinc_oos) and [SNIPS](https://huggingface.co/datasets/bkonkle/snips-joint-intent).
- Dockerized services for consistent deployment.
- Postman collection provided for easy interaction with the API.

## 3. Project Structure
```
docker-compose/
│
├── docker-compose.yaml           # Orchestrates container services
├── .env                          # Environment configuration file
│
├── gateway/
│   ├── Dockerfile                # Dockerfile for the gateway service
│   ├── requirements.txt          # Dependencies for the gateway
│   └── app/
│       ├── api_server.py         # Main API server script
│       ├── __init__.py           # Package initializer
│
└── spacy/
    ├── Dockerfile                # Dockerfile for the spaCy service
    ├── app/
        ├── dataset_converter.py  # Converts datasets for spaCy training
        ├── __init.py__           # Package initializer
        ├── spacy_trainer.py      # Training script
        ├── spacy_tester.py       # Testing script
        ├── spacy_server.py       # API server for spaCy
        ├── custom_logger.py      # Custom logging utility
        ├── dataset_logger.py     # Dataset inspection logger
        ├── logs/                 # Logs for training and testing
```

## 4. Setup Instructions

### 4.1. Prerequisites
- Docker and Docker Compose
- Python 3.8+ for local development

### 4.2. Environment Configuration
Create an `.env` file:
```plaintext
# Gateway port
GPORT=5005
# SpaCy service port
SPORT=5006

SPACY_MODEL="en_core_web_trf"
LOG_LEVEL="INFO"
```

### 4.3. Build and Run the Containers
```bash
docker-compose up --build
```

## 5. API Endpoints (Postman Collection)

### 5.1. Available Endpoints
- **Convert CLINC OOS Dataset**
  - **Method**: POST
  - **Endpoint**: `/convert`
  - **Description**: Converts the CLINC OOS dataset for training.
  
- **Inspect Dataset**
  - **Method**: GET
  - **Endpoint**: `/inspect`
  - **Description**: Inspects the specified dataset for validation.

- **Train Model**
  - **Method**: POST
  - **Endpoint**: `/train`
  - **Description**: Trains a spaCy model with provided data.

- **Test Model**
  - **Method**: POST
  - **Endpoint**: `/test`
  - **Description**: Tests the spaCy model for accuracy and performance.

### 5.2. Postman Collection
Import the Postman collection from the file `SPACY-TRAINING.postman_collection.json` to easily interact with these endpoints.

## 6. Logs and Monitoring
Logs are stored in `spacy/app/logs/`. Use:
```bash
tail -f spacy/app/logs/spacy_training_<timestamp>.log
```

## 7. Advanced Usage
- **Dataset Conversion**: Run `dataset_converter.py` for formatting data.
- **Training Customization**: Modify `spacy_trainer.py` for training parameters.
- **API Interaction**: Use the Postman collection for streamlined requests.

