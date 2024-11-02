import os
from dotenv import load_dotenv
import uvicorn

# Load the .env file
load_dotenv()

if __name__ == "__main__":
    uvicorn.run("spacy_server:app", host="0.0.0.0", port=5001, reload=True)