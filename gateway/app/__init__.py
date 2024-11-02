# This file can remain empty, or you can use it to define variables or
# initialize module-specific objects if necessary.
import uvicorn

if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=5000, reload=True)