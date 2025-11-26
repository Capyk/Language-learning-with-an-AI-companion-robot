from fastapi import FastAPI
from . import endpoints

# Initialize the FastAPI application
app = FastAPI(title="Pepper AI Tutor Backend")

# Include the endpoints defined in endpoints.py
app.include_router(endpoints.router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {"message": "Pepper Tutor Backend is Running"}