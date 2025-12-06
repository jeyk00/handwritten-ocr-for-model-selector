# Model Selector Project

This project implements a handwritten digit recognizer using a PyTorch model, FastAPI backend, and Streamlit frontend.

## Python Version
This project is designed for **Python 3.11**.

## Tech Stack
- **Backend**: FastAPI, PyTorch, Uvicorn
- **Frontend**: Streamlit, OpenCV
- **Infrastructure**: Docker, Docker Compose

## First time setup

Use python3.11.

To setup project for local development (without Docker), run following commands:

```bash
python3.11 -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

### Running with Docker (Recommended)

To run the entire stack:
```bash
docker-compose up --build
```
Access the frontend at `http://localhost:8501`.

