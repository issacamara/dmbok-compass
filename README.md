# DMBOK Compass

DMBOK Compass is a grounded data-management question-answering workspace. The repository contains independent backend and web workspaces so each can be developed and tested locally.

## Local development

### Backend API

From `backend/`, create a virtual environment and install the test extra:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
pytest
uvicorn app.main:app --reload
```

The API is available at `http://127.0.0.1:8000`; its readiness endpoint is `GET /health`.

### Web SPA

From `web/`, install dependencies and run the checks or development server:

```bash
npm install
npm test
npm run build
npm run dev
```

The Vite development server prints its local URL, normally `http://localhost:5173`.

## Workspace boundaries

- `backend/` owns the FastAPI API and Python tests.
- `web/` owns the React/Vite/TypeScript single-page application and frontend tests.
- The backend and SPA communicate through HTTP contracts; neither workspace imports the other.

