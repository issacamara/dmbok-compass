from fastapi import FastAPI

app = FastAPI(title="DMBOK Compass API", version="0.1.0")


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    """Report that the API process is ready to receive requests."""
    return {"status": "ok"}

