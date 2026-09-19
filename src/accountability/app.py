from fastapi import FastAPI


app = FastAPI(title="Verifiable Machine Accountability", version="0.1.0")


@app.get("/health", tags=["operations"])
def health() -> dict[str, str]:
    return {"status": "ok"}

