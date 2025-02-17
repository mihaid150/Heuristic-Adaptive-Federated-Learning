# edge_node/app_factory.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from shared.shared_main import shared_router
from edge_node.edge_main import edge_router
from shared.logging_config import logger
from edge_node.edge_service import EdgeService

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(shared_router, prefix="/node")
app.include_router(edge_router, prefix="/edge")


@app.on_event("startup")
async def startup_event():
    logger.info("Startup event: initializing monitoring thread...")
    EdgeService.init_process()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)
