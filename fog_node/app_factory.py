# fog_node/app_factory.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from fog_node.fog_service import FogService
from shared.shared_main import shared_router
from shared.logging_config import logger
from fog_node.fog_main import fog_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(shared_router, prefix="/node")
app.include_router(fog_router, prefix="/fog")


@app.on_event("startup")
async def startup_event():
    logger.info("Startup event: initializing monitoring thread...")
    FogService.start_monitoring_parent_children_nodes()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)
