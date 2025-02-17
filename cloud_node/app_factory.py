# cloud_node/app_factory.py

from shared.logging_config import logger
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from cloud_node.cloud_service import CloudService
from shared.shared_main import shared_router
from cloud_node.cloud_main import cloud_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(shared_router, prefix="/node")
app.include_router(cloud_router, prefix="/cloud")


@app.on_event("startup")
async def startup_event():
    logger.info("Startup event: initializing monitoring thread...")
    CloudService.start_monitoring_current_node()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)
