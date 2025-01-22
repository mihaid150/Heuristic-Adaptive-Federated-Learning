import uvicorn
from fastapi import APIRouter

from shared_main import app
from routing_paths import RoutingPaths
from fog_service import FogService

fog_router = APIRouter(
    prefix=RoutingPaths.FOG_ROUTE
)

app.include_router(fog_router)

if __name__ == "__main__":
    FogService.init_process()
    uvicorn.run(app, host="0.0.0.0", port=8082)
