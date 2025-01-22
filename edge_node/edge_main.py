import uvicorn
from fastapi import APIRouter

from shared_main import app
from routing_paths import RoutingPaths
from edge_service import EdgeService

edge_router = APIRouter(
    prefix=RoutingPaths.EDGE_ROUTE
)

app.include_router(edge_router)

if __name__ == "__main__":
    EdgeService.init_process()
    uvicorn.run(app, host="0.0.0.0", port=8083)
