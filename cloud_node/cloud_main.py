import uvicorn
from fastapi import APIRouter, HTTPException

from cloud_node.cloud_service import CloudService
from cloud_node.request_templates import InitProcessRequest
from shared_main import app
from routing_paths import RoutingPaths

cloud_router = APIRouter(
    prefix=RoutingPaths.CLOUD_ROUTE
)

app.include_router(cloud_router)


@cloud_router.get(RoutingPaths.CLOUD_STATUS)
def get_cloud_status():
    """
    :return: retrieves the status of the cloud node
    """
    try:
        return CloudService.get_status()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@cloud_router.post(RoutingPaths.CLOUD_INIT)
def init_cloud_process(request: InitProcessRequest):
    try:
        CloudService.init_process(request.start_date, request.end_date, request.is_cache_active,
                                  request.genetic_evaluation_strategy, request.model_type)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)
