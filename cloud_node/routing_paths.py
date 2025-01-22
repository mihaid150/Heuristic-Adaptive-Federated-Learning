from enum import Enum


class RoutingPaths(str, Enum):
    CLOUD_ROUTE = "/cloud"
    CLOUD_INIT = "/initialize-process/"
    CLOUD_STATUS = "/get-status"
