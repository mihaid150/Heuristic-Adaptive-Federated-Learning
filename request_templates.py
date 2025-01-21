from pydantic import BaseModel

from fed_node.fed_node import FedNodeType


class NodeRequest(BaseModel):
    id: int | None
    name: str
    node_type: FedNodeType
    ip_address: str
    port: int
