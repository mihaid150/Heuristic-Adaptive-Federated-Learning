from enum import Enum
from typing import Optional, List
import time


class FedNodeType(Enum):
    CLOUD_NODE = 1
    FOG_NODE = 2
    EDGE_NODE = 3


class FedNode:
    def __init__(self, node_id: int | None, name: str, fed_node_type: FedNodeType, ip_address: str, port: int) -> None:
        self.id = str(int(time.time())) if node_id is None else node_id
        self.name = name
        self.fed_node_type = fed_node_type
        self.ip_address = ip_address
        self.port = port
        self.parent_node: Optional['ParentFedNode'] = None
        self.child_nodes: List['ChildFedNode'] = []

    def set_parent_node(self, parent_node: Optional['ParentFedNode']) -> None:
        """
        Set the parent node for this node.
        """
        self.parent_node = parent_node

    def add_child_node(self, child_node: 'ChildFedNode') -> None:
        """
        Add a child node to this node.
        """
        self.child_nodes.append(child_node)


class ParentFedNode(FedNode):
    def __init__(self, node_id: int, name: str, fed_node_type: FedNodeType, ip_address: str, port: int) -> None:
        super().__init__(node_id, name, fed_node_type, ip_address, port)


class ChildFedNode(FedNode):
    def __init__(self, node_id: int, name: str, fed_node_type: FedNodeType, ip_address: str, port: int) -> None:
        super().__init__(node_id, name, fed_node_type, ip_address, port)
        self.is_evaluation_node: bool = False


