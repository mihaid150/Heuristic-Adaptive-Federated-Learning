from enum import Enum
from typing import Optional, List
import time


class FedNodeType(Enum):
    CLOUD_NODE = 1
    FOG_NODE = 2
    EDGE_NODE = 3


def generate_unique_id(ip: str) -> str:
    """
    Generate a 64-bit unique ID by combining:
      - A 32-bit time component from time.time_ns() modulo 2^32, and
      - A 32-bit integer representation of the IP address.

    The ID structure (from high to low bits):
      [ time_component (32 bits) | ip_component (32 bits) ]
    """

    # Convert an IP address (x.x.x.x) to a 32-bit integer.
    parts = ip.split('.')
    # Convert the IP to a 32-bit integer
    ip_int = (int(parts[0]) << 24) | (int(parts[1]) << 16) | (int(parts[2]) << 8) | int(parts[3])
    # Use the lower 32 bits of time.time_ns() to get a time component
    time_component = time.time_ns() % (1 << 32)
    # Combine them: shift the time component to the high 32 bits and OR with the IP component
    unique_id = (time_component << 32) | ip_int
    return str(unique_id)


class FedNode:
    def __init__(self, node_id: str | None, name: str, fed_node_type: FedNodeType, ip_address: str, port: int) -> None:
        self.id = generate_unique_id(ip_address) if node_id is None else node_id
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

    def add_child_nodes(self, child_nodes: List['ChildFedNode']) -> None:
        """
        Add multiple child nodes to this node.
        :param child_nodes: List of ChildFedNode objects to add as children.
        """
        self.child_nodes.extend(child_nodes)


class ParentFedNode(FedNode):
    def __init__(self, node_id: str, name: str, fed_node_type: FedNodeType, ip_address: str, port: int) -> None:
        super().__init__(node_id, name, fed_node_type, ip_address, port)


class ChildFedNode(FedNode):
    def __init__(self, node_id: str, name: str, fed_node_type: FedNodeType, ip_address: str, port: int) -> None:
        super().__init__(node_id, name, fed_node_type, ip_address, port)
        self.is_evaluation_node: bool = False
