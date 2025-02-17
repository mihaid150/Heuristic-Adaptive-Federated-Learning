from typing import Optional

from shared.fed_node.fed_node import FedNode


class NodeState:
    current_node: Optional[FedNode] = None

    @staticmethod
    def initialize_node(node: FedNode) -> None:
        if NodeState.current_node is not None:
            raise ValueError("Current working node is already initialized.")
        NodeState.current_node = node

    @staticmethod
    def get_current_node() -> Optional[FedNode]:
        return NodeState.current_node

    @staticmethod
    def reset_node() -> None:
        NodeState.current_node = None
