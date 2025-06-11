# shared/shared_main.py

from typing import Dict, Any, Union, List

from fastapi import HTTPException, WebSocket, WebSocketDisconnect, APIRouter
from shared.fed_node.fed_node import FedNodeType, FedNode, ParentFedNode, ChildFedNode
from shared.fed_node.node_state import NodeState
from shared.logging_config import logger

shared_router = APIRouter()


@shared_router.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            message: Dict[str, Any] = await websocket.receive_json()
            operation: str = message.get('operation')
            data: Dict[str, Any] = message.get('data', {})

            try:
                if operation == "initialize":
                    response = initialize_node(data)
                elif operation == "get_node_info":
                    response = get_node_info()
                elif operation == "set_parent":
                    response = set_parent_node(data)
                elif operation == "get_parent":
                    response = get_parent_node()
                elif operation == "remove_parent":
                    response = remove_parent_node()
                elif operation == "set_child":
                    response = set_child_node(data)
                elif operation == "set_children":
                    response = set_children_nodes(data)
                elif operation == "get_children":
                    response = get_children_nodes()
                elif operation == "remove_child":
                    response = remove_child_node(data)
                else:
                    response = {"error": "Invalid operation"}
            except Exception as e:
                response = {"error": str(e)}

            await websocket.send_json(response)
    except WebSocketDisconnect:
        logger.warning("WebSocket disconnected.")


def initialize_node(data: Dict[str, Any]) -> Dict[str, Union[str, Dict[str, Union[str, FedNodeType]]]]:
    """
    Initialize the current node.
    """
    if NodeState.get_current_node() is not None:
        raise HTTPException(status_code=400, detail="Current node is already initialized.")

    if data.get("id") is not None:
        node_id = data.get("id")
    else:
        node_id = None

    node = FedNode(node_id, data.get("name"), data.get("node_type"), data.get("ip_address"), data.get("port"))
    NodeState.initialize_node(node)

    logger.info(f"Successfully created a new node with id {node.id}, name {node.name}, type "
                f"{FedNodeType(node.fed_node_type).name}, ip {node.ip_address} and port {node.port}.")

    return {
        "message": "Node initialized successfully.",
        "node": {
            "id": node.id,
            "name": node.name,
            "type": node.fed_node_type,
            "ip_address": node.ip_address,
            "port": node.port,
            "device_mac": node.device_mac
        },
    }


def get_node_info() -> Dict[str, Union[str, Dict[str, Union[str, FedNodeType]]]]:
    node = NodeState.get_current_node()

    if node is None:
        raise HTTPException(status_code=400, detail="Current node is not initialized.")

    return {
        "message": "Node initialized successfully.",
        "node": {
            "id": node.id,
            "name": node.name,
            "type": node.fed_node_type,
            "ip_address": node.ip_address,
            "port": node.port,
            "device_mac": node.device_mac
        },
    }


def set_parent_node(data: Dict[str, Any]) -> Dict[str, Union[str, Dict[str, Union[str, FedNodeType]]]]:
    """
    Call to set the parent of the current working node
    :return:
    """
    current_node = NodeState.get_current_node()

    if current_node is None:
        raise ValueError("Current node was not initialized yet")
    parent_node = ParentFedNode(data.get("id"), data.get("name"), data.get("node_type"), data.get("ip_address"),
                                data.get("port"))
    current_node.set_parent_node(parent_node)

    return {
        "message": "Parent node of the current working node set successfully.",
        "parent_node": {
            "id": parent_node.id,
            "name": parent_node.name,
            "type": parent_node.fed_node_type,
            "ip_address": parent_node.ip_address,
            "port": parent_node.port,
            "device_mac": parent_node.device_mac
        },
    }


def get_parent_node() -> Dict[str, Union[str, Dict[str, Union[str, FedNodeType]]]]:
    """
    Retrieve the parent of the current working node.
    :return:
    """
    current_node = NodeState.get_current_node()

    if current_node is None:
        raise HTTPException(status_code=400, detail="Current node was not initialized yet")

    if current_node.parent_node is None:
        return {"message": "No parent node is set for the current working node."}

    parent_node = current_node.parent_node

    return {
        "message": "Parent node retrieved successfully.",
        "parent_node": {
            "id": parent_node.id,
            "name": parent_node.name,
            "type": parent_node.fed_node_type,
            "ip_address": parent_node.ip_address,
            "port": parent_node.port,
            "device_mac": parent_node.device_mac
        },
    }


def remove_parent_node() -> Dict[str, str]:
    """
    Call to set the parent of the current working node
    :return:
    """
    current_node = NodeState.get_current_node()

    if current_node is None:
        raise ValueError("Current node was not initialized yet")
    current_node.parent_node = None

    return {
        "message": "Parent node of the current working node removed successfully.",
    }


def set_child_node(data: Dict[str, Any]) -> Dict[str, Union[str, Dict[str, Union[str, FedNodeType]]]]:
    """
    Call to set the child of the current working node
    :return:
    """
    current_node = NodeState.get_current_node()

    if current_node is None:
        raise ValueError("Current node was not initialized yet")
    child_node = ChildFedNode(data.get("id"), data.get("name"), data.get("node_type"), data.get("ip_address"),
                              data.get("port"))
    current_node.add_child_node(child_node)

    return {
        "message": "Child node of the current working node set successfully.",
        "child_node": {
            "id": child_node.id,
            "name": child_node.name,
            "type": child_node.fed_node_type,
            "ip_address": child_node.ip_address,
            "port": child_node.port,
            "device_mac": child_node.device_mac
        },
    }


def set_children_nodes(data) -> dict[str, str | list[dict[str, str | FedNodeType]]]:
    """
    Add multiple child nodes to the current working node.
    :return: A message and the added child nodes.
    """
    current_node = NodeState.get_current_node()

    if current_node is None:
        raise HTTPException(status_code=400, detail="Current node was not initialized yet")

    child_nodes = [
        ChildFedNode(request.get("id"), request.get("name"), request.get("node_type"), request.get("ip_address"),
                     request.get("port"))
        for request in data
    ]

    current_node.add_child_nodes(child_nodes)

    added_children = [
        {
            "id": child.id,
            "name": child.name,
            "type": child.fed_node_type,
            "ip_address": child.ip_address,
            "port": child.port,
            "device_mac": child.device_mac
        }
        for child in child_nodes
    ]

    return {
        "message": f"Added {len(added_children)} child nodes to the current working node.",
        "children": added_children,
    }


def get_children_nodes() -> Dict[str, Union[str, List[Dict[str, Union[str, FedNodeType]]]]]:
    """
    Retrieve the children of the current working node.
    :return:
    """
    current_node = NodeState.get_current_node()

    if current_node is None:
        raise HTTPException(status_code=400, detail="Current node was not initialized yet")

    if not current_node.child_nodes:
        return {"message": "No child nodes are set for the current working node."}

    children = [
        {
            "id": child.id,
            "name": child.name,
            "type": child.fed_node_type,
            "ip_address": child.ip_address,
            "port": child.port,
            "device_mac": child.device_mac
        }
        for child in current_node.child_nodes
    ]

    return {
        "message": "Child nodes retrieved successfully.",
        "children": children,
    }


def remove_child_node(data: Dict[str, Any]) -> Dict[str, Union[str, Dict[str, Union[str, FedNodeType]]]]:
    """
    Call to set the child of the current working node
    :return:
    """
    current_node = NodeState.get_current_node()

    child_id = data.get("child_id")

    if current_node is None:
        raise ValueError("Current node was not initialized yet")

    current_node.child_nodes = [child_node for child_node in current_node.child_nodes if child_node.id != child_id]

    return {
        "message": "Child node of the current working node removed successfully.",
    }
