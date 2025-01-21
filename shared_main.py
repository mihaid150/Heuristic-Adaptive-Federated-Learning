import uvicorn
from fastapi import FastAPI, HTTPException

from fed_node.fed_node import FedNodeType, FedNode, ParentFedNode, ChildFedNode
from fed_node.node_state import NodeState
from request_templates import NodeRequest

app = FastAPI()


@app.post("/initialize")
def initialize_node(request: NodeRequest) -> dict[
                    str, str | dict[str, str | FedNodeType | int]]:
    """
    Initialize the current node.
    """
    if NodeState.get_current_node() is not None:
        raise HTTPException(status_code=400, detail="Current node is already initialized.")

    node = FedNode(None, request.name, request.node_type, request.ip_address, request.port)
    NodeState.initialize_node(node)

    return {
        "message": "Node initialized successfully.",
        "node": {
            "id": node.id,
            "name": request.name,
            "type": request.node_type,
            "ip_address": request.ip_address,
            "port": request.port,
        },
    }


@app.post("/set-parent")
def set_parent_node(request: NodeRequest) -> dict[str, str | dict[str, str | FedNodeType | int]]:
    """
    Call to set the parent of the current working node
    :param request:
    :return:
    """
    current_node = NodeState.get_current_node()

    if current_node is None:
        raise ValueError("Current node was not initialized yet")
    parent_node = ParentFedNode(request.id, request.name, request.node_type, request.ip_address, request.port)
    current_node.set_parent_node(parent_node)

    return {
        "message": "Parent node of the current working node set successfully.",
        "parent_node": {
            "id": request.id,
            "name": request.name,
            "type": request.node_type,
            "ip_address": request.ip_address,
            "port": request.port,
        },
    }


@app.get("/get-parent")
def get_parent_node() -> dict[str, str | dict[str, str | FedNodeType | int]]:
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
        },
    }


@app.post("/remove-parent/")
def remove_parent_node() -> dict[str, str]:
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


@app.post("/set-child")
def set_child_node(request: NodeRequest) -> dict[str, str | dict[str, str | FedNodeType | int]]:
    """
    Call to set the child of the current working node
    :param request:
    :return:
    """
    current_node = NodeState.get_current_node()

    if current_node is None:
        raise ValueError("Current node was not initialized yet")
    child_node = ChildFedNode(request.id, request.name, request.node_type, request.ip_address, request.port)
    current_node.add_child_node(child_node)

    return {
        "message": "Child node of the current working node set successfully.",
        "child_node": {
            "id": request.id,
            "name": request.name,
            "type": request.node_type,
            "ip_address": request.ip_address,
            "port": request.port,
        },
    }


@app.get("/get-children")
def get_children_nodes() -> dict[str, str | list[dict[str, str | FedNodeType | int]]]:
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
        }
        for child in current_node.child_nodes
    ]

    return {
        "message": "Child nodes retrieved successfully.",
        "children": children,
    }


@app.post("/remove-child/{child_id}")
def remove_child_node(child_id: int) -> dict[str, str | dict[str, str | FedNodeType | int]]:
    """
    Call to set the child of the current working node
    :param child_id:
    :return:
    """
    current_node = NodeState.get_current_node()

    if current_node is None:
        raise ValueError("Current node was not initialized yet")

    current_node.child_nodes = [child_node for child_node in current_node.child_nodes if child_node.id != child_id]

    return {
        "message": "Child node of the current working node removed successfully.",
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
