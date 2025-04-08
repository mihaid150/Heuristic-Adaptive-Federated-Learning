import uuid
from enum import Enum
from typing import Optional, List
import subprocess
import re


class FedNodeType(Enum):
    CLOUD_NODE = 1
    FOG_NODE = 2
    EDGE_NODE = 3


class MessageScope(Enum):
    TRAINING = 1
    EVALUATION = 2
    TEST_DATA_ENOUGH_EXISTS = 3
    GENETIC_LOGBOOK = 4
    EVOLUTION_SYSTEM_METRICS = 5


def generate_unique_id(ip: str) -> str:
    """
    Generate a unique identifier using a combination of the IP address and UUID.
    """
    return f"{ip.replace('.', '')}-{uuid.uuid4().hex}"


def get_default_interface() -> Optional[str]:
    """
    Use 'ip route show default' to determine the default network interface.
    """
    try:
        result = subprocess.check_output(["ip", "route", "show", "default"]).decode("utf-8")
        # Look for 'dev <interface>' in the output.
        match = re.search(r"dev (\S+)", result)
        if match:
            return match.group(1)
    except Exception as e:
        print("Error retrieving default interface:", e)
    return None


def get_mac_address() -> str:
    """
    Attempt to retrieve the MAC address for the default interface.
    If that fails, try a list of common interface names.
    """
    interface = get_default_interface()
    if interface:
        try:
            result = subprocess.check_output(["ip", "link", "show", interface]).decode("utf-8")
            mac = re.search(r"link/ether ([0-9a-f:]{17})", result)
            if mac:
                return mac.group(1)
        except subprocess.CalledProcessError as e:
            print(f"Error retrieving MAC for default interface {interface}: {e}")

    # Fallback: try common interface names.
    for iface in ["eth0", "enp2s0", "ens33", "wlan0"]:
        try:
            result = subprocess.check_output(["ip", "link", "show", iface]).decode("utf-8")
            mac = re.search(r"link/ether ([0-9a-f:]{17})", result)
            if mac:
                return mac.group(1)
        except subprocess.CalledProcessError:
            continue

    # If all attempts fail, return a default placeholder.
    return "00:00:00:00:00:00"


class FedNode:
    def __init__(self, node_id: str | None, name: str, fed_node_type: FedNodeType, ip_address: str, port: int) -> None:
        self.id = generate_unique_id(ip_address) if node_id is None else node_id
        self.name = name
        self.fed_node_type = fed_node_type
        self.ip_address = ip_address
        self.port = port
        self.parent_node: Optional['ParentFedNode'] = None
        self.child_nodes: List['ChildFedNode'] = []
        self.device_mac = get_mac_address()

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
        self.last_time_fitness_evaluation_performed_timestamp = None
