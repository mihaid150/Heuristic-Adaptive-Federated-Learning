#!/bin/bash

# Define servers
SERVER_CLOUD=("<cloud_node_username>@<cloud_node_ip_address>")
SERVERS_EDGE=(
  # List all available edge nodes
  "<edge1_node_username>@<edge1_node_ip_address>"
  "<edge2_node_username>@<edge2_node_ip_address>"
  "<edge3_node_username>@<edge3_node_ip_address>"
  "<edge4_node_username>@<edge4_node_ip_address>"
  "<edge5_node_username>@<edge5_node_ip_address>"
  "<edge6_node_username>@<edge6_node_ip_address>"
  "<edge7_node_username>@<edge7_node_ip_address>"
  "<edge8_node_username>@<edge8_node_ip_address>"
  "<edge9_node_username>@<edge9_node_ip_address>"
)
SERVERS_FOG=(
  # List all fog nodes available in the network and configured
  "<fog_node1_username>@<fog_node1_ip_address>"
  "<fog_node2_username>@<fog_node2_ip_address>"
  "<fog_node3_username>@<fog_node3_ip_address>"
)

# Passwords for all servers
PASSWORD_EDGE="edge_node_password"
PASSWORD_CLOUD="cloud_node_password"
PASSWORD_FOG="fog_node_password"

# Function to determine the correct password based on the server's username
get_password() {
  local server="$1"
  local user="${server%@*}"
  if [[ $user == pcnode0* ]]; then
    echo "$PASSWORD_CLOUD"
  elif [[ $user == pinode* ]]; then
    echo "$PASSWORD_EDGE"
  else
    echo "$PASSWORD_FOG"
  fi
}

exe_docker_compose_down_up() {
  local password="$1"
  local server="$2"
  local remote_dir

  # Extract username from the server string
  local user="${server%@*}"  # part before '@'

  # Define the remote directory dynamically based on the user
  if [[ $user == pcnode0* ]]; then
    remote_dir="/home/$user/cloud_node_app/cloud_node"
  elif [[ $user == pinode* ]]; then
    remote_dir="/home/$user/edge_node_app/edge_node"
  else
    remote_dir="/home/$user/fog_node_app/fog_node"
  fi

  echo "Restarting services on $server with remote path $remote_dir..."

  # Execute docker-compose down and up on the remote server
  sshpass -p "$password" ssh -o StrictHostKeyChecking=no "$server" <<EOF
    cd $remote_dir
    echo "[$server]: Stopping services..."
    docker-compose down
    echo "[$server]: Starting services..."
    docker-compose up -d
EOF
}

# Execute docker-compose operations in parallel for an array of servers
execute_in_parallel() {
  local servers=("$@")
  local pids=()

  for server in "${servers[@]}"; do
    local pass
    pass=$(get_password "$server")
    exe_docker_compose_down_up "$pass" "$server" &
    pids+=($!)  # Save the PID of the background process
  done

  # Wait for all background processes to complete
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
}

# Flatten all groups into a single list
ALL_SERVERS=("${SERVER_CLOUD[@]}" "${SERVERS_FOG[@]}" "${SERVERS_EDGE[@]}")

# Display available servers with indexes
echo "Available Servers:"
for i in "${!ALL_SERVERS[@]}"; do
  echo "[$i] ${ALL_SERVERS[$i]}"
done

# Prompt user to select the nodes to target
read -p "Enter the indexes of the servers to restart (space-separated): " -a selected_indexes

# Build array of selected servers
selected_servers=()
for index in "${selected_indexes[@]}"; do
  if [[ $index =~ ^[0-9]+$ ]] && [ "$index" -ge 0 ] && [ "$index" -lt "${#ALL_SERVERS[@]}" ]; then
    selected_servers+=("${ALL_SERVERS[$index]}")
  else
    echo "Invalid index: $index. Skipping..."
  fi
done

# Execute docker-compose down/up commands in parallel on selected servers
if [ ${#selected_servers[@]} -gt 0 ]; then
  echo "Restarting services on selected servers in parallel..."
  execute_in_parallel "${selected_servers[@]}"
else
  echo "No valid servers selected. Exiting..."
fi

