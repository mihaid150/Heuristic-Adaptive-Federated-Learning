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

exe_docker_compose_down_up() {
  local password="$1"
  local server="$2"
  local remote_dir

  # Extract username from the server string
  local user="${server%@*}"  # part before '@'
  local host="${server#*@}"  # part after '@'

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

execute_in_parallel() {
  local password="$1"
  shift
  local servers=("$@")
  local pids=()

  for server in "${servers[@]}"; do
    exe_docker_compose_down_up "$password" "$server" &
    pids+=($!)
  done

  for pid in "${pids[@]}"; do
    wait "$pid"
  done
}

# Ask which groups to target
echo "Available groups:"
echo "1 - Cloud Servers"
echo "2 - Fog Servers"
echo "3 - Edge Servers"
read -p "Enter the numbers of the groups you want to target (space-separated): " -a groups

for group in "${groups[@]}"; do
  case $group in
    1)
      echo "Restarting Cloud Servers..."
      execute_in_parallel "$PASSWORD_CLOUD" "${SERVER_CLOUD[@]}"
      ;;
    2)
      echo "Restarting Fog Servers..."
      execute_in_parallel "$PASSWORD_FOG" "${SERVERS_FOG[@]}"
      ;;
    3)
      echo "Restarting Edge Servers..."
      execute_in_parallel "$PASSWORD_EDGE" "${SERVERS_EDGE[@]}"
      ;;
    *)
      echo "Invalid group: $group. Skipping..."
      ;;
  esac
done
