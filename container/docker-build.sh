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

exe_docker_build() {
  local password="$1"
  local server="$2"
  local remote_dir
  local image_name

  # Extract username from the server string
  local user="${server%@*}" # Extract the part before '@'
  local host="${server#*@}" # Extract the part after '@'

  # Define the remote directory dynamically based on the user
  if [[ $user == pcnode0* ]]; then
    remote_dir="/home/$user/cloud_node_app/cloud_node"
  elif [[ $user == pinode* ]]; then
    remote_dir="/home/$user/edge_node_app/edge_node"
  else
    remote_dir="/home/$user/fog_node_app/fog_node"
  fi

  # Determine the image name based on the directory
  if [[ $remote_dir == *cloud* ]]; then
    image_name="cloud_node_app"
  elif [[ $remote_dir == *fog* ]]; then
    image_name="fog_node_app"
  elif [[ $remote_dir == *edge* ]]; then
    image_name="edge_node_app"
  else
    echo "Unknown directory type for $server, skipping..."
    return
  fi

  echo "Building Docker image on $server with remote path $remote_dir and image name $image_name..."

  # Execute the Docker build command and print the output
  sshpass -p "$password" ssh -o StrictHostKeyChecking=no "$server" \
    "cd $remote_dir && docker build -f Dockerfile -t $image_name:latest .." 2>&1 | while IFS= read -r line; do
    echo "[$server]: $line"
  done
}

execute_in_parallel() {
  local password="$1"
  shift
  local servers=("$@")
  local pids=()

  for server in "${servers[@]}"; do
    exe_docker_build "$password" "$server" &
    pids+=($!) # Save the PID of the background process
  done

  # Wait for all background processes to complete
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
}

# Ask user which groups to target
echo "Available groups:"
echo "1 - Cloud Servers"
echo "2 - Fog Servers"
echo "3 - Edge Servers"
read -p "Enter the numbers of the groups you want to target (space-separated, e.g., 1 2): " -a groups

# Execute the function for the selected groups in parallel
for group in "${groups[@]}"; do
  case $group in
    1)
      echo "Executing for Cloud Servers in parallel..."
      execute_in_parallel "$PASSWORD_CLOUD" "${SERVER_CLOUD[@]}"
      ;;
    2)
      echo "Executing for Fog Servers in parallel..."
      execute_in_parallel "$PASSWORD_FOG" "${SERVERS_FOG[@]}"
      ;;
    3)
      echo "Executing for Edge Servers in parallel..."
      execute_in_parallel "$PASSWORD_EDGE" "${SERVERS_EDGE[@]}"
      ;;
    *)
      echo "Invalid group: $group. Skipping..."
      ;;
  esac
done
