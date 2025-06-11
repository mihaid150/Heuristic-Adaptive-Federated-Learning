#!/bin/bash
# Cache Backup and Restore Script for Cloud and Fog Nodes
# This script provides two functionalities:
# 1) Backup the 'cache' folder by copying it into the 'backup' folder with a timestamp and a description message.
# 2) Restore a selected backup version of the 'cache' folder (based on the timestamp) from the 'backup' folder
#    to replace the current 'cache' folder (the backup message file is not copied back).
#
# Operations are executed in parallel on Cloud and Fog nodes using SSH.

# Define server lists
SERVER_CLOUD=("<cloud_node_username>@<cloud_node_ip_address>")
SERVERS_FOG=(
  # List all fog nodes available in the network and configured
  "<fog_node1_username>@<fog_node1_ip_address>"
  "<fog_node2_username>@<fog_node2_ip_address>"
  "<fog_node3_username>@<fog_node3_ip_address>"
)

# Credentials
PASSWORD_CLOUD="cloud_node_password"
PASSWORD_FOG="fog_node_password"

# Remote base paths (adjust if needed)
REMOTE_PATH_CLOUD="~/cloud_node_app"
REMOTE_PATH_FOG="~/fog_node_app"

#-----------------------------------------------------------
# Function to perform backup on a single server:
#  - Creates (if needed) the backup folder
#  - Copies "cache" into backup with a new folder name "cache_<timestamp>"
#  - Writes the provided description into backup_message.txt inside that folder
#-----------------------------------------------------------
exe_backup() {
  local password="$1"
  local server="$2"
  local remote_path="$3"
  local ts="$4"
  local msg="$5"

  sshpass -p "$password" ssh -o StrictHostKeyChecking=no "$server" "
    cd $remote_path &&
    mkdir -p backup &&
    cp -r cache backup/cache_${ts} &&
    echo \"$msg\" > backup/cache_${ts}/backup_message.txt
  " 2>&1 | while IFS= read -r line; do
    echo "[$server]: $line"
  done
}

#-----------------------------------------------------------
# Function to perform restore on a single server:
#  - Removes the current "cache" folder
#  - Copies the selected backup (backup/cache_<timestamp>) back to "cache"
#  - Removes the backup_message.txt from the restored cache folder
#-----------------------------------------------------------
exe_restore() {
  local password="$1"
  local server="$2"
  local remote_path="$3"
  local ts="$4"

  sshpass -p "$password" ssh -o StrictHostKeyChecking=no "$server" "
    cd $remote_path &&
    rm -rf cache &&
    cp -r backup/cache_${ts} cache &&
    rm -f cache/backup_message.txt
  " 2>&1 | while IFS= read -r line; do
    echo "[$server]: $line"
  done
}

#-----------------------------------------------------------
# Execute backup in parallel on all Cloud and Fog nodes
#-----------------------------------------------------------
execute_backup_all() {
  local ts="$1"
  local msg="$2"
  local pids=()

  echo "Starting backup on Cloud Nodes..."
  for server in "${SERVER_CLOUD[@]}"; do
    exe_backup "$PASSWORD_CLOUD" "$server" "$REMOTE_PATH_CLOUD" "$ts" "$msg" &
    pids+=($!)
  done

  echo "Starting backup on Fog Nodes..."
  for server in "${SERVERS_FOG[@]}"; do
    exe_backup "$PASSWORD_FOG" "$server" "$REMOTE_PATH_FOG" "$ts" "$msg" &
    pids+=($!)
  done

  # Wait for all background processes to complete
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
}

#-----------------------------------------------------------
# Execute restore in parallel on all Cloud and Fog nodes
#-----------------------------------------------------------
execute_restore_all() {
  local ts="$1"
  local pids=()

  echo "Starting restore on Cloud Nodes..."
  for server in "${SERVER_CLOUD[@]}"; do
    exe_restore "$PASSWORD_CLOUD" "$server" "$REMOTE_PATH_CLOUD" "$ts" &
    pids+=($!)
  done

  echo "Starting restore on Fog Nodes..."
  for server in "${SERVERS_FOG[@]}"; do
    exe_restore "$PASSWORD_FOG" "$server" "$REMOTE_PATH_FOG" "$ts" &
    pids+=($!)
  done

  # Wait for all background processes
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
}

#-----------------------------------------------------------
# Main menu: choose backup or restore
#-----------------------------------------------------------
echo "Select an option:"
echo "1) Backup 'cache' folder (add timestamp & description)"
echo "2) Restore 'cache' folder from a backup version"
read -p "Enter your choice (1 or 2): " choice

if [ "$choice" == "1" ]; then
  # --- Functionality 1: Backup ---
  timestamp=$(date +%Y%m%d_%H%M%S)
  read -p "Enter a description message for this backup: " backup_message
  echo "Backing up 'cache' folder with timestamp: $timestamp"
  execute_backup_all "$timestamp" "$backup_message"
  echo "Backup operation completed on all nodes."

elif [ "$choice" == "2" ]; then
  # --- Functionality 2: Restore ---
  echo "Fetching available backup versions from Cloud Node..."
  # Get backup directories along with their description messages from the first cloud node
  backup_list=$(sshpass -p "$PASSWORD_CLOUD" ssh -o StrictHostKeyChecking=no "${SERVER_CLOUD[0]}" "
    cd $REMOTE_PATH_CLOUD/backup &&
    for d in cache_*; do
      if [ -d \"\$d\" ]; then
        msg=\$( [ -f \"\$d/backup_message.txt\" ] && cat \"\$d/backup_message.txt\" || echo \"No description\" );
        echo \"\$d - \$msg\";
      fi;
    done
  ")

  if [ -z "$backup_list" ]; then
    echo "No backup versions found on the Cloud Node."
    exit 1
  fi

  echo "Available backup versions:"
  IFS=$'\n' read -rd '' -a backup_options <<< "$backup_list"

  # Present a selection menu (using bash 'select')
  PS3="Select the backup version to restore (enter the number): "
  select opt in "${backup_options[@]}"; do
    if [ -n "$opt" ]; then
      # Extract the folder name (before the " - " delimiter)
      backup_folder=$(echo "$opt" | awk -F' - ' '{print $1}')
      # Remove the "cache_" prefix to get the timestamp
      selected_timestamp=${backup_folder#cache_}
      echo "You selected backup with timestamp: $selected_timestamp"
      break
    else
      echo "Invalid selection. Try again."
    fi
  done

  echo "Restoring backup version $selected_timestamp to the active 'cache' folder..."
  execute_restore_all "$selected_timestamp"
  echo "Restore operation completed on all nodes."

else
  echo "Invalid choice. Exiting."
  exit 1
fi
