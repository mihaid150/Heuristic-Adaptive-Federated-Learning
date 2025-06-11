#!/bin/bash

SERVERS_FOG=(
  # List all fog nodes available in the network and configured
  "<fog_node1_username>@<fog_node1_ip_address>"
  "<fog_node2_username>@<fog_node2_ip_address>"
  "<fog_node3_username>@<fog_node3_ip_address>"
)

PASSWORD="fog_node_password"

# Complete the path project repository folders for you workstation
LOCAL_FOG_FILES="/cygdrive/c/Users/.../fog_node/"
LOCAL_COMMON_FILES="/cygdrive/c/Users/.../shared/"

REMOTE_PATH="~/fog_node_app"
# EXCLUDE_PATTERN=("*.git" "*.log" "node_modules" "temp/")
EXCLUDE_PATTERN=("__pycache__/")

# Function to check if a file or folder exists
check_existence() {
    local path=$1
    if [ ! -e "$path" ]; then
        echo "Error: $path does not exist. Skipping..."
        return 1
    fi
    return 0
}

# Function to build exclude options for rsync
build_exclude_options() {
    local exclude_opts=()
    for pattern in "${EXCLUDE_PATTERN[@]}"; do
        exclude_opts+=("--exclude=$pattern")
    done
    echo "${exclude_opts[@]}"
}

# Function to transfer files or folders
transfer_files() {
    local source=$1
    local server=$2
    local exclude_opts=$(build_exclude_options)

    if [ -d "$source" ]; then
        echo "Starting transfer of folder $source to $server:$REMOTE_PATH ..."
        sshpass -p "$PASSWORD" rsync -avz $exclude_opts "$source" "$server:$REMOTE_PATH/$(basename "$source")"
    else
        echo "Starting transfer of file $source to $server:$REMOTE_PATH ..."
        sshpass -p "$PASSWORD" rsync -avz $exclude_opts "$source" "$server:$REMOTE_PATH"
    fi

    if [ $? -eq 0 ]; then
        echo "Successfully transferred $source to $server:$REMOTE_PATH"
    else
        echo "Error: Failed to transfer $source to $server:$REMOTE_PATH"
    fi
}

# Main transfer logic
for server in "${SERVERS_FOG[@]}"; do
    echo "Processing server: $server"

    # Check and transfer LOCAL_FOG_FILES
    check_existence "$LOCAL_FOG_FILES"
    if [ $? -eq 0 ]; then
        transfer_files "$LOCAL_FOG_FILES" "$server"
    fi

   # Check and transfer LOCAL_COMMON_FILES
    check_existence "$LOCAL_COMMON_FILES"
    if [ $? -eq 0 ]; then
        transfer_files "$LOCAL_COMMON_FILES" "$server"
    fi
done

echo "All transfers completed."
