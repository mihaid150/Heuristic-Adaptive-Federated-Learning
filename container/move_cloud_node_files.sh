#!/bin/bash

SERVER_CLOUD=("<cloud_node_username>@<cloud_node_ip_address>")

PASSWORD="cloud_node_password"

# Complete the path project repository folders for you workstation
LOCAL_CLOUD_FILES="/cygdrive/c/Users/.../cloud_node/"
LOCAL_COMMON_FILES="/cygdrive/c/Users/.../shared/"

REMOTE_PATH="~/cloud_node_app"
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
for server in "${SERVER_CLOUD[@]}"; do
    echo "Processing server: $server"

    # Check and transfer LOCAL_CLOUD_FILES
    check_existence "$LOCAL_CLOUD_FILES"
    if [ $? -eq 0 ]; then
        transfer_files "$LOCAL_CLOUD_FILES" "$server"
    fi
   
    # Check and transfer LOCAL_COMMON_FILES
    check_existence "$LOCAL_COMMON_FILES"
    if [ $? -eq 0 ]; then
        transfer_files "$LOCAL_COMMON_FILES" "$server"
    fi
done

echo "All transfers completed."
