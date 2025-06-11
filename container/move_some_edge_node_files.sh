#!/bin/bash

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

PASSWORD="edge_node_password"

# Complete the path project repository folders for you workstation
LOCAL_EDGE_FILES="/cygdrive/c/Users/.../edge_node/"
LOCAL_COMMON_FILES="/cygdrive/c/Users/.../shared/"

REMOTE_PATH="~/edge_node_app"
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
    local exclude_opts
    exclude_opts=$(build_exclude_options)

    # Define the SSH options to disable host key checking and avoid known_hosts update.
    local ssh_opts="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

    if [ -d "$source" ]; then
        echo "Starting transfer of folder $source to $server:$REMOTE_PATH ..."
        sshpass -p "$PASSWORD" rsync -avz -e "ssh $ssh_opts" $exclude_opts "$source" "$server:$REMOTE_PATH/$(basename "$source")"
    else
        echo "Starting transfer of file $source to $server:$REMOTE_PATH ..."
        sshpass -p "$PASSWORD" rsync -avz -e "ssh $ssh_opts" $exclude_opts "$source" "$server:$REMOTE_PATH"
    fi

    if [ $? -eq 0 ]; then
        echo "Successfully transferred $source to $server:$REMOTE_PATH"
    else
        echo "Error: Failed to transfer $source to $server:$REMOTE_PATH"
    fi
}

# Display available edge servers with indexes
echo "Available edge servers:"
for i in "${!SERVERS_EDGE[@]}"; do
    echo "[$i] ${SERVERS_EDGE[$i]}"
done

# Ask user to input indexes of servers to apply file move operations
read -p "Enter the indexes of the edge servers to apply file move operations (separated by space): " -a selected_indexes

# Main transfer logic for selected servers
for index in "${selected_indexes[@]}"; do
    # Validate index is a valid number and exists in the array
    if [[ $index =~ ^[0-9]+$ ]] && [ "$index" -ge 0 ] && [ "$index" -lt "${#SERVERS_EDGE[@]}" ]; then
        server="${SERVERS_EDGE[$index]}"
        echo "Processing server: $server"

        # Check and transfer LOCAL_EDGE_FILES
        check_existence "$LOCAL_EDGE_FILES"
        if [ $? -eq 0 ]; then
            file_name=$(basename "$LOCAL_EDGE_FILES")
            echo "[$server] Deleting remote file: $REMOTE_PATH/$file_name"
            sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$server" "rm -f \"$REMOTE_PATH/$file_name\""
            transfer_files "$LOCAL_EDGE_FILES" "$server"
        fi

        # Check and transfer LOCAL_COMMON_FILES
        check_existence "$LOCAL_COMMON_FILES"
        if [ $? -eq 0 ]; then
            file_name=$(basename "$LOCAL_COMMON_FILES")
            echo "[$server] Deleting remote file: $REMOTE_PATH/$file_name"
            sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$server" "rm -f \"$REMOTE_PATH/$file_name\""
            transfer_files "$LOCAL_COMMON_FILES" "$server"
        fi
    else
        echo "Invalid index: $index. Skipping..."
    fi
done

echo "All transfers completed."

