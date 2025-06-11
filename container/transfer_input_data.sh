#!/bin/bash

# Complete the right edges usernames and ip address together with the local path (...) of the datasets to transfer them on the corresponding edges
declare -A TRANSFER_MAP=(
    ["<edge2_node_username>@<edge2_node_ip_address>"]="/cygdrive/c/Users/.../data_source/pinode2/input_data.csv:~/edge_node_app/edge_node/data/"
    ["<edge3_node_username>@<edge3_node_ip_address>"]="/cygdrive/c/Users/.../data_source/pinode3/input_data.csv:~/edge_node_app/edge_node/data/"
    ["<edge4_node_username>@<edge4_node_ip_address>"]="/cygdrive/c/Users/.../data_source/pinode4/input_data.csv:~/edge_node_app/edge_node/data/"
    ["<edge5_node_username>@<edge5_node_ip_address>"]="/cygdrive/c/Users/.../data_source/pinode5/input_data.csv:~/edge_node_app/edge_node/data/"
    ["<edge6_node_username>@<edge6_node_ip_address>"]="/cygdrive/c/Users/.../data_source/pinode6/input_data.csv:~/edge_node_app/edge_node/data/"
    ["<edge7_node_username>@<edge7_node_ip_address>"]="/cygdrive/c/Users/.../data_source/pinode7/input_data.csv:~/edge_node_app/edge_node/data/"
    ["<edge8_node_username>@<edge8_node_ip_address>"]="/cygdrive/c/Users/.../data_source/pinode8/input_data.csv:~/edge_node_app/edge_node/data/"
    ["<edge9_node_username>@<edge9_node_ip_address>"]="/cygdrive/c/Users/.../data_source/pinode9/input_data.csv:~/edge_node_app/edge_node/data/"
    ["<edge10_node_username>@<edge10_node_ip_address>"]="/cygdrive/c/Users/.../data_source/pinode10/input_data.csv:~/edge_node_app/edge_node/data/"
)

PASSWORD="edge_node_password"

check_existence() {
    local path="$1"
    if [ ! -e "$path" ]; then
        echo "Error: $path does not exist. Skipping..."
        return 1
    fi
    return 0
}

transfer_file() {
    local host="$1"
    local local_path="$2"
    local mapping="$3"

    # Split mapping into local file and remote destination directory
    IFS=":" read -r local_file remote_path <<< "$mapping"
    local file_name
    file_name=$(basename "$local_file")

    # Delete the remote file (if any) before transferring
    echo "[$host] Deleting remote file: $remote_path/$file_name"
    sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$host" "rm -f \"$remote_path/$file_name\""
    if [ $? -eq 0 ]; then
        echo "[$host] Remote file $remote_path/$file_name deleted (if existed)."
    else
        echo "[$host] Error: Failed to delete remote file $remote_path/$file_name."
    fi

    # Transfer the file using rsync with SSH options to bypass host key checking.
    echo "[$host] Transferring file from $local_path to $host:$remote_path ..."
    sshpass -p "$PASSWORD" rsync -avz -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" "$local_path" "$host:$remote_path"
    if [ $? -eq 0 ]; then
        echo "[$host] Successfully transferred $local_path to $host:$remote_path."
    else
        echo "[$host] Error: Failed to transfer $local_path to $host:$remote_path."
    fi
}

# Iterate over each host and transfer the file in parallel
for host in "${!TRANSFER_MAP[@]}"; do
    IFS=":" read -r local_path remote_path <<< "${TRANSFER_MAP[$host]}"
    # Check if the local file exists
    check_existence "$local_path"
    if [ $? -eq 0 ]; then
        # Run each transfer in the background
        transfer_file "$host" "$local_path" "${TRANSFER_MAP[$host]}" &
    fi
done

# Wait for all parallel transfers to complete
wait

echo "All transfers completed."
