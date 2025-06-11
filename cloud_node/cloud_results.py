from cloud_node.db_manager import save_node_record_to_db, clear_all_data, update_node_records_and_relink_ids


def execute_clear_cloud_results():
    clear_all_data()
    return {"message": "Successfully cleared cloud results from the database."}


def execute_save_node_record_to_db(data):
    try:
        # data is expected to be a list of node objects.
        save_node_record_to_db(data)
        return {"message": "Node records saved successfully."}
    except Exception as e:
        return {"error": str(e)}


def execute_update_node_records_and_relink_ids(data):
    try:
        update_node_records_and_relink_ids(data)
        return {"message": "Update the node records an relink ids successfully."}
    except Exception as e:
        return {"error": str(e)}
