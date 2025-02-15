# fog_statistics.py
import sys
import json
import matplotlib.pyplot as plt


def plot_time_for_all_edges_readiness(fog_data_readiness):
    """
    Plot the time taken for all edges to be ready for each fog node.

    Args:
        fog_data_readiness (dict): A dictionary where keys are fog names and values are lists of readiness times.
    """
    plt.figure(figsize=(12, 6))

    for fog_name, readiness_times in fog_data_readiness.items():
        plt.plot(readiness_times, label=fog_name, marker='o')

    plt.title('Time for All Edges Readiness for Each Fog')
    plt.xlabel('Index')
    plt.ylabel('Time for All Edges Readiness (seconds)')
    plt.legend(title='Fog Name')

    plt.grid(True)

    plt.savefig('/app/images/elapsed_time_chart_all_edge_readiness.png', format='png')
    plt.close()


def plot_time_for_all_edges_genetic(fog_data_genetic_time):
    """
    Plot the genetic evaluation time for each fog node.

    Args:
        fog_data_genetic_time (dict): A dictionary where keys are fog names and values are lists of genetic evaluation times.
    """
    plt.figure(figsize=(12, 6))

    for fog_name, genetic_times in fog_data_genetic_time.items():
        plt.plot(genetic_times, label=fog_name, marker='o')

    plt.title('Genetic Evaluation Time for Each Fog')
    plt.xlabel('Index')
    plt.ylabel('Time for Genetic Evaluation (seconds)')
    plt.legend(title='Fog Name')

    plt.grid(True)

    plt.savefig('/app/images/elapsed_time_chart_fog_genetic_time.png', format='png')
    plt.close()


def plot_time_received_edge_model(fog_data_received_edge_model):
    """
    Plot the time taken to receive edge models for each fog node.

    Args:
        fog_data_received_edge_model (dict): A dictionary where keys are fog names and values are lists of lists of received times.
    """
    plt.figure(figsize=(12, 6))

    for fog_name, received_model_times in fog_data_received_edge_model.items():
        # Flatten the list of lists into a single list
        flattened_times = [time for model_times in received_model_times for time in model_times]
        plt.plot(flattened_times, label=fog_name, marker='o')

    plt.title('Time Received for Edge Models for Each Fog')
    plt.xlabel('Index')
    plt.ylabel('Time Received (seconds)')
    plt.legend(title='Fog Name')

    plt.grid(True)

    plt.savefig('/app/images/elapsed_time_chart_received_edge_model.png', format='png')
    plt.close()


def plot_time_finish_aggregation_for_each_edge(fog_data_finish_aggregation):
    """
    Plot the time taken to finish aggregation for edge models for each fog node.

    Args:
        fog_data_finish_aggregation (dict): A dictionary where keys are fog names and values are lists of lists of finish aggregation times.
    """
    plt.figure(figsize=(12, 6))

    for fog_name, finish_aggregation_times in fog_data_finish_aggregation.items():
        flattened_times = [time for model_times in finish_aggregation_times for time in model_times]
        plt.plot(flattened_times, label=fog_name, marker='o')

    plt.title('Time Finished Aggregation for Edge Models for Each Fog')
    plt.xlabel('Index')
    plt.ylabel('Time Finish Aggregation (seconds)')
    plt.legend(title='Fog Name')

    plt.grid(True)

    plt.savefig('/app/images/elapsed_time_chart_finish_aggregation_edge_model.png', format='png')
    plt.close()


def process_elapsed_time_data_fog(elapsed_time_map):
    """
    Process the elapsed time data for fog nodes and generate plots for readiness, genetic evaluation,
    received edge model times, and finish aggregation times.

    Args:
        elapsed_time_map (dict): A dictionary where keys are fog names and values are lists of dictionaries
                                 containing various elapsed time metrics.
    """
    fog_data_readiness = {}
    fog_data_genetic_time = {}
    fog_data_received_edge_model = {}
    fog_data_finish_aggregation = {}

    for fog_name, elapsed_times in elapsed_time_map.items():
        readiness_times = []
        genetic_evaluation_times = []
        received_edge_model_times = []
        finish_aggregation_times = []

        for elapsed_time in elapsed_times:
            readiness_time = elapsed_time['timeForAllEdgesReadiness']
            genetic_time = elapsed_time['timeGeneticEvaluation']
            received_edge_time = elapsed_time['timeReceivedEdgeModel']
            finish_aggregation_time = elapsed_time['timeFinishAggregation']

            readiness_times.append(readiness_time)
            genetic_evaluation_times.append(genetic_time)
            received_edge_model_times.append(received_edge_time)
            finish_aggregation_times.append(finish_aggregation_time)

        fog_data_readiness[fog_name] = readiness_times
        fog_data_genetic_time[fog_name] = genetic_evaluation_times
        fog_data_received_edge_model[fog_name] = received_edge_model_times
        fog_data_finish_aggregation[fog_name] = finish_aggregation_times

    plot_time_for_all_edges_readiness(fog_data_readiness)
    plot_time_for_all_edges_genetic(fog_data_genetic_time)
    plot_time_received_edge_model(fog_data_received_edge_model)
    plot_time_finish_aggregation_for_each_edge(fog_data_finish_aggregation)


if __name__ == "__main__":
    """
    Main script to process elapsed time data for fog nodes.

    Args:
        sys.argv[1] (str): JSON string containing the elapsed time map.

    Returns:
        None
    """
    json_data = sys.argv[1]
    elapsed_time_map_arg = json.loads(json_data)
    process_elapsed_time_data_fog(elapsed_time_map_arg)
