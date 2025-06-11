import os
from typing import List, Dict, Any
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, UniqueConstraint
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session, InstrumentedAttribute
from cloud_node.cloud_resources_paths import CloudResourcesPaths
from shared.logging_config import logger

# Define the path for the SQLite database file.
DB_FILE = os.path.join(CloudResourcesPaths.RESULTS_FOLDER_PATH, CloudResourcesPaths.DB_RESULTS_FILE)
DATABASE_URL = f"sqlite:///{DB_FILE}"

# Create SQLAlchemy engine and session factory.
engine = create_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)

# Define the base class.
Base = declarative_base()


# ------------------ ORM Models ------------------


class Evaluation(Base):
    __tablename__ = "evaluation"
    id = Column(Integer, primary_key=True, autoincrement=True)
    evaluation_date = Column(String, unique=True, nullable=False)
    # Relationship to performance records.
    performance_records = relationship("PerformanceRecord", back_populates="evaluation", cascade="all, delete")
    # Relationships to genetic and prediction records.
    genetic_records = relationship("GeneticRecord", back_populates="evaluation", cascade="all, delete")
    prediction_records = relationship("PredictionRecord", back_populates="evaluation", cascade="all, delete")
    system_metrics = relationship("SystemMetricRecord", back_populates="evaluation", cascade="all, delete")


class PerformanceRecord(Base):
    __tablename__ = "performance_record"
    id = Column(Integer, primary_key=True, autoincrement=True)
    evaluation_id = Column(Integer, ForeignKey("evaluation.id"), nullable=False)
    fog_mac = Column(String, nullable=False)
    edge_mac = Column(String, nullable=False)
    mse = Column(Float)
    mae = Column(Float)
    r2 = Column(Float)
    logcosh = Column(Float)
    huber = Column(Float)
    msle = Column(Float)

    evaluation = relationship("Evaluation", back_populates="performance_records")


class GeneticRecord(Base):
    __tablename__ = "genetic_record"
    id = Column(Integer, primary_key=True, autoincrement=True)
    evaluation_id = Column(Integer, ForeignKey("evaluation.id"), nullable=False)
    fog_mac = Column(String, nullable=False)
    generation = Column(Integer, nullable=False)
    nevals = Column(Integer)
    avg = Column(Float)
    std = Column(Float)
    min = Column(Float)
    max = Column(Float)
    genotypic_diversity = Column(Float)
    phenotypic_diversity = Column(Float)

    evaluation = relationship("Evaluation", back_populates="genetic_records")


class PredictionRecord(Base):
    __tablename__ = "prediction_record"
    id = Column(Integer, primary_key=True, autoincrement=True)
    evaluation_id = Column(Integer, ForeignKey("evaluation.id"), nullable=False)
    fog_mac = Column(String, nullable=False)
    edge_mac = Column(String, nullable=False)
    pair_index = Column(Integer, nullable=False)
    real_value = Column(Float)
    predicted_value = Column(Float)

    evaluation = relationship("Evaluation", back_populates="prediction_records")


class SystemMetricRecord(Base):
    __tablename__ = "system_metric_record"
    id = Column(Integer, primary_key=True, autoincrement=True)
    evaluation_id = Column(Integer, ForeignKey("evaluation.id"), nullable=False)
    fog_mac = Column(String, nullable=False)
    generation = Column(Integer, nullable=False)
    cpu_usage = Column(Float)
    memory_usage = Column(Float)
    load_avg = Column(Float)
    normalized_cpu_freq = Column(Float)
    average_sensor_temp = Column(Float)
    resource_load = Column(Float)

    evaluation = relationship("Evaluation", back_populates="system_metrics")


class NodeRecord(Base):
    __tablename__ = "node_record"
    id = Column(Integer, primary_key=True, autoincrement=True)
    node_id = Column(String, nullable=False)
    older_node_id = Column(String, nullable=True)
    node_label = Column(String, nullable=False)
    parent_id = Column(String, nullable=True)
    parent_label = Column(String, nullable=True)
    device_mac = Column(String, nullable=True)


# ------------------ Database Setup ------------------


def init_db():
    """Create all tables in the database."""
    Base.metadata.create_all(engine)
    logger.info("Database initialized with tables.")


def clear_all_data() -> None:
    """
    Clear all data from all tables.
    This function deletes all rows from the prediction, genetic, performance, and evaluation tables.
    """
    session: Session = SessionLocal()
    try:
        # Delete rows from child tables first
        session.query(PredictionRecord).delete()
        session.query(GeneticRecord).delete()
        session.query(PerformanceRecord).delete()
        session.query(SystemMetricRecord).delete()
        # Then delete rows from the parent table
        session.query(Evaluation).delete()
        session.query(NodeRecord).delete()
        session.commit()
        logger.info("All tables cleared successfully.")
    except Exception as e:
        session.rollback()
        logger.error("Error clearing tables: %s", e)
    finally:
        session.close()


# ------------------ Save Functions ------------------

def save_node_record_to_db(nodes: List[Dict[str, Any]]) -> None:
    session: Session = SessionLocal()

    session.query(NodeRecord).delete()
    session.commit()
    try:
        for node in nodes:
            node_record = NodeRecord(
                node_id=node.get("id"),
                older_node_id="",
                node_label=node.get("label"),
                parent_id=node.get("parent_id"),
                parent_label=node.get("parent_label"),
                device_mac=node.get("device_mac"),
            )
            session.add(node_record)
        session.commit()
        logger.info("Node records saved successfully.")
    except Exception as e:
        session.rollback()
        logger.error("Error saving node records: %s", e)
        raise e
    finally:
        session.close()


def update_node_records_and_relink_ids(nodes: List[Dict[str, Any]]) -> None:
    """
    Receives a list of node objects from the frontend. For each node:
      - If a record with the same node_label already exists, update it by:
          - Storing the old node_id in older_node_id
          - Setting node_id to the new id from frontend
          - Updating parent_id and parent_label accordingly
      - Otherwise, insert a new NodeRecord.
    After updating NodeRecord, update other tables (PerformanceRecord, GeneticRecord,
    PredictionRecord) by replacing any occurrence of an old id with the new id.
    Before and after these updates, logs the unique ids in the related tables.
    """
    session: Session = SessionLocal()
    try:
        # Build mapping from old_id to new_id.
        id_mapping = {}

        # Process each node from the payload.
        # for new_node in nodes:
        #     new_id = new_node.get("id")
        #     label = new_node.get("label")
        #     parent_id = new_node.get("parent_id")  # parent's local id as received
        #     parent_label = new_node.get("parent_label")
        #     mac_address = new_node.get("device_mac")
        #
        #     # Try to find an existing node record with the same mac.
        #     existing_record = session.query(NodeRecord).filter_by(device_mac=mac_address).first()
        #
        #     if existing_record:
        #         old_id = existing_record.node_id
        #         # Update existing record.
        #         existing_record.older_node_id = old_id
        #         existing_record.node_id = new_id
        #         existing_record.parent_id = parent_id
        #         existing_record.parent_label = parent_label
        #         existing_record.device_mac = mac_address
        #         # Map old id to new id.
        #         id_mapping[old_id] = new_id
        #     else:
        #         # Insert new record if not found.
        #         new_record = NodeRecord(
        #             node_id=new_id,
        #             older_node_id="",
        #             node_label=label,
        #             parent_id=parent_id,
        #             parent_label=parent_label,
        #             device_mac=mac_address,
        #         )
        #         session.add(new_record)
        # session.commit()
        # logger.info("Node records updated successfully.")
        # logger.info(f"ID Mapping: {id_mapping}")
        #
        # # Now, update all other tables that reference these node ids.
        # for old_id, new_id in id_mapping.items():
        #     session.query(PerformanceRecord).filter(PerformanceRecord.fog_id == old_id).update(
        #         {PerformanceRecord.fog_id: new_id}, synchronize_session=False)
        #     session.query(PerformanceRecord).filter(PerformanceRecord.edge_id == old_id).update(
        #         {PerformanceRecord.edge_id: new_id}, synchronize_session=False)
        #     session.query(GeneticRecord).filter(GeneticRecord.fog_id == old_id).update(
        #         {GeneticRecord.fog_id: new_id}, synchronize_session=False)
        #     session.query(PredictionRecord).filter(PredictionRecord.fog_id == old_id).update(
        #         {PredictionRecord.fog_id: new_id}, synchronize_session=False)
        #     session.query(PredictionRecord).filter(PredictionRecord.edge_id == old_id).update(
        #         {PredictionRecord.edge_id: new_id}, synchronize_session=False)
        #     session.query(SystemMetricRecord).filter(SystemMetricRecord.fog_id == old_id).update(
        #         {SystemMetricRecord.fog_id: new_id}, synchronize_session=False)
        # session.commit()
        logger.info("Re-linked node ids in related tables successfully.")

    except Exception as e:
        session.rollback()
        logger.error("Error updating node records and relinking ids: %s", e)
        raise e
    finally:
        session.close()


def save_performance_results_to_db(current_working_date: str,
                                   received_fog_performance_results: List[Dict[str, Any]]) -> None:
    """
    Save numeric performance evaluation results into the database.
    `received_fog_performance_results` is a list of fog records, each containing a fog_id and a list of edge records.
    """
    if not current_working_date:
        logger.error("No current working date provided; cannot save performance results.")
        return

    session: Session = SessionLocal()
    try:
        evaluation = session.query(Evaluation).filter_by(evaluation_date=current_working_date).first()
        if not evaluation:
            evaluation = Evaluation(evaluation_date=current_working_date)
            session.add(evaluation)
            session.flush()  # assign an id

        for fog_record in received_fog_performance_results:
            fog_mac = fog_record.get("fog_mac")
            results = fog_record.get("results", [])
            for rec in results:
                edge_mac = rec.get("edge_mac")
                metrics = rec.get("metrics", {})
                record = PerformanceRecord(
                    evaluation_id=evaluation.id,
                    fog_mac=fog_mac,
                    edge_mac=edge_mac,
                    mse=metrics.get("mse"),
                    mae=metrics.get("mae"),
                    r2=metrics.get("r2"),
                    logcosh=metrics.get("logcosh"),
                    huber=metrics.get("huber"),
                    msle=metrics.get("msle")
                )
                session.add(record)
        session.commit()
        logger.info("Performance results for %s saved to the database.", current_working_date)
    except Exception as e:
        session.rollback()
        logger.error("Error saving performance results: %s", e)
    finally:
        session.close()


def save_genetic_results_to_db(current_working_date: str,
                               received_fog_genetic_results: List[Dict[str, Any]]) -> None:
    """
    Save genetic evaluation results into the database.
    Each record in received_fog_genetic_results is expected to have:
      - "fog_id" (optional)
      - Either "records": a list of generation dictionaries, or "record": a single generation dictionary,
        with keys:
          "gen", "nevals", "avg", "std", "min", "max", "genotypic_diversity", "phenotypic_diversity"
    """
    if not current_working_date:
        logger.error("No current working date provided; cannot save genetic results.")
        return

    session: Session = SessionLocal()
    try:
        evaluation = session.query(Evaluation).filter_by(evaluation_date=current_working_date).first()
        if not evaluation:
            evaluation = Evaluation(evaluation_date=current_working_date)
            session.add(evaluation)
            session.flush()  # assign an id

        for record in received_fog_genetic_results:
            fog_mac = record.get("fog_mac")

            # First, try to retrieve generation records from the "records" key.
            genetic_data = record.get("records")
            # If "records" key isn't present, try "record".
            if genetic_data is None:
                genetic_data = record.get("record")
            # If no genetic data is found, default to an empty list.
            if genetic_data is None:
                genetic_data = []
            # If genetic_data is a single dict, wrap it in a list.
            if isinstance(genetic_data, dict):
                genetic_data = [genetic_data]

            for gen_rec in genetic_data:
                gen_number = gen_rec.get("gen")
                # Extract all desired fields from the generation record.
                nevals = gen_rec.get("nevals")
                avg_val = gen_rec.get("avg")
                std_val = gen_rec.get("std")
                min_val = gen_rec.get("min")
                max_val = gen_rec.get("max")
                genotypic_diversity = gen_rec.get("genotypic_diversity")
                phenotypic_diversity = gen_rec.get("phenotypic_diversity")
                # Only add if the required fields exist.
                if gen_number is not None and avg_val is not None:
                    genetic = GeneticRecord(
                        evaluation_id=evaluation.id,
                        fog_mac=fog_mac,
                        generation=gen_number,
                        nevals=nevals,
                        avg=avg_val,
                        std=std_val,
                        min=min_val,
                        max=max_val,
                        genotypic_diversity=genotypic_diversity,
                        phenotypic_diversity=phenotypic_diversity
                    )
                    session.add(genetic)
        session.commit()
        logger.info("Genetic results for %s saved to the database.", current_working_date)
    except Exception as e:
        session.rollback()
        logger.error("Error saving genetic results: %s", e)
    finally:
        session.close()


def save_prediction_results_to_db(current_working_date: str,
                                  received_fog_prediction_results: List[Dict[str, Any]]) -> None:
    """
    Save prediction evaluation results into the database.
    Each fog record contains a fog_id and a list of prediction records.
    Each prediction record is expected to have an edge_id and a list of prediction pairs.
    Each pair is stored as a row with pair_index, real_value, and predicted_value.
    """
    if not current_working_date:
        logger.error("No current working date provided; cannot save prediction results.")
        return

    session: Session = SessionLocal()
    try:
        evaluation = session.query(Evaluation).filter_by(evaluation_date=current_working_date).first()
        if not evaluation:
            evaluation = Evaluation(evaluation_date=current_working_date)
            session.add(evaluation)
            session.flush()

        for record in received_fog_prediction_results:
            fog_mac = record.get("fog_mac")
            # Assuming record["results"] is a list of prediction records.
            for pred in record.get("results", []):
                edge_mac = pred.get("edge_mac")
                pairs = pred.get("prediction_pairs", [])
                for idx, pair in enumerate(pairs):
                    if isinstance(pair, list) and len(pair) >= 2:
                        prediction = PredictionRecord(
                            evaluation_id=evaluation.id,
                            fog_mac=fog_mac,
                            edge_mac=edge_mac,
                            pair_index=idx,
                            real_value=pair[0],
                            predicted_value=pair[1]
                        )
                        session.add(prediction)
        session.commit()
        logger.info("Prediction results for %s saved to the database.", current_working_date)
    except Exception as e:
        session.rollback()
        logger.error("Error saving prediction results: %s", e)
    finally:
        session.close()


# ------------------ Load Functions ------------------


def load_performance_results_from_db() -> dict[InstrumentedAttribute, dict[str, list[dict[str, list[Any] | Any]]]]:
    """
    Retrieve performance results from the database.
    Returns a dict keyed by evaluation_date with structure similar to:
    {
      "2018-04-05": {
         "performance_results": [
             {
                "fog_id": "...",
                "results": [
                    {
                      "edge_id": "...",
                      "metrics": { "mse": ..., "mae": ..., ... },
                      "evaluation_date": "..."
                    }
                ]
             },
             ...
         ]
      },
      ...
    }
    """
    session: Session = SessionLocal()
    results_by_date = {}
    try:
        evaluations = session.query(Evaluation).order_by(Evaluation.evaluation_date).all()
        for eval_obj in evaluations:
            date = eval_obj.evaluation_date
            records_by_fog = {}
            for record in eval_obj.performance_records:
                fog_mac = record.fog_mac
                result_entry = {
                    "edge_mac": record.edge_mac,
                    "metrics": {
                        "mse": record.mse,
                        "mae": record.mae,
                        "r2": record.r2,
                        "logcosh": record.logcosh,
                        "huber": record.huber,
                        "msle": record.msle
                    },
                    "evaluation_date": date
                }
                if fog_mac not in records_by_fog:
                    records_by_fog[fog_mac] = {"fog_mac": fog_mac, "results": []}
                records_by_fog[fog_mac]["results"].append(result_entry)
            results_by_date[date] = {"performance_results": list(records_by_fog.values())}
    except Exception as e:
        logger.error("Error loading performance results: %s", e)
    finally:
        session.close()
    return results_by_date


def load_genetic_results_from_db() -> dict:
    """
    Retrieve genetic results from the database.
    Returns a dict keyed by evaluation_date in the following format:
    {
      "2018-04-05": {
         "genetic_results": [
             {
                "fog_id": "...",  // may be None if not provided; you can default to "all"
                "records": [
                    {
                      "gen": 0,
                      "nevals": 3,
                      "avg": 0.2687050916450229,
                      "std": 0.06256429460265911,
                      "min": 0.19229583571702377,
                      "max": 0.34554462151247795,
                      "genotypic_diversity": 55.50298111940518,
                      "phenotypic_diversity": 0.06256429460265911
                    },
                    { ... }
                ],
                "evaluation_date": "2018-04-05"
             },
             ...
         ]
      },
      ...
    }
    """
    session: Session = SessionLocal()
    results_by_date = {}
    try:
        evaluations = session.query(Evaluation).order_by(Evaluation.evaluation_date).all()
        for eval_obj in evaluations:
            date = eval_obj.evaluation_date
            records_by_fog = {}
            for record in eval_obj.genetic_records:
                fog_mac = record.fog_mac if record.fog_mac is not None else "all"
                record_entry = {
                    "gen": record.generation,
                    "nevals": record.nevals,
                    "avg": record.avg,
                    "std": record.std,
                    "min": record.min,
                    "max": record.max,
                    "genotypic_diversity": record.genotypic_diversity,
                    "phenotypic_diversity": record.phenotypic_diversity
                }
                if fog_mac not in records_by_fog:
                    records_by_fog[fog_mac] = {"fog_mac": fog_mac, "records": []}
                records_by_fog[fog_mac]["records"].append(record_entry)
            genetic_results = []
            for rec in records_by_fog.values():
                rec["evaluation_date"] = date
                genetic_results.append(rec)
            results_by_date[date] = {"genetic_results": genetic_results}
    except Exception as e:
        logger.error("Error loading genetic results: %s", e)
    finally:
        session.close()
    return results_by_date


def load_prediction_results_from_db() -> dict[InstrumentedAttribute, dict[str, list[dict[str, list[Any] | str | Any]]]]:
    """
    Retrieve prediction results from the database.
    Returns a dict keyed by evaluation_date in the following format:
    {
      "2018-04-05": {
         "prediction_results": [
             {
                "fog_id": "...",  // optional if available
                "results": [
                    {
                      "edge_id": "...",
                      "prediction_pairs": [
                          [real_value, predicted_value],
                          ...
                      ],
                      "evaluation_date": "..."
                    }
                ]
             },
             ...
         ]
      },
      ...
    }
    """
    session: Session = SessionLocal()
    results_by_date = {}
    try:
        evaluations = session.query(Evaluation).order_by(Evaluation.evaluation_date).all()
        for eval_obj in evaluations:
            date = eval_obj.evaluation_date
            records_by_fog = {}
            # Group prediction records by fog_id and edge_id.
            for record in eval_obj.prediction_records:
                fog_mac = record.fog_mac if record.fog_mac is not None else "all"
                key = (fog_mac, record.edge_mac)
                if key not in records_by_fog:
                    records_by_fog[key] = {
                        "fog_mac": fog_mac,
                        "edge_mac": record.edge_mac,
                        "prediction_pairs": []
                    }
                records_by_fog[key]["prediction_pairs"].append([record.real_value, record.predicted_value])
            # Build result entries.
            prediction_results = []
            for rec in records_by_fog.values():
                rec["evaluation_date"] = date
                prediction_results.append(rec)
            results_by_date[date] = {"prediction_results": prediction_results}
    except Exception as e:
        logger.error("Error loading prediction results: %s", e)
    finally:
        session.close()
    return results_by_date


def get_edge_node_by_device_mac(device_mac: str):
    """
    Query the database for a node record matching the provided device_mac
    and return a dict structure.
    """
    session: Session = SessionLocal()
    try:
        node_record = session.query(NodeRecord).filter(device_mac=device_mac).first()
        if node_record is None:
            raise ValueError(f"No FedNode found with device_mac: {device_mac}")
        return {
            "id": node_record.node_id,
            "label": node_record.node_label,
            "parent_id": node_record.parent_id,
            "parent_label": node_record.parent_label
        }
    except Exception as e:
        logger.error("Error retrieving node record for device_mac %s: %s", device_mac, e)
        raise e
    finally:
        session.close()


def save_system_metrics_to_db(current_working_data: str, received_fog_system_metrics):
    if not current_working_data:
        logger.error("No current working date provided; cannot save system metrics.")
        return

    session: Session = SessionLocal()

    try:
        evaluation = session.query(Evaluation).filter_by(evaluation_date=current_working_data).first()
        if not evaluation:
            evaluation = Evaluation(evaluation_date=current_working_data)
            session.add(evaluation)
            session.flush()

        for record in received_fog_system_metrics:
            fog_mac = record.get("fog_mac")
            system_metrics = record.get("system_metrics")

            for gen_rec in system_metrics:
                gen_number = gen_rec.get("gen")
                cpu_usage = gen_rec.get("cpu_usage")
                memory_usage = gen_rec.get("memory_usage")
                load_avg = gen_rec.get("load_avg")
                normalized_cpu_freq = gen_rec.get("normalized_cpu_freq")
                average_sensor_temp = gen_rec.get("average_sensor_temp")
                resource_load = gen_rec.get("resource_load")

                if gen_number is not None:
                    system_metric_record = SystemMetricRecord(
                        evaluation_id=evaluation.id,
                        fog_mac=fog_mac,
                        generation=gen_number,
                        cpu_usage=cpu_usage,
                        memory_usage=memory_usage,
                        load_avg=load_avg,
                        normalized_cpu_freq=normalized_cpu_freq,
                        average_sensor_temp=average_sensor_temp,
                        resource_load=resource_load
                    )
                    session.add(system_metric_record)
        session.commit()
        logger.info("Successfully saved system metrics.")
    except Exception as e:
        session.rollback()
        logger.error("Error while saving system metrics: %s", e)
    finally:
        session.close()


def load_system_metrics_from_db() -> dict:
    session: Session = SessionLocal()
    results_by_date = {}
    try:
        evaluations = session.query(Evaluation).order_by(Evaluation.evaluation_date).all()
        for eval_obj in evaluations:
            date = eval_obj.evaluation_date
            records_by_fog = {}
            for record in eval_obj.system_metrics:
                fog_mac = record.fog_mac if record.fog_mac is not None else "all"
                record_entry = {
                    "gen": record.generation,
                    "cpu_usage": record.cpu_usage,
                    "memory_usage": record.memory_usage,
                    "load_avg": record.load_avg,
                    "normalized_cpu_freq": record.normalized_cpu_freq,
                    "average_sensor_temp": record.average_sensor_temp,
                    "resource_load": record.resource_load
                }
                if fog_mac not in records_by_fog:
                    records_by_fog[fog_mac] = {"fog_mac": fog_mac, "system_metrics": []}
                records_by_fog[fog_mac]["system_metrics"].append(record_entry)

                system_metric_results = []
                for rec in records_by_fog.values():
                    rec["evaluation_date"] = date
                    system_metric_results.append(rec)
                results_by_date[date] = {"system_metrics": system_metric_results}
    except Exception as e:
        logger.error("Error loading system metrics: %s", e)
    finally:
        session.close()
    return results_by_date
