from __future__ import annotations

import io
import logging
from datetime import datetime, timedelta
from itertools import product

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from airflow.sdk import DAG, Param
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

log = logging.getLogger(__name__)

S3_BUCKET        = "de-school-educational-data"
S3_INPUT_PREFIX  = "input"
S3_OUTPUT_PREFIX = "output/siranush_hakobyan"
AWS_CONN_ID      = "aws_default"
N_CLUSTERS       = 5


def _get_s3():
    hook = S3Hook(aws_conn_id=AWS_CONN_ID)
    return hook.get_conn()


def _read_csv_from_s3(s3, key: str) -> pd.DataFrame:
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))


def _upload_df_to_s3(s3, df: pd.DataFrame, key: str) -> None:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=buf.getvalue().encode())


def fetch_files(dataset_id: str, **context) -> None:
    s3 = _get_s3()

    scaffold_key = f"{S3_INPUT_PREFIX}/{dataset_id}_scaffolds.csv"
    rgroup_key   = f"{S3_INPUT_PREFIX}/{dataset_id}_r_groups.csv"

    scaffolds = _read_csv_from_s3(s3, scaffold_key)
    r_groups  = _read_csv_from_s3(s3, rgroup_key)

    for name, df in [("scaffolds", scaffolds), ("r_groups", r_groups)]:
        if "smiles" not in df.columns:
            raise ValueError(f"{name} CSV is missing the required 'smiles' column")

    log.info("Found %d scaffolds and %d r-groups", len(scaffolds), len(r_groups))

    ti = context["ti"]
    ti.xcom_push("scaffolds_json", scaffolds.to_json())
    ti.xcom_push("r_groups_json",  r_groups.to_json())


def generate_molecules(**context) -> None:
    ti = context["ti"]
    scaffolds = pd.read_json(ti.xcom_pull(task_ids="fetch_files", key="scaffolds_json"))
    r_groups  = pd.read_json(ti.xcom_pull(task_ids="fetch_files", key="r_groups_json"))

    records = []

    for (_, sc_row), (_, rg_row) in product(scaffolds.iterrows(), r_groups.iterrows()):
        scaffold_smi = sc_row["smiles"]
        rg_smi       = rg_row["smiles"]

        rg_fragment  = rg_smi.replace("*", "")
        combined_smi = scaffold_smi.replace("*", rg_fragment)

        mol = Chem.MolFromSmiles(combined_smi)
        if mol is None:
            log.warning("Skipping invalid combination: scaffold=%s + r_group=%s", scaffold_smi, rg_smi)
            continue

        records.append({
            "smiles":          Chem.MolToSmiles(mol),
            "scaffold_smiles": scaffold_smi,
            "rgroup_smiles":   rg_smi,
        })

    molecules_df = pd.DataFrame(records)
    log.info("Generated %d valid molecules", len(molecules_df))
    ti.xcom_push("molecules_json", molecules_df.to_json())


def calculate_properties(**context) -> None:
    ti = context["ti"]
    molecules_df = pd.read_json(
        ti.xcom_pull(task_ids="generate_molecules", key="molecules_json")
    )

    def calc_props(smi: str) -> dict:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return {}
        return {
            "logP": round(Descriptors.MolLogP(mol), 3),
            "HBA":  Descriptors.NumHAcceptors(mol),
            "HBD":  Descriptors.NumHDonors(mol),
            "MW":   round(Descriptors.MolWt(mol), 3),
            "TPSA": round(Descriptors.TPSA(mol), 3),
            "QED":  round(QED.qed(mol), 3),
        }

    prop_rows = [calc_props(smi) for smi in molecules_df["smiles"]]
    props_df  = pd.DataFrame(prop_rows)

    result = pd.concat([molecules_df.reset_index(drop=True), props_df], axis=1)
    result.dropna(inplace=True)

    log.info("Calculated properties for %d molecules", len(result))
    ti.xcom_push("props_json", result.to_json())


def cluster_molecules(**context) -> None:
    ti = context["ti"]
    props_df = pd.read_json(
        ti.xcom_pull(task_ids="calculate_properties", key="props_json")
    )

    feature_cols = ["logP", "HBA", "HBD", "MW", "TPSA", "QED"]
    X = props_df[feature_cols].values
    X_scaled = StandardScaler().fit_transform(X)

    k = min(N_CLUSTERS, len(props_df))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    props_df["cluster"] = kmeans.fit_predict(X_scaled)

    log.info("Assigned molecules into %d clusters", k)
    ti.xcom_push("clustered_json", props_df.to_json())


def upload_results(dataset_id: str, **context) -> None:
    ti = context["ti"]
    clustered_df = pd.read_json(
        ti.xcom_pull(task_ids="cluster_molecules", key="clustered_json")
    )

    s3 = _get_s3()
    out_key = f"{S3_OUTPUT_PREFIX}/{dataset_id}_results.csv"
    _upload_df_to_s3(s3, clustered_df, out_key)

    log.info("Uploaded %d molecules to s3://%s/%s", len(clustered_df), S3_BUCKET, out_key)


default_args = {
    "owner":            "siranush_hakobyan",
    "retries":          1,
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="drug_discovery_pipeline_v1",
    description="Cheminformatics pipeline: molecule generation → properties → clustering",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["cheminformatics", "drug-discovery", "step1"],
    params={
        "dataset_id": Param(
            default="",
            type="string",
            description="Expects <dataset_id>_scaffolds.csv and <dataset_id>_r_groups.csv in S3.",
        ),
    },
) as dag:

    t_fetch = PythonOperator(
        task_id="fetch_files",
        python_callable=fetch_files,
        op_kwargs={"dataset_id": "{{ params.dataset_id }}"},
    )

    t_generate = PythonOperator(
        task_id="generate_molecules",
        python_callable=generate_molecules,
    )

    t_props = PythonOperator(
        task_id="calculate_properties",
        python_callable=calculate_properties,
    )

    t_cluster = PythonOperator(
        task_id="cluster_molecules",
        python_callable=cluster_molecules,
    )

    t_upload = PythonOperator(
        task_id="upload_results",
        python_callable=upload_results,
        op_kwargs={"dataset_id": "{{ params.dataset_id }}"},
    )

    t_fetch >> t_generate >> t_props >> t_cluster >> t_upload
