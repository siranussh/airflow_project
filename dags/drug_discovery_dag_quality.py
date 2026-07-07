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

from lib.utils.teams import send_teams_alert

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


def _output_exists(s3, dataset_id: str) -> bool:
    key = f"{S3_OUTPUT_PREFIX}/{dataset_id}_results.csv"
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=key)
        return True
    except Exception:
        return False


def fetch_files(s3, dataset_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    scaffold_key = f"{S3_INPUT_PREFIX}/{dataset_id}_scaffolds.csv"
    rgroup_key   = f"{S3_INPUT_PREFIX}/{dataset_id}_r_groups.csv"

    scaffolds = _read_csv_from_s3(s3, scaffold_key)
    r_groups  = _read_csv_from_s3(s3, rgroup_key)

    for name, df in [("scaffolds", scaffolds), ("r_groups", r_groups)]:
        if "smiles" not in df.columns:
            raise ValueError(f"{name} CSV is missing the required 'smiles' column")

    log.info("Found %d scaffolds and %d r-groups", len(scaffolds), len(r_groups))
    return scaffolds, r_groups


def check_input_quality(scaffolds: pd.DataFrame, r_groups: pd.DataFrame) -> None:
    if len(scaffolds) == 0:
        raise ValueError("Quality check failed: scaffolds file is empty")
    if len(r_groups) == 0:
        raise ValueError("Quality check failed: r_groups file is empty")

    invalid_scaffolds = scaffolds[~scaffolds["smiles"].str.contains(r"\*")]
    multiple_attachment = scaffolds[scaffolds["smiles"].str.count(r"\*") > 1]

    if len(invalid_scaffolds) > 0:
        raise ValueError(
            f"Quality check failed: {len(invalid_scaffolds)} scaffolds missing attachment point '*'"
        )
    if len(multiple_attachment) > 0:
        raise ValueError(
            f"Quality check failed: {len(multiple_attachment)} scaffolds have more than one '*' attachment point"
        )

    invalid_r_groups = r_groups[~r_groups["smiles"].str.contains(r"\*")]
    multiple_attachment_rg = r_groups[r_groups["smiles"].str.count(r"\*") > 1]

    if len(invalid_r_groups) > 0:
        raise ValueError(
            f"Quality check failed: {len(invalid_r_groups)} r_groups missing attachment point '*'"
        )
    if len(multiple_attachment_rg) > 0:
        raise ValueError(
            f"Quality check failed: {len(multiple_attachment_rg)} r_groups have more than one '*' attachment point"
        )

    log.info("Input quality checks passed ✅")


def generate_molecules(scaffolds: pd.DataFrame, r_groups: pd.DataFrame) -> pd.DataFrame:
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
    return molecules_df


def check_molecules_quality(scaffolds: pd.DataFrame, r_groups: pd.DataFrame, molecules_df: pd.DataFrame) -> None:
    if len(molecules_df) == 0:
        raise ValueError("Quality check failed: no valid molecules were generated")

    duplicates = molecules_df["smiles"].duplicated().sum()
    if duplicates > 0:
        log.warning("Found %d duplicate SMILES", duplicates)

    total_combinations = len(scaffolds) * len(r_groups)
    valid_rate = len(molecules_df) / total_combinations
    if valid_rate < 0.5:
        raise ValueError(
            f"Quality check failed: only {valid_rate:.1%} of combinations produced valid molecules "
            f"(minimum 50% required)"
        )

    log.info("Molecule quality checks passed ✅ (%d molecules, %.1f%% valid)", len(molecules_df), valid_rate * 100)


def calculate_properties(molecules_df: pd.DataFrame) -> pd.DataFrame:
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
    return result


def check_properties_quality(props_df: pd.DataFrame) -> None:
    feature_cols = ["logP", "HBA", "HBD", "MW", "TPSA", "QED"]

    null_counts = props_df[feature_cols].isnull().sum()
    if null_counts.any():
        raise ValueError(f"Quality check failed: null values found in properties: {null_counts[null_counts > 0].to_dict()}")

    if not props_df["MW"].between(0, 1000).all():
        raise ValueError("Quality check failed: some molecules have MW outside 0-1000 Da range")

    if not props_df["logP"].between(-10, 10).all():
        raise ValueError("Quality check failed: some molecules have logP outside -10 to 10 range")

    if not props_df["QED"].between(0, 1).all():
        raise ValueError("Quality check failed: some molecules have QED outside 0-1 range")

    log.info("Property quality checks passed ✅")


def cluster_molecules(props_df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = ["logP", "HBA", "HBD", "MW", "TPSA", "QED"]
    X = props_df[feature_cols].values
    X_scaled = StandardScaler().fit_transform(X)

    k = min(N_CLUSTERS, len(props_df))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    props_df["cluster"] = kmeans.fit_predict(X_scaled)

    log.info("Assigned molecules into %d clusters", k)
    return props_df


def upload_results(s3, dataset_id: str, clustered_df: pd.DataFrame) -> None:
    out_key = f"{S3_OUTPUT_PREFIX}/{dataset_id}_results.csv"
    _upload_df_to_s3(s3, clustered_df, out_key)
    log.info("Uploaded %d molecules to s3://%s/%s", len(clustered_df), S3_BUCKET, out_key)


def get_new_datasets(**context) -> None:
    s3 = _get_s3()
    overwrite = context["params"]["overwrite"]
    last_run: datetime = context["data_interval_start"]

    response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_INPUT_PREFIX + "/")
    objects = response.get("Contents", [])

    new_scaffold_ids = set()
    for obj in objects:
        key = obj["Key"]
        last_modified = obj["LastModified"].replace(tzinfo=None)
        if key.endswith("_scaffolds.csv"):
            if overwrite or last_modified > last_run.replace(tzinfo=None):
                dataset_id = key.replace(f"{S3_INPUT_PREFIX}/", "").replace("_scaffolds.csv", "")
                new_scaffold_ids.add(dataset_id)

    datasets_to_process = []
    for dataset_id in new_scaffold_ids:
        if overwrite or not _output_exists(s3, dataset_id):
            datasets_to_process.append(dataset_id)
        else:
            log.info("Skipping %s — results already exist and overwrite=False", dataset_id)

    log.info("Datasets to process: %s", datasets_to_process)
    context["ti"].xcom_push("dataset_ids", datasets_to_process)


def process_all_datasets(**context) -> None:
    ti = context["ti"]
    dataset_ids = ti.xcom_pull(task_ids="get_new_datasets", key="dataset_ids")

    if not dataset_ids:
        log.info("No new datasets to process.")
        return

    s3 = _get_s3()

    for dataset_id in dataset_ids:
        log.info("Processing dataset: %s", dataset_id)

        scaffolds, r_groups = fetch_files(s3, dataset_id)
        check_input_quality(scaffolds, r_groups)

        molecules_df = generate_molecules(scaffolds, r_groups)
        check_molecules_quality(scaffolds, r_groups, molecules_df)

        props_df = calculate_properties(molecules_df)
        check_properties_quality(props_df)

        clustered_df = cluster_molecules(props_df)
        upload_results(s3, dataset_id, clustered_df)

        log.info("Finished dataset: %s", dataset_id)


default_args = {
    "owner":              "siranush_hakobyan",
    "retries":            1,
    "retry_delay":        timedelta(minutes=5),
    "email_on_failure":   False,
    "on_failure_callback": send_teams_alert,
}

with DAG(
    dag_id="drug_discovery_dag_quality",
    description="Cheminformatics pipeline: weekly schedule, quality checks, MS Teams alerts",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule="@weekly",
    catchup=False,
    tags=["cheminformatics", "drug-discovery", "step3"],
    params={
        "overwrite": Param(
            default=False,
            type="boolean",
            description="If True, reprocess all datasets even if results already exist.",
        ),
    },
) as dag:

    t_get_datasets = PythonOperator(
        task_id="get_new_datasets",
        python_callable=get_new_datasets,
    )

    t_process_all = PythonOperator(
        task_id="process_all_datasets",
        python_callable=process_all_datasets,
    )

    t_get_datasets >> t_process_all