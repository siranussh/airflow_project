# Drug Discovery Cheminformatics Pipeline

## Overview

This project is an Apache Airflow pipeline designed to support the early stages of drug discovery. In pharmaceutical research, scientists need to test large numbers of candidate molecules to find ones with good drug-like properties. Testing molecules in the lab is expensive and time-consuming, so computational screening is used to filter out bad candidates before any lab work begins.

This pipeline automates that screening process. Scientists upload molecular building blocks to S3 — scaffolds (the core molecular skeleton) and R-groups (small chemical fragments that attach to the scaffold). The pipeline combines them into thousands of candidate molecules, scores each one using standard drug-likeness metrics, and groups similar molecules into clusters so that scientists only need to test one representative from each group instead of testing everything.

## Pipeline Architecture

The pipeline follows a medallion-inspired architecture:

- **Bronze** — raw input files uploaded by scientists to S3. Minimal validation, data as received.
- **Silver** — generated molecules enriched with calculated molecular properties. Cleaned and analysis-ready.
- **Gold** — final clustered dataset ready for scientists to use. Each molecule has properties and a cluster label.

## How Molecule Generation Works

Each scaffold SMILES contains a `*` character which marks the attachment point — the position where an R-group fragment will be connected. The pipeline tries every combination of scaffold × R-group, replaces the `*` with the R-group fragment, and asks RDKit to validate the result. Invalid combinations are logged and skipped. Valid combinations are kept and passed to the next stage.

For example:
- Scaffold: `CCC*` (propyl chain with attachment point)
- R-group: `*N` (amine group)
- Result: `CCCN` (propylamine)

## Molecular Properties

For each valid molecule, RDKit calculates six standard drug-likeness properties based on the Lipinski Rule of Five — the industry standard for early drug candidate filtering:

| Property | Description |
|----------|-------------|
| logP | Lipophilicity — how the molecule distributes between fat and water |
| HBA | Hydrogen bond acceptors — affects cell membrane crossing |
| HBD | Hydrogen bond donors — affects cell membrane crossing |
| MW | Molecular weight in Daltons |
| TPSA | Topological polar surface area — affects gut absorption |
| QED | Quantitative estimate of drug-likeness (0 = bad, 1 = ideal) |

## Clustering

After property calculation, K-means clustering groups molecules into 5 clusters based on their property profile. Features are scaled using StandardScaler first so that MW (in hundreds) does not dominate logP (single digits) purely due to numeric scale. Scientists can then pick one representative molecule per cluster for lab testing instead of testing thousands of similar compounds.

## DAGs

### Step 1 — `drug_discovery_dag`

Manual trigger. The scientist passes a `dataset_id` parameter which tells the DAG which files to process. The DAG looks for `input/<dataset_id>_scaffolds.csv` and `input/<dataset_id>_r_groups.csv` in S3 and runs the full pipeline on them.

Task flow: `fetch_files → generate_molecules → calculate_properties → cluster_molecules → upload_results`

### Step 2 — `drug_discovery_dag_schedule`

Weekly schedule (`@weekly`). Instead of manual triggering, the DAG automatically scans the S3 input folder for files uploaded since the last run and processes all new datasets. An `overwrite` parameter (default `False`) controls whether already-processed datasets should be reprocessed — useful when input data is corrected and needs to be rerun.

### Step 3 — `drug_discovery_dag_quality`

Extends Step 2 with data quality checks at every stage of the pipeline and MS Teams notifications when something fails. In production you cannot assume input data is always clean — scientists may upload files with missing columns, empty files, or scaffolds without attachment points. Quality checks catch these problems early and alert the team immediately.

**Quality checks:**
- **Input quality** — files are not empty, all scaffold SMILES contain a `*` attachment point
- **Molecule quality** — at least one valid molecule was generated, valid rate above 50% of total combinations
- **Property quality** — all calculated values within chemically reasonable ranges (MW 0–1000 Da, logP −10 to 10, QED 0–1, no null values)

If any task fails, `on_failure_callback` automatically fires `send_teams_alert` which posts a formatted alert card to the MS Teams channel with the DAG name, task name, and error details.

## Tech Stack

- **Apache Airflow 3** — pipeline orchestration and scheduling
- **RDKit** — molecule generation, SMILES parsing, and property calculation
- **scikit-learn** — K-means clustering and feature scaling
- **AWS S3** — input/output file storage
- **Docker** — local development environment

## Input Format

```csv
smiles
CCC*
c1ccccc1*
CC(=O)*
```

Both scaffold and R-group files must have a `smiles` column. Scaffold SMILES must contain `*` to mark where the R-group attaches.

## Output

Results are uploaded to `output/siranush_hakobyan/<dataset_id>_results.csv` and contain:
- Original SMILES of the generated molecule
- Scaffold and R-group SMILES used to build it
- All six calculated properties
- Cluster label (0 to 4)

## Local Setup

```bash
git clone https://github.com/siranussh/airflow_project.git
cd airflow_project
cp .env.example .env
# Add your AWS credentials and MS Teams webhook URL to .env
docker-compose up -d
```

Open `http://localhost:8080` to access the Airflow UI.
