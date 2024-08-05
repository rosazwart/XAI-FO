# XAI-FO Pipeline

## Introduction

This repository provides a drug-repurposing pipeline that predicts new drug candidates that can potentially treat a symptom related to a given disease such as Duchenne muscular dystrophy (DMD), Huntington's disease (HD) and Osteogenesis imperfecta (OI). For these predictions, explanations in the form of subgraphs of the input knowledge graph are generated.

Previous research by Pablo Perdomo Quinteiro[^1] provided this drug-repurposing pipeline. This is a project that builds upon this pipeline focusing on improving the conceptual model that the input knowledge graph conforms to and finding out whether the predictions and explanations improve as well.

## Building Knowledge Graphs

Two kinds of knowledge graphs can be built given any disease being a knowledge graph that aligns with the data model of previous research (original KG) and a knowledge graph that has undergone structural changes in order to conform to a newly designed conceptual model using Foundational Ontologies: 

![image of final model](https://github.com/rosazwart/XAI-FO/blob/main/images/final_model.png)

### Collecting Data for Knowledge Graphs

The data that will initially populate the knowledge graphs come from the graph build of the Monarch Initiative from September 2021 and is reached by using the [tsv exports](https://data.monarchinitiative.org/202109/tsv/all_associations/index.html) of this graph build. The tsv files are also stored in folder [`localfetcher/deprecated_data`](https://github.com/rosazwart/XAI-FO/tree/main/localfetcher/deprecated_data) such that it can be reached by the fetching script in this repository.

To get all associations in the knowledge graph for a given disease, a list of seeds needs to be initialized that contain the most important entities that relate to the disease that serves as the foundation of the knowledge graph.

- [`localfetcher/main.py`](https://github.com/rosazwart/XAI-FO/tree/main/localfetcher/main.py) - Running this script will prompt the user to enter the disease that will serve as the foundation of the knowledge graph. A list of seeds already exists for DMD, HD and OI, containing the identifier of the disease itself and the identifier of the most important causal/correlated genes. The output is a csv file found in [`localfetcher/output`](https://github.com/rosazwart/XAI-FO/tree/main/localfetcher/output) with all association triples given the initial seeds. The name of the file indicates which group of seeds is used and the date of creation (for example `dmd_monarch_associations_2024-07-18.csv`).
  
### Acquiring the Original Knowledge Graph

The Monarch Initiative associations collected by the previously mentioned fetcher ([`localfetcher/main.py`](https://github.com/rosazwart/XAI-FO/tree/main/localfetcher/main.py)) do not yet contain drug information. To add drug information, data is used from two different datasets being DrugCentral and Therapeutic Target Database (TTD). The former contains drug-phenotype interactions while the latter includes drug-protein interactions. In order to merge the drug information with the associations that are already included in the knowledge graph, the following scripts in folder [`kg_builder/original`](https://github.com/rosazwart/XAI-FO/tree/main/kg_builder/original) need to be run:

- [`kg_builder/original/restructurer_main.py`](https://github.com/rosazwart/XAI-FO/blob/main/kg_builder/original/restructurer_main.py) - The entities found in the Monarch Initiative associations need to be organized into different conceptual classes such that the concepts, relations and triples are the same as found in the knowledge graph built in previous research[^1]. The resulting nodes after reorganizing are found in [`kg_builder/original/output`](https://github.com/rosazwart/XAI-FO/tree/main/kg_builder/original/output) such as `prev_dmd_monarch_nodes.csv`. All associations are found in folder [`output`](https://github.com/rosazwart/XAI-FO/tree/main/output) within the subfolder that corresponds to the relevant disease. For example DMD, for which the file [`output/dmd/prev_dmd_monarch_associations.csv`](https://github.com/rosazwart/XAI-FO/blob/main/output/dmd/prev_dmd_monarch_associations.csv) contains all associations that conform with the data model of the knowledge graph from previous research[^1].
- [`kg_builder/original/drug_info_merger_main.ipynb`](https://github.com/rosazwart/XAI-FO/blob/main/kg_builder/original/drug_info_merger_main.ipynb) - The drug information data from Drug Central and TTD are prepared to be compatible with the associations from Monarch Initiative. For example, acquiring the Human Phenotype identifiers of the disease entities found in the Drug Central dataset or acquiring the corresponding genes given the proteins that are targets of drugs given the TTD data. The relevant drug-disease pairs are found in [`kg_builder/original/output`](https://github.com/rosazwart/XAI-FO/tree/main/kg_builder/original/output) such as file `matched_drug_to_disease_dmd.csv`. In the same output folder, the relevant drug-gene pairs are stored in for example `matched_drug_targets_dmd.csv`.
- [`kg_builder/original/kg_drug_info_merger_main.ipynb`](https://github.com/rosazwart/XAI-FO/blob/main/kg_builder/original/kg_drug_info_merger_main.ipynb) - This script will transform the found drug-disease and drug-gene pairs to associations that conform to the data model of the knowledge graph using the correct relations between the entities. The associations are found in folder [`output`](https://github.com/rosazwart/XAI-FO/tree/main/output). For example the files with the drug information associations for DMD being `output/dmd/prev_dmd_drugcentral_associations.csv` and `output/dmd/prev_dmd_ttd_associations.csv`.
- [`kg_builder/original/kg_builder_main.py`](https://github.com/rosazwart/XAI-FO/blob/main/kg_builder/original/kg_builder_main.py) - Now, the knowledge graph is built that contains the associations from Monarch Initiative, DrugCentral and TTD. All the nodes and edges of this complete knowledge graph are stored for DMD in files `output/dmd/prev_dmd_kg_nodes.csv` and `output/dmd/prev_kg_dmd_edges.csv`.

### Acquiring the Restructured Knowledge Graph

The Monarch Initiative associations collected by the previously mentioned fetcher ([`localfetcher/main.py`](https://github.com/rosazwart/XAI-FO/tree/main/localfetcher/main.py)) do not yet contain drug information. To add drug information, data is used from two different datasets being DrugCentral and Therapeutic Target Database (TTD). The former contains drug-phenotype interactions while the latter includes drug-protein interactions. In order to merge the drug information with the associations that are already included in the knowledge graph, the following script needs to be performed:

- [`kg_builder/restructured/kg_builder_main.py`](https://github.com/rosazwart/XAI-FO/blob/main/kg_builder/restructured/kg_builder_main.py) - This script merges the drug information from Drug Central and TTD into the knowledge graph with the Monarch Initiative associations. The complete knowledge graph is stored into two files being `output/dmd/restr_dmd_kg_nodes.csv` and `output/dmd/restr_dmd_kg_edges.csv`.

## Generating Predictions

Predictions are generated by training a graph neural network (GNN) model on one of the two KG variations. This process is taken from previous research[^1]. However, the script for performing these steps in the pipeline are modified to allow for different input variations while maintaining the foundation of the already developed method. 

### Indexing

First, the nodes and edges of the knowledge graph need to be indexed such that these indices can be used consistently throughout all the steps of the prediction workflow for retrieving the correct nodes given an index.

- [`predictor/loader.ipynb`](https://github.com/rosazwart/XAI-FO/blob/main/predictor/loader.ipynb) - This Jupyter Notebook needs to be run to get the indexed nodes and edges stored into files found in the [output](https://github.com/rosazwart/XAI-FO/tree/main/output) folder. For example for DMD and the original knowledge graph, the files are `output/dmd/restr_dmd_indexed_nodes.csv` and `output/dmd/restr_dmd_indexed_edges.csv`.

### (Optional) Hyperparameter Optimization

### Node Embedding

#### Hyperparameters

| Parameters          | DMD            |               | HD            |               | OI            |               |
| ------------------- | -------------- | ------------- | --------------| ------------- | ------------- | ------------- |
|                     | Original KG    | Restructured KG| Original KG   | Restructured KG| Original KG   | Restructured KG|
| Number of walks     |                |               |               |               |               |               |
| Walk length         |                |               |               |               |               |               |
| Embedding dimension |                |               |               |               |               |               |
| p                   |                |               |               |               |               |               |
| q                   |                |               |               |               |               |               |
| epochs              |                |               |               |               |               |               |

### Training GNN Model

#### Hyperparameters

| Parameters                   | DMD            |               | HD            |               | OI            |               |
| ---------------------------- | -------------- | ------------- | --------------| ------------- | ------------- | ------------- |
|                              | Original KG    | Restructured KG| Original KG   | Restructured KG| Original KG   | Restructured KG|
| Hidden dimension             |                |               |               |               |               |               |
| Output dimension             |                |               |               |               |               |               |
| Layers                       |                |               |               |               |               |               |
| Aggregation function         |                |               |               |               |               |               |
| Dropout                      |                |               |               |               |               |               |
| Learning rate                |                |               |               |               |               |               |
| Epochs                       |                |               |               |               |               |               |
| Edge Negative Sampling Ratio |                |               |               |               |               |               |

## Generating Explanations

[^1]: [Master's thesis project](https://github.com/PPerdomoQ/rare-disease-explainer) of Pablo Perdomo Quinteiro
