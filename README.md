# XAI-FO Pipeline

## Introduction

This repository provides a drug-repurposing pipeline that predicts new drug candidates that can potentially treat a symptom related to a given disease such as Duchenne muscular dystrophy (DMD), Huntington's disease (HD) and Osteogenesis imperfecta (OI). For these predictions, explanations in the form of subgraphs of the input knowledge graph are generated.

Previous research by Pablo Perdomo Quinteiro[^1] provided this drug-repurposing pipeline. This is a project that builds upon this pipeline focusing on improving the conceptual model that the input knowledge graph conforms to and finding out whether the predictions and explanations improve as well.

In the implementation of the workflow, the keyword `prev` is used to indicate the knowledge graphs that comply with the original data model from previous research[^1]. The keyword `restr` implies that the knowledge graph is the restructured knowledge graph, complying to the newly designed conceptual model. The keywords `dmd`, `hd` and `oi` state that the knowledge graph is built from entities related to the disease DMD, HD or OI as seeds, respectively. 

## Building Knowledge Graphs

Two kinds of knowledge graphs can be built given any disease being a knowledge graph that aligns with the data model of previous research (original KG) and a knowledge graph that has undergone structural changes in order to conform to a newly designed conceptual model using Foundational Ontologies: 

![image of final model](https://github.com/rosazwart/XAI-FO/blob/main/images/final_model.png)

### Collecting Data for Knowledge Graphs

The data that will initially populate the knowledge graphs come from the graph build of the Monarch Initiative from September 2021 and is reached by using the [tsv exports](https://data.monarchinitiative.org/202109/tsv/all_associations/index.html) of this graph build. The tsv files are also stored in folder [`localfetcher/deprecated_data`](https://github.com/rosazwart/XAI-FO/tree/main/localfetcher/deprecated_data) such that it can be reached by the fetching script in this repository.

To get all associations in the knowledge graph for a given disease, a list of seeds needs to be initialized that contain the most important entities that relate to the disease that serves as the foundation of the knowledge graph.

- [`localfetcher/main.py`](https://github.com/rosazwart/XAI-FO/tree/main/localfetcher/main.py) - Running this script will prompt the user to enter the disease that will serve as the foundation of the knowledge graph. A list of seeds already exists for DMD, HD and OI, containing the identifier of the disease itself and the identifier of the most important causal/correlated genes. The output is a csv file found in [`localfetcher/output`](https://github.com/rosazwart/XAI-FO/tree/main/localfetcher/output) with all association triples given the initial seeds. The name of the file indicates which group of seeds is used and the date of creation (for example `dmd_monarch_associations_2024-07-18.csv`).
  
### Acquiring the Original Knowledge Graph

The Monarch Initiative associations collected by the previously mentioned fetcher ([`localfetcher/main.py`](https://github.com/rosazwart/XAI-FO/tree/main/localfetcher/main.py)) do not yet contain drug information. To add drug information, data is used from two different datasets being DrugCentral and Therapeutic Target Database (TTD). The former contains drug-phenotype interactions while the latter includes drug-protein interactions. In order to merge the drug information with the associations that are already included in the knowledge graph, the following scripts in folder [`kg_builder/original`](https://github.com/rosazwart/XAI-FO/tree/main/kg_builder/original) need to be run in the given order:

- [`kg_builder/original/1_restructurer_main.py`](https://github.com/rosazwart/XAI-FO/blob/main/kg_builder/original/1_restructurer_main.py) - The entities found in the Monarch Initiative associations need to be organized into different conceptual classes such that the concepts, relations and triples are the same as found in the knowledge graph built in previous research[^1]. The resulting nodes after reorganizing are found in [`kg_builder/original/output`](https://github.com/rosazwart/XAI-FO/tree/main/kg_builder/original/output) such as `prev_dmd_monarch_nodes.csv`. All associations are found in folder [`output`](https://github.com/rosazwart/XAI-FO/tree/main/output) within the subfolder that corresponds to the relevant disease. For example DMD, for which the file [`output/dmd/prev_dmd_monarch_associations.csv`](https://github.com/rosazwart/XAI-FO/blob/main/output/dmd/prev_dmd_monarch_associations.csv) contains all associations that conform with the data model of the knowledge graph from previous research[^1].
- [`kg_builder/original/2_drug_info_merger_main.ipynb`](https://github.com/rosazwart/XAI-FO/blob/main/kg_builder/original/2_drug_info_merger_main.ipynb) - The drug information data from Drug Central and TTD are prepared to be compatible with the associations from Monarch Initiative. For example, acquiring the Human Phenotype identifiers of the disease entities found in the Drug Central dataset or acquiring the corresponding genes given the proteins that are targets of drugs given the TTD data. The relevant drug-disease pairs are found in [`kg_builder/original/output`](https://github.com/rosazwart/XAI-FO/tree/main/kg_builder/original/output) such as file `matched_drug_to_disease_dmd.csv`. In the same output folder, the relevant drug-gene pairs are stored in for example `matched_drug_targets_dmd.csv`.
- [`kg_builder/original/3_kg_drug_info_merger_main.ipynb`](https://github.com/rosazwart/XAI-FO/blob/main/kg_builder/original/3_kg_drug_info_merger_main.ipynb) - This script will transform the found drug-disease and drug-gene pairs to associations that conform to the data model of the knowledge graph using the correct relations between the entities. The associations are found in folder [`output`](https://github.com/rosazwart/XAI-FO/tree/main/output). For example the files with the drug information associations for DMD being `output/dmd/prev_dmd_drugcentral_associations.csv` and `output/dmd/prev_dmd_ttd_associations.csv`.
- [`kg_builder/original/4_kg_builder_main.py`](https://github.com/rosazwart/XAI-FO/blob/main/kg_builder/original/4_kg_builder_main.py) - Now, the knowledge graph is built that contains the associations from Monarch Initiative, DrugCentral and TTD. All the nodes and edges of this complete knowledge graph are stored for DMD in files `output/dmd/prev_dmd_kg_nodes.csv` and `output/dmd/prev_kg_dmd_edges.csv`.

### Acquiring the Restructured Knowledge Graph

The Monarch Initiative associations collected by the previously mentioned fetcher ([`localfetcher/main.py`](https://github.com/rosazwart/XAI-FO/tree/main/localfetcher/main.py)) do not yet contain drug information. To add drug information, data is used from two different datasets being DrugCentral and Therapeutic Target Database (TTD). The former contains drug-phenotype interactions while the latter includes drug-protein interactions. In order to merge the drug information with the associations that are already included in the knowledge graph, the following script needs to be performed:

- [`kg_builder/restructured/kg_builder_main.py`](https://github.com/rosazwart/XAI-FO/blob/main/kg_builder/restructured/kg_builder_main.py) - This script merges the drug information from Drug Central and TTD into the knowledge graph with the Monarch Initiative associations. The complete knowledge graph is stored into two files being `output/dmd/restr_dmd_kg_nodes.csv` and `output/dmd/restr_dmd_kg_edges.csv`.

### Analyzing Knowledge Graphs

To analyse the built knowledge graphs, run [`analyser/kg_analyser.ipynb`](https://github.com/rosazwart/XAI-FO/blob/main/analyser/kg_analyser.ipynb). In [`analyser/data_params.py`](https://github.com/rosazwart/XAI-FO/blob/main/analyser/data_params.py) the parameters can be set to determine which knowledge graphs need to be included. The analysis outputs multiple files that can be found in the `output` folder and related subfolder such as [`output/dmd`](https://github.com/rosazwart/XAI-FO/tree/main/output/dmd). The files contain information about for example all existing triples in the knowledge graph and statistics for each node- and edge type. Also, the knowledge graphs are stored into GEXF (Graph Exchange XML Format) files to support loading in the network into various network visualization applications.

## Generating Predictions

Predictions are generated by training a graph neural network (GNN) model on one of the two KG variations. This process is taken from previous research[^1]. However, the script for performing these steps in the pipeline are modified to allow for different input variations while maintaining the foundation of the already developed method. 

The results from the node embedding and GNN training steps are found in the following folder given a knowledge graph complying to the original data model (`prev`) and on disease DMD (`dmd`):

- `output/dmd/prev_e2v/run_001`

Multiple runs of the embedding and prediction process on the same knowledge graphs are allowed (and even recommended for analyzing the workflow performance) where each run is stored into a separate folder. For example, the first run is found in this subfolder called `run_001`. This can be done using the script:

- [`predictor/run_predictor_notebooks.py`](https://github.com/rosazwart/XAI-FO/blob/main/predictor/run_predictor_notebooks.py) - This script will run the prediction pipeline (embedding and GNN model training). To adjust the parameters that select which knowledge graph is used as input, change the values in file [`predictor/data_params.py`](https://github.com/rosazwart/XAI-FO/blob/main/predictor/data_params.py).

### Indexing

First, the nodes and edges of the knowledge graph need to be indexed such that these indices can be used consistently throughout all the steps of the prediction workflow for retrieving the correct nodes given an index.

- [`predictor/1_loader.ipynb`](https://github.com/rosazwart/XAI-FO/blob/main/predictor/1_loader.ipynb) - This Jupyter Notebook needs to be run to get the indexed nodes and edges stored into files found in the [output](https://github.com/rosazwart/XAI-FO/tree/main/output) folder. For example for DMD and the original knowledge graph, the files are `output/dmd/restr_dmd_indexed_nodes.csv` and `output/dmd/restr_dmd_indexed_edges.csv`.

### (Optional) Hyperparameter Optimization

For the node embedding step and training the GNN model, a number of hyperparameters need to be set. In order to get the best results from the workflow, these parameter values need to be optimized. This can be done using the following script:

- [`hyperparameter_opt.py`](https://github.com/rosazwart/XAI-FO/blob/main/hyperparam_opt.py) - Hyperparameter optimization is run using Random Search for both the node embedding step as well as training the GNN model process. The resulting optimized hyperparameter values can be found in TXT files found in the main folder such as [`optimized_params_prev_dmd.txt`](https://github.com/rosazwart/XAI-FO/blob/main/optimized_params_prev_dmd.txt).

### Node Embedding

For the node embedding step, method Edge2vec[^2] has been implemented. The script can be found here:

- [`predictor/2_edge2vec_embedding.ipynb`](https://github.com/rosazwart/XAI-FO/blob/main/predictor/2_edge2vec_embedding.ipynb) - It outputs the final transition matrix and node embeddings into the corresponding `run_xxx` folder. 

#### Hyperparameters

| Parameters          | DMD            |                | HD            |                | OI            |                |
| ------------------- | -------------- | -------------- | --------------| -------------- | ------------- | -------------- |
|                     | Original KG    | Restructured KG| Original KG   | Restructured KG| Original KG   | Restructured KG|
| Number of walks     | 6              | 4              | 6             | 2              | 6             | 4              |
| Walk length         | 7              | 7              | 7             | 7              | 7             | 7              |
| Embedding dimension | 128            | 32             | 64            | 128            | 128           | 32             |
| p                   | 1.0            | 0.75           | 0.5           | 1.0            | 1.0           | 0.5            |
| q                   | 0.5            | 0.5            | 0.75          | 1.0            | 0.5           | 0.5            |
| epochs              | 10             | 10             | 10            | 10             | 10            | 10             |

### Training GNN Model

For the GNN training step, this script is used:

- [`predictor/3_predictor.ipynb`](https://github.com/rosazwart/XAI-FO/blob/main/predictor/3_predictor.ipynb) - After running this script, it outputs the resulting model weights in the corresponding `run_xxx` folder that are used in the next step of the workflow being the generation of explanations. The subfolder `run_xxx/pred` is created in which other results of this step are stored that are used for generating the explanation as well and also enable the analysis of the prediction performance. These results include for example the probability of an edge existing between a symptom and a drug calculated by the trained GNN model or the overall prediction performance scores during the training process and after in various metrics.

#### Hyperparameters

| Parameters                   | DMD            |                | HD            |                | OI            |                |
| ---------------------------- | -------------- | -------------- | --------------| -------------- | ------------- | -------------- |
|                              | Original KG    | Restructured KG| Original KG   | Restructured KG| Original KG   | Restructured KG|
| Hidden dimension             | 128            | 64             | 256           | 256            | 256           | 64             |
| Output dimension             | 256            | 64             | 64            | 64             | 64            | 128            |
| Layers                       | 4              | 2              | 4             | 6              | 2             | 2              |
| Aggregation function         | mean           | mean           | mean          | sum            | mean          | mean           |
| Dropout                      | 0.1            | 0.2            | 0.1           | 0.2            | 0.2           | 0.1            |
| Learning rate                | 0.012352       | 0.003191       | 0.015119      | 0.0364471      | 0.000606      | 0.026789       |
| Epochs                       | 200            | 150            | 150           | 150            | 100           | 150            |
| Edge Negative Sampling Ratio | 0.5            | 1.0            | 1.5           | 0.5            | 1.5           | 1.0            |

### Analyzing Prediction Performance and Results

To analyse the predictions and accuracy of the trained GNN model, run [`analyser/prediction_analyser.ipynb`](https://github.com/rosazwart/XAI-FO/blob/main/analyser/prediction_analyser.ipynb). In [`analyser/data_params.py`](https://github.com/rosazwart/XAI-FO/blob/main/analyser/data_params.py) the parameters can be set to determine which GNN models are included based on which knowledge graphs are used as training data. The analyser outputs files in the corresponding folder specifying which disease subject and which data model are used in the knowledge graph training data such as `output/dmd/prev_e2v`. For example, the predicted drug-symptom pair overlap between all independent runs of the GNN model trained on the same knowledge graph or training curves. The analysis considers all runs that have been performed, utilizing the prediction results from all run folders identified as for example `output/dmd/prev_e2v/run_xxx`. Some analysis results are stored into the parent folder such as `output/dmd` when it consists of the prediction results from both knowledge graphs (`prev` and `restr`) on the same disease such as the comparison of AUC ROC and F1 scores between the GNN models trained on the differently structured knowledge graphs given the same disease as subject.

## Generating Explanations

Explanations are generated using the script:

- [`predictor/4_explainer.ipynb`](https://github.com/rosazwart/XAI-FO/blob/main/predictor/4_explainer.ipynb) - As for the prediction process, use file [`predictor/data_params.py`](https://github.com/rosazwart/XAI-FO/blob/main/predictor/data_params.py) to adjust for which knowledge graph explanations need to be generated. In the Jupyter Notebook itself, it needs to be set which drug-symptom pairs are considered during the explanation generation. This is decided by indicating for how many runs the included drug-symptom pairs are found. For example, a value of `5` is the threshold of a drug-symptom pair to be included for finding explanations when it is found in at least 5 runs. In this case and for an original DMD KG, the explainer will output the explanation graphs in the folder `output/dmd/prev_e2v/expl_5`. This folder contains all found complete and incomplete explanations. Explanations are considered complete when there exists a direct or indirect path between the symptom and drug of the pair that is explained in the graph. The explanation graphs are stored in multiple formats such as an image or the raw data (`gpickle`, `pkl`).

### Hyperparameters 

The hyperparameters are not adjusted during the hyperparameter optimization step for each different knowledge graph and is thus fixed for each input.

| Parameters                   | Values         |
| ---------------------------- | -------------- |
| Epochs                       | 700            |
| Number of hops               | 1              |
| Maximum size of explanation  | 15             |
| Search iterations            | 10             |
| Learning rate                | 0.01           |

### Analyzing Generated Explanations

The generated explanations are analyzed using:

- [`analyser/explanation_analyser.ipynb`](https://github.com/rosazwart/XAI-FO/blob/main/analyser/explanation_analyser.ipynb) - This analysis script looks at the objective measurements to assess the explanations as well as the assessment of how many complete and incomplete the explainer yielded given the number of drug-symptom pairs that is included during the explanation generation.

This script outputs the following file for analyzing for example the explanations on the DMD KGs:

- `output/dmd/dmd_explanation_objective_measurements.csv` - Stores the objective measurements of the explanations found for each DMD KG

For each KG and the set of drug-symptom pairs that is used for generating the explanations, a file is stored that shows the yield of the explanainer. For example, for the original DMD KG looking at the explanations generated for the drug-symptom pairs that are found in at least 5 runs, the file is found here:

- `output/dmd/prev_e2v/expl_5/dmd_prev_expl_5_explanation_results.csv`

[^1]: [Master's thesis project](https://github.com/PPerdomoQ/rare-disease-explainer) of Pablo Perdomo Quinteiro
[^2]: Gao, Z., Fu, G., Ouyang, C. et al. edge2vec: Representation learning using edge semantics for biomedical knowledge discovery. BMC Bioinformatics 20, 306 (2019). [https://doi.org/10.1186/s12859-019-2914-2](https://doi.org/10.1186/s12859-019-2914-2)
