# XAI-FO Pipeline

## Introduction

This repository provides a drug-repurposing pipeline that predicts new drug candidates that can potentially treat a symptom related to a given disease such as Duchenne muscular dystrophy (DMD), Huntington's disease (HD) and Osteogenesis imperfecta (OI). For these predictions, explanations in the form of subgraphs of the input knowledge graph are generated.

[Previous research by Pablo Perdomo Quinteiro](https://github.com/PPerdomoQ/rare-disease-explainer) provided this drug-repurposing pipeline. This is a project that builds upon this pipeline focusing on improving the conceptual model that the input knowledge graph conforms to and finding out whether the predictions and explanations improve as well.

## Building Knowledge Graphs

Two kinds of knowledge graphs can be built given any disease being a knowledge graph that aligns with the data model of previous research (original KG) and a knowledge graph that has undergone structural changes in order to conform to a newly designed conceptual model using Foundational Ontologies: 

![here](https://github.com/rosazwart/XAI-FO/blob/main/images/final_model.png)

### Collecting Data for Knowledge Graphs

The data that will initially populate the knowledge graphs come from the graph build of the Monarch Initiative from September 2021 and is reached by using the [tsv exports](https://data.monarchinitiative.org/202109/tsv/all_associations/index.html) of this graph build. The tsv files are also stored in folder [`localfetcher/deprecated_data`](https://github.com/rosazwart/XAI-FO/tree/main/localfetcher/deprecated_data) such that it can be reached by the fetching script in this repository.

To get all associations in the knowledge graph for a given disease, a list of seeds needs to be initialized that contain the most important entities that relate to the disease that serves as the foundation of the knowledge graph.

- [`localfetcher/main.py`](https://github.com/rosazwart/XAI-FO/blob/main/localfetcher/main.py) - Running this script will prompt the user to enter the disease that will serve as the foundation of the knowledge graph. A list of seeds already exists for DMD, HD and OI, containing the identifier of the disease itself and the identifier of the most important causal/correlated genes. The output is a csv file found in [`localfetcher/output`](https://github.com/rosazwart/XAI-FO/tree/main/localfetcher/output) with all association triples given the initial seeds.
