# PAGED

Code and Dataset for our ACL 2024 Paper "[PAGED: A Benchmark for Procedural Graphs Extraction from Documents](https://arxiv.org/abs/2408.03630)"

![ProceduralGraph](/imgs/ProceduralGraph.png "ProceduralGraph")

<p align="center">An Example of Procedural Graph</p>

![Document](/imgs/Document.png "Document")

<p align="center">Corresponding Document</p>

### About PAGED
Automatic extraction of procedural graphs from documents creates a low-cost way for users to easily understand a complex procedure by skimming visual graphs. Despite the progress in recent studies, it remains unanswered: whether the existing studies have well solved this task (Q1) and whether the emerging large language models (LLMs) can bring new opportunities to this task (Q2). To this end, we propose a new benchmark PAGED, equipped with a large high-quality dataset and standard evaluations. It investigates five state-of-the-art baselines, revealing that they fail to extract optimal procedural graphs well because of their heavy reliance on hand-written rules and limited available data. We further involve three advanced LLMs in PAGED and enhance them with a novel self-refine strategy. The results point out the advantages of LLMs in identifying textual elements and their gaps in building logical structures. We hope PAGED can serve as a major landmark for automatic procedural graph extraction and the investigations in PAGED can offer insights into the research on logic reasoning among non-sequential elements.

### Guidance of PAGED

#### 1. extract procedural graphs from documents with baseline models
```python
python evaluation/predict_model_outputs.py
```

#### 2. evaluate the extracted procedural graphs
```python
python evaluation/evaluate_saved_outputs.py
```

#### 3. fine-tune the LLMs for better performance
```python
# prepare the data for fine-tuning
python evaluation/baselines/prepare_traning_data.py

# fine-tune FlanT5 model
python evaluation/baselines/FlanT5/fine_tune/trainer_flan_t5_xxl.py
# fill model paths information in evaluation/baselines/FlanT5/fine_tune/flan_t5_xxl_for_eva.py,
# then using it for evaluation

# fine-tune Llama2 model
python evaluation/baselines/Llama2/fine_tune/70b_training_ds_trainer_v2.py
# fill model paths information in evaluation/baselines/Llama2/fine_tune/Llama_2_70b.py,
# then using it for evaluation
```

Cite PAGED by the following BibTeX entry:
```latex
@inproceedings{du2024paged,
  title={PAGED: A Benchmark for Procedural Graphs Extraction from Documents},
  author={Du, Weihong and Liao, Wenrui and Liang, Hongru and Lei, Wenqiang},
  booktitle={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={10829--10846},
  year={2024}
}
```
