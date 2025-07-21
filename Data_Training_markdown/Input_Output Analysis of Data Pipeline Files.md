# Input/Output Analysis of Data Pipeline Files

This document details the input and output of each major file in the Universal Translation System data pipeline.

## 1. data/config.yaml
- **Input:** N/A (manually edited)
- **Output:** Configuration for languages, training pairs, vocabulary groups, and thresholds

## 2. data/download_training_data.py
- **Input:** Language codes, config.yaml
- **Output:** Raw parallel data in `data/raw/`

## 3. data/practical_data_pipeline.py
- **Input:** config.yaml, all data modules
- **Output:** Orchestrates the entire pipeline, producing processed data in `data/processed/`, final data in `data/final/`, and logs

## 4. smart_sampler.py
- **Input:** Raw/processed data
- **Output:** Sampled, high-quality data in `data/sampled/`

## 5. synthetic_augmentation.py
- **Input:** Monolingual and parallel data
- **Output:** Augmented data (backtranslations, pivots) in `data/final/`

## 6. vocabulary/create_vocabulary_packs_from_data.py
- **Input:** Processed data, config.yaml
- **Output:** Vocabulary packs in `vocabulary/`

## 7. pipeline_connector.py
- **Input:** Processed data
- **Output:** Monolingual corpora, final training files

## 8. logs/
- **Input:** All pipeline steps
- **Output:** Logs for monitoring and debugging

## 9. Monitoring
- **Input:** Prometheus metrics from pipeline and coordinator
- **Output:** Real-time dashboards and alerts

---

For a visual workflow, see [Streamlined_Training_Workflow.mmd](Streamlined_Training_Workflow.mmd).