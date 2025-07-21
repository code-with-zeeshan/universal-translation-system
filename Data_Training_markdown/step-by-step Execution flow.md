# Step-by-Step Execution Flow

This guide describes the step-by-step execution of the Universal Translation System data pipeline and training workflow.

## 1. Configure Languages and Pairs
- Edit `data/config.yaml` to add/remove languages, training pairs, and vocabulary groups

## 2. Run the Integrated Data Pipeline
- Execute `python data/practical_data_pipeline.py` to download, preprocess, sample, augment, and assemble all data
- All steps are orchestrated and logged

## 3. Create Vocabulary Packs
- Run `python vocabulary/create_vocabulary_packs_from_data.py` to generate or update vocabulary packs

## 4. Initialize Models
- Run `python training/bootstrap_from_pretrained.py` to initialize encoder/decoder from pretrained checkpoints

## 5. Train Models
- Run `python training/train_universal_system.py` (config auto-detection will select the best hardware config)

## 6. Monitor Progress
- Check logs in `logs/` and use the coordinator dashboard for real-time monitoring
- Prometheus metrics are available for all major steps

## 7. Convert and Optimize Models
- Use `python training/convert_models.py` and `python training/optimize_for_mobile.py` for deployment

---

For a visual workflow, see [Streamlined_Training_Workflow.mmd](Streamlined_Training_Workflow.mmd).