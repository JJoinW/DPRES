# DPRES: Disentangled Prompt Representation for Multi-Task Essay Scoring

Source code of our EMNLP 2026 Findings paper **DPRES: Disentangled Prompt Representation for Multi-Task Essay Scoring**

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

A pre-trained RoBERTa checkpoint is expected at `./models/robert/` (e.g. downloaded with the Hugging Face `roberta-base` model). 

### Train and evaluate a single prompt/fold

```bash
python run_main.py --prompt_id 1 --fold_id 0 --batch_size 16 --num_epochs 50 --gpu 0
```

## License

This project is licensed under the [MIT License](LICENSE).
