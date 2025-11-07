# 🧠 SUMMN Reproduction (Simplified Implementation)

Reproduction of the paper:
**"SUMMN: Long Document Summarization via Hierarchical Splitting and Merging"**
- Original Repo: https://github.com/psunlpgroup/Summ-N
- Paper: https://arxiv.org/abs/2110.10150v2

This project reimplements SUMMN using Hugging Face Transformers for easier, CPU-based summarization on Windows.

# 🔧 Environment Setup
```bash
conda create -n summn python=3.10
conda activate summn
```

# Install dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers sentencepiece tqdm numpy
```

# Folder Structure
- configure: the running configures for each dataset, such as number of stages, beam width etc.
- dataset_loader: the python scripts to convert original dataset to the uniform format.
- models: SummN model
- data_segment: including source and target segmentation code;
- gen_summary: inference on the source text and generate coarse summaries;
- train_summarizor.sh: we use fairseq-train command to train the model.
- scripts: all scripts to run experiments on different datasets.
- utils: utilities such as config parser & dataset reader etc.
- run.py: the entrance of the code.
- summarize.py: Enhanced BART-based summarizer that splits long text into chunks, summarizes each with dynamic parameters, and merges them into a coherent final summary.

# 🧩 Run summarizer
```bash
python summarize.py
```
# 📊 Paper vs Our Reproduction (ICSI dataset)

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|--------|----------|----------|----------|
| SUMMN (Paper) | 45.57 | 13.72 | 42.11 |
| BART (Ours) | ~37.2 | ~11.4 | ~35.0 |

✅ BART gives coherent summaries.

⚠️ Missing multi-stage hierarchical training.

# Citation
```bash
@inproceedings{zhang2021summn,
  title={Summ\^{} N: A Multi-Stage Summarization Framework for Long Input Dialogues and Documents},
  author={Zhang, Yusen and Ni, Ansong and Mao, Ziming and Wu, Chen Henry and Zhu, Chenguang and Deb, Budhaditya and Awadallah, Ahmed H and Radev, Dragomir and Zhang, Rui},
  booktitle={ACL 2022},
  year={2022}
}
```


