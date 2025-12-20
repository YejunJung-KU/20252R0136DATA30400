# 2025 Big Data Analysis (DATA304) Final Project: Hierarchical Multi-Label Text Classification

고려대학교 보건과학대학 바이오의공학부 2021250031 정예준


## 1. How to reproduce my main code

0. It is assumed that basic tools such as Git and Python are installed and run at a cmd in Windows.
   I made the best [`.csv`] result executing [`final_project_advanced.ipynb`] in Google Colab Tesla T4 GPU environment.
   It may create different output if you reproduce my code using other environment such as cmd.

1. Set the working directory to the desktop.

    ```bash
    cd C:\Users\user\Desktop
    ```

2. Clone my repository

    ```bash
    git clone https://github.com/YejunJung-KU/20252R0136DATA30400.git
    ```

3. Move the working directory to the cloned folder.

    ```bash
    cd C:\Users\user\Desktop\20252R0136DATA30400
    ```

4. Install the libraries required for execution.

    ```bash
    pip install -r requirements.txt
    ```

5. Run my Python code.

    ```bash
    python final_project_2021250031.py
    ```


## 2. Main code

0. I made the best [`.csv`] result executing [`final_project_advanced.ipynb`] in Google Colab Tesla T4 GPU environment.
   It may create different output if you reproduce my code using other environment such as cmd.

1. [`final_project_advanced.ipynb`](./final_project_advanced.ipynb) - Full pipeline with several advanced strategies.
- Silver labels generation
- GCN enhanced classifier
- Fine-tuning BERT & label/document enbeddings
- 250 LLM API calls
- Self-training: pseudo-label generation
- Consistency regularization

2. [`final_project_2021250031.py`](./final_project_2021250031.py) - [`.py`] version of main code for reproducing.


## 3. Intermediate codes (No need to reproduce. Just for reference.)

1. [`final_project_baseline.ipynb`](./final_project_baseline.ipynb) - Basic pipeline without any advanced strategies. Only silver labels generation + GCN enhanced classifier.
2. [`generate_embeddings.ipynb`](./generate_embeddings.ipynb) - Generate label/document enbeddings using BERT.
3. [`generate_embeddings_fine_tuning.ipynb`](./generate_embeddings_fine_tuning.ipynb) - Generate label/document embeddings using BERT (Fine-tuning).
4. [`llm_api_calls_250.ipynb`](./llm_api_calls_250.ipynb) - Generate high-confidence labels using 250 LLM API calls.


## 4. Intermediate files (I implemented all these files myself.)

1. [`label_bert_mean.pt`], [`train_bert_mean.pt`], [`test_bert_mean.pt`] - Mean-pooled BERT embeddings
2. [`label_bert_mean_dapt.pt`], [`train_bert_mean_dapt.pt`], [`test_bert_mean_dapt.pt`] - Mean-pooled BERT embeddings with MLM fine-tuning (Domain-Adaptive Pretraining).
3. [`bert_dapt_mlm`] - BERT model and tokenizer (etc.) fine-tuned on Amazon product data. 
   You can download [`model.safetensors`](https://drive.google.com/file/d/1-XMgAUpR7MFzjTzvTCP-oUiRCFG6_kzE/view?usp=sharing) file(417.8MB) in the following link
   and move it to [`bert_dapt_mlm`] folder if you want to use it. But you don't need to use these files to reproduce the main code.
4. [`llm_labels_250.jsonl`] - High-confidence labels using 250 LLM API calls
   
