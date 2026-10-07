# LLM-Based Dialogue Summarization with PEFT & Reinforcement Learning

An end-to-end NLP project for generating concise summaries of multi-turn conversations using **Flan-T5**, **Parameter-Efficient Fine-Tuning (PEFT)**, and **Proximal Policy Optimization (PPO)**.

The project explores two objectives:

1. Improving dialogue summarization quality through instruction fine-tuning.
2. Reducing toxic or unsafe generated content using reinforcement learning with a RoBERTa-based reward model.

The final model was integrated into a **Flask web application**, Dockerized, and deployed on **Hugging Face Spaces**.

---

##  Project Overview

Large volumes of conversational text are generated through chatbots, customer support systems, and human-to-human conversations.

Reading long conversations to extract the important information can be time-consuming. This project investigates whether a fine-tuned language model can automatically generate concise summaries while also reducing undesirable or toxic outputs.

The complete pipeline consists of:

**Dialogue → Flan-T5 → PEFT Fine-Tuning → Evaluation → PPO Alignment → Web Application**

---

##  Key Results

| Metric | Result |
|---|---:|
| Base Model ROUGE-L | 0.24 |
| PEFT Fine-Tuned ROUGE-L | **0.40** |
| ROUGE-L Improvement | **+66.7%** |
| Toxicity Reduction after PPO | **52%** |
| Training Dataset | **14,460 dialogues** |

The results show that parameter-efficient fine-tuning substantially improved summarization performance while PPO-based optimization reduced toxicity in generated outputs.

> **Note:** Training was performed under limited computational resources, which restricted extensive hyperparameter tuning and experimentation.

---

##  Architecture

The project follows a two-stage fine-tuning strategy.

### Stage 1 — Dialogue Summarization

The pretrained **Flan-T5 Base (~250M parameters)** model is adapted for dialogue summarization using Parameter-Efficient Fine-Tuning (PEFT).

Instead of updating all parameters of the language model, PEFT allows the model to be adapted with significantly lower computational requirements.

```text
Conversation
     │
     ▼
Prompt Template
     │
     ▼
Flan-T5
     │
     ▼
PEFT Fine-Tuning
     │
     ▼
Generated Summary
     │
     ▼
ROUGE Evaluation
```

### Stage 2 — Reinforcement Learning for Safer Generation

After supervised fine-tuning, the model is further optimized using **Proximal Policy Optimization (PPO)**.

A **RoBERTa hate-speech classifier** is used as the reward model. The reward signal encourages the policy model to generate outputs with lower toxicity.

```text
Dialogue
   │
   ▼
PEFT Fine-Tuned Flan-T5
   │
   ▼
Generated Response
   │
   ├──────────────► RoBERTa Reward Model
   │                       │
   │                       ▼
   │                 Reward Signal
   │                       │
   └───────────────────────┘
                           │
                           ▼
                     PPO Update
```

During PPO training, the system:

1. Generates responses using the PEFT policy model.
2. Evaluates the generated responses using the RoBERTa reward model.
3. Calculates a reward based on the toxicity classification.
4. Updates the policy using PPO.

---

## 📚 Dataset

The model was fine-tuned using the **DialogSum** dataset from Hugging Face:

`knkarthick/dialogsum`

The dataset contains **14,460 dialogues with corresponding human-written summaries**.

Some observations from the dataset:

- Average dialogue length: ~104 tokens
- Average summary length: ~19 tokens
- Dialogues contain conversational structures such as `#Person1` and `#Person2`
- Stopwords were retained because they carry important grammatical and contextual information for dialogue summarization

---

## 📊 Evaluation

### Summarization Quality

Summarization performance was evaluated using:

- ROUGE-1
- ROUGE-2
- ROUGE-L

The most notable improvement was observed in **ROUGE-L**:

```text
Original Flan-T5       0.24
PEFT Fine-Tuned Model  0.40
```

This represents approximately a **66.7% relative improvement in ROUGE-L**.

### Toxicity Evaluation

After the summarization model was fine-tuned, PPO was used to optimize the model against toxicity signals produced by the RoBERTa reward model.

The reinforcement-learning stage resulted in a:

**52% reduction in toxicity in generated outputs.**

---

## 🛠️ Tech Stack

### Machine Learning

- Python
- PyTorch
- Hugging Face Transformers
- Flan-T5
- PEFT
- PPO
- RoBERTa
- Hugging Face Datasets

### Evaluation

- ROUGE
- RoBERTa-based toxicity / hate-speech scoring

### Application & Deployment

- Flask
- Docker
- Hugging Face Spaces
- Git / GitHub

---

## 🌐 Web Application

A lightweight Flask application was developed to provide an interface for model inference.

Users can provide conversational text and generate a summary using the fine-tuned model.

### Live Demo

👉 **[Try the application on Hugging Face Spaces](https://ydvshri1412-textsummarisation.hf.space/)**

> If the hosted application is unavailable or sleeping, see the project presentation and demo material below for the complete workflow and results.

---

##  Demo

A short demonstration of the complete pipeline can be added here:

**Input Dialogue → Generated Summary → Model Output**

>  **Demo video:** Coming soon

---

##  Project Presentation

A detailed presentation covering the methodology, dataset analysis, PEFT fine-tuning, PPO training, evaluation, web application and results is available here:

 **[View Project Presentation](./presentation/LLM_Based_Text_Summarisation.pdf)**

---

##  Project Structure

```text
LLM_Based_Text_Summarisation/
│
├── ppo_trained_model/          # PPO fine-tuned model
├── static/                     # Web application styling
├── templates/                  # HTML templates
├── presentation/
│   └── LLM_Based_Text_Summarisation.pdf
├── app.py                      # Flask inference application
├── Untitled.ipynb              # Model training and experimentation
├── requirements.txt
└── README.md
```

---

##  Running Locally

### Clone the repository

```bash
git clone https://github.com/ydv1412/LLM_Based_Text_Summarisation.git
cd LLM_Based_Text_Summarisation
```

### Create an environment

```bash
conda create -n text-summarization python=3.10 -y
conda activate text-summarization
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
python app.py
```

Then open the local Flask URL shown in the terminal.

---

##  Limitations

The project was developed with limited computational resources.

As a result:

- extensive hyperparameter tuning was not possible;
- PPO training was computationally expensive;
- some training runs terminated before completion;
- different reward models and alignment strategies were not extensively compared.

These limitations leave room for further experimentation with more efficient training and evaluation strategies.

---

## Future Work

Potential extensions include:

- Experimenting with alternative reward models
- More systematic hyperparameter optimization
- Evaluating summarization quality using semantic and LLM-based metrics in addition to ROUGE
- Comparing PPO with newer preference-optimization approaches
- Evaluating safety improvements across multiple toxicity benchmarks
- Deploying the model through a lightweight inference API

---

##  Authors

**Shri Prakash Yadav**  
M.Sc. Data Science — University of Naples Federico II

---

## Repository

If you found this project interesting, feel free to explore the implementation, experiments and presentation available in this repository.
