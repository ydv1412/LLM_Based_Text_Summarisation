# LLM_Based_Text_Summarisation
This project focuses on fine-tuning Google's Flan-T5 large language model using the PEFT (Parameter Efficient Fine-Tuning) technique for text summarization. Additionally, I have further fine-tuned the model using PPO (Proximal Policy Optimization) reinforcement learning to generate less toxic summaries, using Facebook's RoBERTa model as the reward model.



Pointers:-
1. I am using Flan-T5 model (a transformer based model developed by google) as my base model for this task.
2. Fine-tuning is performed using PEFT (Parameter Efficient Fine-Tuning) to optimize the model for a single task: text summarization.
3. The ROUGE (Recall-Oriented Understudy for Gisting Evaluation) metric is used to evaluate the summarization performance.
4. To ensure less toxic summaries, I further fine-tuned the model using PPO (Proximal Policy Optimization), a reinforcement learning technique.
5. The Facebook RoBERTa model is used as a reward model to penalize toxic summaries and promote safe and informative outputs.  

# Software & Tools:-
1. [Github](https://github.com/)
2. [VS_Ide](https://code.visualstudio.com/)
3. [GitCLI](https://git-scm.com/book/en/v2/Getting-Started-The-Command-Line)
4. Hugging Face Transformers
5. PEFT
6. RLHF


# Create a virtual Environment:
```
conda create -p venv python==3.7 -y 
conda activate venv
```

# Install Dependencies
```
pip install -r requirements.txt
```
The model was trained on jupyter notebook for long hours. It was saved and moved here for web app deployment.

# To run the project:
```
python app.py
```
#Evaluation Metrics
The performance of the fine-tuned model is evaluated using:
1. ROUGE Score (ROUGE-1, ROUGE-2, ROUGE-L)
2. Toxicity Score (from RoBERTa-based reward model)

# Project Structure
```
LLM_Based_Text_Summarisation/
│── ppo_trained_model/          # Saved fine-tuned model
│── static/                   # Styles
|── templates/                 # Html pages
│── app.py                     # Flask/FastAPI application for inference
│── Untitled.ipynb              # Jupyter notebook
│── requirements.txt            # Required dependencies
│── README.md                   # Documentation
```

# Future Enhancements
1. Training the model on high end systems for better results.
2. Experiment with different reward models for improving content quality.
3. Fine-tune the hyperparameters for better summarization accuracy.
4. Convert the model into a Hugging Face Inference API for easy access.


