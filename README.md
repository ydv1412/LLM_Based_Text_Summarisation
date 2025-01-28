# LLM_Based_Text_Summarisation
In this project I have tried to fine tune T5 large language model developed by Google using PEFT technique, followed by finetuning finetuning for less toxic sequence generation using Facebook's Roberta model as reward model. 


Pointers:-
1. I am using a transformer based model T5 trained by Google for multiple tasks.
2. My gola is to finetune this model using PEFT(Parameter Efficient Fine Tuning) technique for single task which is text summarization.
3. The performance matriz used to judge the finetuned model is rogue.
4. After that I further finetuned this model using PPO(Proximal Policy Optimisation) reinforcement learning technique to generate less toxic summaries.
5. I used Facebook's roberta model as the reward model for my Policy.  

# Software & Tools:-
1. [Github](https://github.com/)
2. [VS_Ide](https://code.visualstudio.com/)
3. [GitCLI](https://git-scm.com/book/en/v2/Getting-Started-The-Command-Line)


# Start a virtual Environment:
conda create -p venv python==3.7 -y