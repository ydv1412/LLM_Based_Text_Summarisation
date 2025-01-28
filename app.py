from flask import Flask,render_template,request,jsonify
from peft import PeftModel, LoraConfig, TaskType
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, GenerationConfig
from transformers import AutoModelForCausalLM
import torch






app = Flask(__name__)
device = 0 if torch.cuda.is_available() else "cpu"
model_name = "google/flan-t5-base"
lora_config = LoraConfig(
    r=32, # Rank
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM # FLAN-T5
)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name, 
                                              torch_dtype=torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")

peft_model_3 = PeftModel.from_pretrained(model, 
                                       './ppo_trained_model', 
                                       lora_config=lora_config,
                                       torch_dtype=torch.bfloat16, 
                                       device_map="auto",                                       
                                       is_trainable=False)

generation_kwargs = {
    "min_length": 5,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True
}

reward_kwargs = {
    "top_k": None, # Return all scores.
    "function_to_apply": "none", # You want the raw logits without softmax.
    "batch_size": 16
}


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    input_text = data.get('text', '')

    if not input_text:
        return jsonify({'summary': 'No text provided'}), 400

    prompt = f"""
Summarize the following conversation.

{input_text}

Summary:
"""
    ids = tokenizer.encode(prompt)
        
    # This must be called "query", which is a requirement of our PPO library.
    query = tokenizer.decode(ids)
    summary = peft_model_3.generate(
        input_ids=torch.as_tensor(ids).unsqueeze(dim=0).to(device), 
        **generation_kwargs
    ).squeeze()

    actual_summary = tokenizer.decode(summary)
    actual_summary = actual_summary.replace("<pad>","").replace("</s>","")
    return jsonify({'summary': actual_summary})



if __name__== '__main__':
    app.run(debug = True)


