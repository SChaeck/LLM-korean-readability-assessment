from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import json
from tqdm import tqdm
from prompt import zero_shot_prompt, one_shot_prompt
import re

max_model_len, tp_size = 131072, 1
model_name = "THUDM/glm-4-9b-chat"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
llm = LLM(
    model=model_name,
    tensor_parallel_size=tp_size,
    max_model_len=max_model_len,
    trust_remote_code=True,
    enforce_eager=True,
)
stop_token_ids = [151329, 151336, 151338]
sampling_params = SamplingParams(temperature=0.3, max_tokens=8192, stop_token_ids=stop_token_ids)


with open('../processed_data/test_data.json', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)
    data_list = raw_data['contexts']

answer_list = []
for i, data in enumerate(data_list):
    print(f"Processing item {i+1}/{len(data_list)}")  # 진행 상태 출력
    try:
        messages = [
            {"role": "user", "content": zero_shot_prompt.format(data=data)},
        ]
        inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = llm.generate(prompts=inputs, sampling_params=sampling_params)
        answer = outputs[0].outputs[0].text
        
        if 'Q:' in answer or 'A:' in answer:
            split_answer = re.split(r'Q:|A:', answer)
            final_answer = split_answer[-1].strip()  # 가장 마지막 원소를 사용하고 공백 제거
        else:
            final_answer = answer.strip()
            
        answer_list.append(final_answer)
    except:
        print('0-shot error: ',i)
        continue
    
with open('../outputs/glm-4_0-shot.json', 'w', encoding='utf-8') as f:
    raw_data['results'] = answer_list
    json.dump(raw_data, f, ensure_ascii=False, indent=4)


answer_list = []
for j, data in enumerate(data_list):
    print(f"Processing item {j+i+1}/{len(data_list)*2}")  # 진행 상태 출력
    try:
        messages = [
            {"role": "user", "content": one_shot_prompt.format(data=data)},
        ]
        inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = llm.generate(prompts=inputs, sampling_params=sampling_params)
        answer = outputs[0].outputs[0].text
        
        if 'Q:' in answer or 'A:' in answer:
            split_answer = re.split(r'Q:|A:', answer)
            final_answer = split_answer[-1].strip()  # 가장 마지막 원소를 사용하고 공백 제거
        else:
            final_answer = answer.strip()
            
        answer_list.append(final_answer)
    except:
        print('1-shot error: ',j)
        continue
    
with open('../outputs/glm-4_1-shot.json', 'w', encoding='utf-8') as f:
    raw_data['results'] = answer_list
    json.dump(raw_data, f, ensure_ascii=False, indent=4)