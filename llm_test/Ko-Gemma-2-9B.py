# Use a pipeline as a high-level helper
from prompt import zero_shot_prompt, one_shot_prompt
import json
from tqdm import tqdm
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

with open('../processed_data/test_data.json', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)
    data_list = raw_data['contexts']

tokenizer = AutoTokenizer.from_pretrained("rtzr/ko-gemma-2-9b-it")
model = AutoModelForCausalLM.from_pretrained(
    "rtzr/ko-gemma-2-9b-it",
    device_map="cuda:0",
    torch_dtype=torch.bfloat16,
)

model.eval()

def generate_answer(data):
    # 메시지 포맷 설정
    messages = [
        {"role": "user", "content": zero_shot_prompt.format(data=data)}
        # {"role": "user", "content": 'hi'}
    
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    # 종료 토큰 정의
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<end_of_turn>")
    ]
    
    outputs = model.generate(
        input_ids,
        max_new_tokens=8192,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.3,
    )
    # 생성된 텍스트를 디코딩
    generated_text = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)

    
    return generated_text



answer_list = []
for i, data in tqdm(enumerate(data_list)):
        print(f"Processing item {i+1}/{len(data_list)}")  # 진행 상태 출력

    # try:
        answer = generate_answer(data)

        print(answer)
        if '**' in answer:
            split_answer = answer.split('**')
            final_answer = split_answer[0].strip()  # 가장 첫번째 원소를 사용하고 공백 제거
        else:
            final_answer = answer.strip()
        
        answer_list.append(final_answer)
    # except:
    #     print('error: ',i)
    #     continue
    
with open('../outputs/KO-Gemma_0-shot.json', 'w', encoding='utf-8') as f:
    raw_data['results'] = answer_list
    json.dump(raw_data, f, ensure_ascii=False, indent=4)

answer_list = []
for j, data in tqdm(enumerate(data_list)):
        print(f"Processing item {j+i+1}/{len(data_list)*2}")  # 진행 상태 출력

    # try:
        answer = generate_answer(data)
        
        if '**' in answer:
            split_answer = answer.split('**')
            final_answer = split_answer[0].strip()  # 가장 첫번째 원소를 사용하고 공백 제거
        else:
            final_answer = answer.strip()
        
        print(final_answer)

        answer_list.append(final_answer)
    # except:
    #     print('error: ',i)
    #     continue
    
with open('../outputs/KO-Gemma_1-shot.json', 'w', encoding='utf-8') as f:
    raw_data['results'] = answer_list
    json.dump(raw_data, f, ensure_ascii=False, indent=4)
