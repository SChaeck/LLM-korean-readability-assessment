from langchain_anthropic import ChatAnthropic
import json
from tqdm import tqdm
from prompt import zero_shot_prompt, one_shot_prompt
import re
from dotenv import load_dotenv

load_dotenv()

llm = ChatAnthropic(
    model="claude-3-5-sonnet-20240620",
    temperature=0.3,
    max_tokens=4096,
    timeout=None
)

with open('../processed_data/test_data.json', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)
    data_list = raw_data['contexts']

answer_list = []
for i, data in enumerate(data_list):
        print(f"Processing item {i+1}/{len(data_list)}")  # 진행 상태 출력
    # try:
        messages = [
            ("human", zero_shot_prompt.format(data=data)),
        ]
        ai_msg = llm.invoke(messages)
        answer = ai_msg.content
        print(answer)
        if 'Q:' in answer or 'A:' in answer:
            split_answer = re.split(r'Q:|A:', answer)
            final_answer = split_answer[-1].strip()  # 가장 마지막 원소를 사용하고 공백 제거
        else:
            final_answer = answer.strip()
        answer_list.append(final_answer)
    # except:
    #     print('0-shot error: ',i)
    #     continue
    
with open('../outputs/cluade-3-5-sonnet_0-shot.json', 'w', encoding='utf-8') as f:
    raw_data['results'] = answer_list
    json.dump(raw_data, f, ensure_ascii=False, indent=4)

answer_list = []
for j, data in enumerate(data_list):
        print(f"Processing item {j+i+1}/{len(data_list)*2}")  # 진행 상태 출력
    # try:
        messages = [
            ("human", one_shot_prompt.format(data=data)),
        ]
        ai_msg = llm.invoke(messages)
        answer = ai_msg.content
        print(answer)
        if 'Q:' in answer or 'A:' in answer:
            split_answer = re.split(r'Q:|A:', answer)
            final_answer = split_answer[-1].strip()  # 가장 마지막 원소를 사용하고 공백 제거
        else:
            final_answer = answer.strip()

        answer_list.append(final_answer)
    # except:
        # print('1-shot error: ',j)
        # continue
    
with open('../outputs/cluade-3-5-sonnet_1-shot.json', 'w', encoding='utf-8') as f:
    raw_data['results'] = answer_list
    json.dump(raw_data, f, ensure_ascii=False, indent=4)