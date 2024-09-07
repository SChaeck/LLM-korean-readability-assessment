from langchain_openai import ChatOpenAI
import json
from tqdm import tqdm
import re
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o",
    top_p=0.1,
    max_tokens=4096,
    timeout=None
)

original_file_path = '../processed_data/test_data.json'
file_list = [
    # original_file_path,
    # 'cluade-3-5-sonnet_0-shot', 
    # 'cluade-3-5-sonnet_1-shot', 
    # 'gemini-pro-1.5_0-shot',
    # 'gemini-pro-1.5_1-shot', 
    # 'glm-4_0-shot',
    # 'glm-4_1-shot',
    # 'gpt-4o_0-shot',
    # 'gpt-4o_1-shot',
    # 'gpt-4o-mini_0-shot',
    # 'gpt-4o-mini_1-shot',
    'Ko-Gemma_0-shot',
    'Ko-Gemma_1-shot'
]

prompt = """###규칙
- 문단의 가독성(readability)는 가장 안좋을 때 0이고, 가장 좋을 때 10이야.
- 정수값으로만 측정해.
- 정답만 말해

###문단
{paragraph}

###출력 양식
{{
    "readability": INT
}}"""


error_list = []
score_list = []
for file_path in tqdm(file_list):
    total_score = 0
    each_error_count = 0
    
    if file_path == original_file_path:
        with open(original_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            text_list = data['contexts']
        
        for text in tqdm(text_list):
            messages = [
                ("system", "너는 한국어에 능숙한 언어학자야. 주어진 문단의 가독성이 얼마나 좋은지를 측정해줘."),
                ("human", prompt.format(paragraph=text)),
            ]
            ai_msg = llm.invoke(messages)
            answer = ai_msg.content
            try:
                answer_dict = json.loads(answer)
                total_score += answer_dict['readability']
            except:
                error_list.append(f'{file_path} / {text} / {answer}')
                each_error_count += 1
                print(f'{file_path} / {text} / {answer}')
                
        print('original_score:', total_score/(len(text_list)-each_error_count))
        score_list.append(total_score/(len(text_list)-each_error_count))
        
    else:
        with open(f'../outputs/{file_path}.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            text_list = data['results']
        
        for text in tqdm(text_list):
            messages = [
                ("system", "너는 한국어에 능숙한 언어학자야. 주어진 문단의 가독성이 얼마나 좋은지를 측정해줘."),
                ("human", prompt.format(paragraph=text)),
            ]
            ai_msg = llm.invoke(messages)
            answer = ai_msg.content
            try:
                answer_dict = json.loads(answer)
                total_score += answer_dict['readability']
            except:
                error_list.append(f'{file_path} / {text} / {answer}')
                each_error_count += 1
                print(f'{file_path} / {text} / {answer}')

        print(f'{file_path}:', total_score/len(text_list)-each_error_count)
        score_list.append(total_score/(len(text_list)-each_error_count))

print(error_list)
print(score_list)