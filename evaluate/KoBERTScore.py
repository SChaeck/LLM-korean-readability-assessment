from KoBERTScore import BERTScore
import json
from tqdm import tqdm

batch_size = 300
model_name = "beomi/kcbert-base"
bertscore = BERTScore(model_name, best_layer=4)

original_file_path = '../../processed_data/test_data.json'
file_list = [
    original_file_path,
    'cluade-3-5-sonnet_0-shot', 
    'cluade-3-5-sonnet_1-shot', 
    'gemini-pro-1.5_0-shot',
    'gemini-pro-1.5_1-shot', 
    'glm-4_0-shot',
    'glm-4_1-shot',
    'gpt-4o_0-shot',
    'gpt-4o_1-shot',
    'gpt-4o-mini_0-shot',
    'gpt-4o-mini_1-shot',
    'Ko-Gemma_0-shot',
    'Ko-Gemma_1-shot'
]
import nltk
from tqdm import tqdm
import json
from KoBERTScore import BERTScore
from transformers import AutoTokenizer

nltk.download('punkt')
from nltk.tokenize import sent_tokenize

model_name = "beomi/kcbert-base"
bertscore = BERTScore(model_name, best_layer=4)

# BERT 모델의 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(model_name)

# references 파일 열기
with open(original_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
    references = data['contexts']

# 문단을 문장 단위로 분할
references_sentences = [sent_tokenize(paragraph) for paragraph in references]

result_list = []
for file_path in tqdm(file_list):
    if file_path == original_file_path:
        with open(original_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            candidates = data['contexts']
    else:
        with open(f'../../outputs/{file_path}.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            candidates = data['results']

    # candidates도 문장 단위로 분할
    candidates_sentences = [sent_tokenize(paragraph) for paragraph in candidates]

    total_score = 0
    total_count = 0
    
    for ref_sents, cand_sents in zip(references_sentences, candidates_sentences):
        # 긴 문장을 스킵하도록 처리
        valid_ref_sents = []
        valid_cand_sents = []

        for ref_sent, cand_sent in zip(ref_sents, cand_sents):
            # 토큰화 후 토큰 수 확인
            ref_tokens = tokenizer.tokenize(ref_sent)
            cand_tokens = tokenizer.tokenize(cand_sent)

            # 각 문장이 300 토큰 이하인 경우만 유효한 문장으로 처리
            if len(ref_tokens) <= 280 and len(cand_tokens) <= 280:
                valid_ref_sents.append(ref_sent)
                valid_cand_sents.append(cand_sent)

        if valid_ref_sents and valid_cand_sents:
            # 유효한 문장 쌍에 대해서만 BERTScore 계산
            total_score += sum(list(bertscore(valid_ref_sents, valid_cand_sents, batch_size=1)))
            total_count += len(valid_cand_sents)

    if total_count > 0:
        result_list.append(total_score / total_count)
    else:
        result_list.append('No valid sentences (all skipped)')
        
for i, file_path in enumerate(file_list):
    print(f'{file_path}:', result_list[i])