### 구분복잡성 지수

from konlpy.tag import Kkma
from konlpy.tag import Komoran

# KoNLPy의 형태소 분석기 로드
kkma = Kkma()
komoran = Komoran()

def syntactic_complexity_korean(text):
    sentences = kkma.sentences(text)  # 텍스트를 문장으로 분리합니다.
    num_sentences = len(sentences)
    
    num_clauses = 0
    for sentence in sentences:
        morphemes = komoran.pos(sentence)  # 문장을 형태소 분석합니다.
        num_clauses += sum(1 for word, tag in morphemes if tag in ['VV', 'VA'])  # 동사나 형용사를 종속 절로 간주합니다.
    
    return num_clauses / num_sentences if num_sentences > 0 else 0


import math
import re
from konlpy.tag import Okt


def calculate_fog_and_length(text):
    # 한국어 형태소 분석기
    okt = Okt()

    # 텍스트를 문장 단위로 분리
    sentences = re.split(r'[.!?]', text)

    # 문장당 평균 단어 수 계산 (ASL)
    total_words = 0
    total_sentences = len(sentences)
    for sentence in sentences:
        words = okt.morphs(sentence.strip())
        total_words += len(words)

    ASL = total_words / total_sentences if total_sentences > 0 else 0

    # 복잡한 단어 비율 계산 (3음절 이상의 단어)
    complex_words = 0
    for word in okt.morphs(text):
        if len(word) >= 3:
            complex_words += 1

    percentage_of_complex_words = (complex_words / total_words) * 100 if total_words > 0 else 0

    # FOG 계산
    FOG = (ASL + percentage_of_complex_words) * 0.4

    # LENGTH 계산
    Word_num = total_words
    LENGTH = math.log(Word_num) if Word_num > 0 else 0

    return ASL, percentage_of_complex_words, FOG, LENGTH


if __name__ == "__main__":

    import json
    from tqdm import tqdm
    
    original_file_path = '../processed_data/test_data.json'
    
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
    
    for file_path in tqdm(file_list):
        total_score = 0
        
        if file_path == original_file_path:
            with open(original_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                text_list = data['contexts']
            
            for text in tqdm(text_list):
                ASL, complex_words, FOG, LENGTH = calculate_fog_and_length(text)    
                total_score += FOG
            print('original_score:', total_score/len(text_list))
            
        else:
            with open(f'../outputs/{file_path}.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                text_list = data['results']
            
            for text in tqdm(text_list):
                ASL, complex_words, FOG, LENGTH = calculate_fog_and_length(text)    
                total_score += FOG
            print(f'{file_path}:', total_score/len(text_list))