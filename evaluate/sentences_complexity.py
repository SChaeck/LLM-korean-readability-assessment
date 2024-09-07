import re
from konlpy.tag import Kkma

# 꼬꼬마 분석기 초기화
kkma = Kkma()

# 문장 복잡도 분석 기준 정의
class SentenceComplexityAnalyzer:
    def __init__(self):
        # 기본 문장 구조 점수
        self.basic_structure = {
            '주술': 1,
            '주목술': 2,
            '주보술': 2,
            '주목보술': 3
        }
        # 첨가 조건에 따른 추가 점수
        self.addition_conditions = {
            '관형어': [0.99, 1.99, 0.33],  # 3번 이하, 4-6번, 7번 이상
            '부사어': [0.99, 1.99, 0.33],
            '독립어': [0.99, 1.99, 0.33]
        }
        # 절의 점수
        self.clauses = {
            '명사절': 3,
            '관형절': 3,
            '부사절': 3,
            '인용절': 3,
            '서술절': 3,
        }

    # HTML 태그 제거 함수
    def remove_html_tags(self, text):
        clean_text = re.sub(r'<[^>]+>', '', text)  # HTML 태그 제거
        return clean_text

    # 형태소 분석 결과를 기반으로 문장 복잡도를 계산
    def calculate_complexity(self, sentence):
        morphs = kkma.pos(sentence)
        
        complexity_score = 0
        structure_score = self.detect_basic_structure(morphs)
        complexity_score += structure_score

        condition_score = self.detect_addition_conditions(morphs)
        complexity_score += condition_score

        clause_score = self.detect_clauses(morphs)
        complexity_score += clause_score

        return complexity_score

    # 기본 문장 구조 분석
    def detect_basic_structure(self, morphs):
        has_subject = any(tag in ['NNG', 'NNP', 'NP'] for word, tag in morphs)
        has_predicate = any(tag.startswith('VV') or tag.startswith('VA') for word, tag in morphs)
        
        if has_subject and has_predicate:
            return self.basic_structure['주술']
        
        has_object = any(tag == 'JKO' for word, tag in morphs)
        has_complement = any(tag == 'JKC' for word, tag in morphs)

        if has_subject and has_object and has_predicate:
            return self.basic_structure['주목술']
        elif has_subject and has_complement and has_predicate:
            return self.basic_structure['주보술']
        elif has_subject and has_object and has_complement and has_predicate:
            return self.basic_structure['주목보술']
        
        return 0

    # 첨가 조건 분석
    def detect_addition_conditions(self, morphs):
        condition_count = {'관형어': 0, '부사어': 0, '독립어': 0}

        for word, tag in morphs:
            if tag == 'ETM':  # 관형어
                condition_count['관형어'] += 1
            elif tag == 'MAG' or tag == 'MAJ':  # 부사어
                condition_count['부사어'] += 1
            elif tag == 'IC':  # 독립어
                condition_count['독립어'] += 1

        total_score = 0
        for condition, count in condition_count.items():
            if count <= 3:
                total_score += self.addition_conditions[condition][0]
            elif count <= 6:
                total_score += self.addition_conditions[condition][1]
            else:
                total_score += count * self.addition_conditions[condition][2]
        
        return total_score

    # 절 분석
    def detect_clauses(self, morphs):
        clause_score = 0
        clause_count = 0

        for word, tag in morphs:
            if tag == 'ETN':  # 명사형 전성 어미
                clause_score += self.clauses['명사절']
                clause_count += 1
            elif tag == 'ETM':  # 관형형 전성 어미
                clause_score += self.clauses['관형절']
                clause_count += 1
            elif tag == 'ECD':  # 부사절
                clause_score += self.clauses['부사절']
                clause_count += 1
            elif tag == 'JKQ':  # 인용절
                clause_score += self.clauses['인용절']
                clause_count += 1

        if clause_count > 6:
            clause_score = 18

        return clause_score

    # 문단 복잡도 계산 (문장 복잡도의 평균 계산)
    def calculate_paragraph_complexity(self, paragraph):
        # HTML 태그 제거
        paragraph = self.remove_html_tags(paragraph)

        # 문단을 문장으로 분리
        sentences = kkma.sentences(paragraph)
        total_complexity = 0
        sentence_count = len(sentences)

        # 각 문장에 대해 복잡도 계산
        for sentence in sentences:
            total_complexity += self.calculate_complexity(sentence)

        # 문장 복잡도의 평균을 문단 복잡도로 반환
        if sentence_count > 0:
            return total_complexity / sentence_count
        else:
            return 0


if __name__ == "__main__":
    import json
    from tqdm import tqdm
    
    analyzer = SentenceComplexityAnalyzer()
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
                paragraph_complexity = analyzer.calculate_paragraph_complexity(text)    
                total_score += paragraph_complexity
            print('original_score:', total_score/len(text_list))
            
        else:
            with open(f'../outputs/{file_path}.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                text_list = data['results']
            
            for text in tqdm(text_list):
                paragraph_complexity = analyzer.calculate_paragraph_complexity(text)    
                total_score += paragraph_complexity
            print(f'{file_path}:', total_score/len(text_list))