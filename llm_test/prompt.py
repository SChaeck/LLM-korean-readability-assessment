zero_shot_prompt = """Your task is to simplify the difficult Korean sentences I provide into easier Korean, without losing any of the original meaning. When simplifying, make sure that no part of the original sentence’s meaning is left out and that the key information is retained. 

<Conditions>:
1. Never omit or lose the original meaning of the sentence.
2. Ensure no important details(numerical, sementic, ..) are missing.
3. Use simple words and concise sentences for the simplification.
4. Only provide an answer.

<TODO>:
Q: {data}
A: 
"""


one_shot_prompt = """Your task is to simplify the difficult Korean sentences I provide into easier Korean, without losing any of the original meaning. When simplifying, make sure that no part of the original sentence’s meaning is left out and that the key information is retained. 

<Conditions>:
1. Never omit or lose the original meaning of the sentence.
2. Ensure no important details(numerical, sementic, ..) are missing.
3. Use simple words and concise sentences for the simplification.
4. Only provide an answer.

<Example>:
Q: 인플레이션 타게팅(IT)가 물가․경제 안정에 일정 부분 기여하였다는 점을 인정하더라도 이러한 안정기조가 지속되면서 발생한 여러 문제에 대해서는 대처가 미흡하였던 것으로 평가되고 있다. 중국의 세계경제 편입 등으로 물가가 안정세를 유지함에 따라 주요 중앙은행은 통화정책을 장기간에 걸쳐 완화적으로 운영하였는데 이에 따라 신용팽창, 과도한 위험선호, 무역 및 자본이동의 글로벌 불균형(global imbalance) 등에 대한 문제가 나타나게 되었다. 특히 Adrian and Shin(2008)은 금융기관의 신용확대는 중앙은행의 정책금리 수준에 크게 영향을 받는다는 것을 보이고 소폭의 정책금리 조정이 금융기관의 수익성에 영향을 미침으로써 신용증가세에 무시할 수 없는(non-negligible) 영향을 미칠 수 있다고 지적하였다. Altunbas et al.(2010)도 미국, 유럽지역 국가 등 16개국 643개 은행을 대상으로 분석한 결과 정책금리가 장기간 낮은 수준을 유지할 경우 은행의 위험선호(risk-taking)경향이 높아진다고 분석하였다. 
A: 인플레이션 타게팅이 물가와 경제 안정에 기여했지만, 안정이 지속되면서 생긴 여러 문제에 대처는 미흡했다고 평가받고 있다. 중국이 세계 경제에 편입되면서 물가가 안정되자 주요 중앙은행은 통화정책을 오랫동안 완화적으로 운영했고, 그 결과 신용팽창, 과도한 위험선호, 글로벌 불균형 등의 문제가 나타났다. 특히 Adrian과 Shin(2008)은 금융기관의 신용확대가 중앙은행의 정책금리에 크게 영향을 받으며, 작은 정책금리 조정이 금융기관 수익성과 신용증가에 영향을 미친다고 지적했다. Altunbas 등(2010)은 16개국 643개 은행을 분석한 결과, 정책금리가 오래 낮게 유지되면 은행의 위험선호가 높아진다고 분석했다. 

<TODO>:
Q: {data}
A: 
"""