import json
from run_editor_sequential import revise_claim

with open('NQdataset.json') as f:
    data = json.load(f)

claim_obj = data[0]
# manually set claim to long_answer as read_claims() would do
claim_obj['claim'] = claim_obj['long_answer']

print('Question:', claim_obj.get('question_text'))
print()
print('INPUT:', claim_obj['claim'][:150])
print()

result = revise_claim(
    claim_obj=claim_obj,
    model='gpt-3.5-turbo',
    temperature_qgen=0.7,
    search_params={
        'max_search_results_per_query': 1,
        'max_sentences_per_passage': 2,
        'sliding_distance': 1,
        'max_passages_per_search_result_to_return': 1,
    },
    hallucinate_evidence=False,
)

print('ORIGINAL:', result['claim'][:300])
print()
print('REVISED:', result['revised_claim'][:300])
