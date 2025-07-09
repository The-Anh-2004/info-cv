from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline


def show_text(arr):
    if len(arr) > 0:
        text = arr[0]
        for i in range(1, len(arr)):
            text = text+', '+arr[i]
    else:
        text = ''
    return text

def extract_address(text):
    address = []
    try:
        tokenizer = AutoTokenizer.from_pretrained("NlpHUST/ner-vietnamese-electra-base", model_max_length=50)
        model = AutoModelForTokenClassification.from_pretrained("NlpHUST/ner-vietnamese-electra-base")
        nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="max")
    except Exception as err:
        print('err1:', err)
    ner_results = nlp(text)
    print('ner_results:', ner_results)
    for ent in ner_results:
        if (ent['entity_group'] == 'LOCATION'):
            address.append(ent['word'])
    return show_text(address)