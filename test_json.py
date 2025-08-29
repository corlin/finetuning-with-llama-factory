import json

with open('sample_qa_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    print('JSON loaded successfully')
    print(f'Items: {len(data)}')
    print(f'First item: {data[0]}')