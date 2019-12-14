import json
'''
with open("data/data_prepro.json") as json_file:
    json_data = json.load(json_file)
    print(json_data)

meta_data = json.load(open("data/data_prepro.json", 'r'))
meta_data['ix_to_word'] = {str(word):int(i) for i,word in meta_data['ix_to_word'].items()}
#meta_data['ix_to_ans'] = {str(word):int(i) for i,word in meta_data['ix_to_word'].items()}

#print(meta_data[])
num_classes = (meta_data['ix_to_ans'].keys())
num_words = (meta_data['ix_to_word'].keys())
print(num_classes)
print(num_words)
'''
meta_data = json.load(open("data/mscoco_val2014_annotations.json",'r'))
print(meta_data)
