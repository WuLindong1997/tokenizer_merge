import os
os.chdir('/mnt/vepfs/lingxin/Pretrain-data/wulindong/tokenizer_merge')


with open('data/document-tibetan.txt','r',encoding='utf-8')as file:
   data = file.read()

result = []

for line in data.split('\n'):
   text = line.split('\t',1)[-1]
   result.append(text)

with open('data/fileter.txt','w',encoding='utf-8')as file:
   file.write('\n'.join(result))
