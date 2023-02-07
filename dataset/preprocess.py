import json
import os

for language in ['ruby','go','java','javascript','php','python']:
    print(language)
    train,valid,test,codebase=[],[],[], []
    for root, dirs, files in os.walk(language+'/final'):
        for file in files:
            temp=os.path.join(root,file)
            if '.jsonl' in temp:
                if 'train' in temp:
                    train.append(temp)
                elif 'valid' in temp:
                    valid.append(temp)
                    codebase.append(temp)
                elif 'test' in temp:
                    test.append(temp) 
                    codebase.append(temp)
                    
    train_data,valid_data,test_data,codebase_data={},{},{},{}
    for files,data in [[train,train_data],[valid,valid_data],[test,test_data],[codebase,codebase_data]]:
            for file in files:
                if '.gz' in file:
                    os.system("gzip -d {}".format(file))
                    file=file.replace('.gz','')
                with open(file) as f:
                    for line in f:
                        line=line.strip()
                        js=json.loads(line)
                        data[js['url']]=js
                        
    with open('{}/codebase.jsonl'.format(language),'w') as f3:
        for tag,data in [['train',train_data],['valid',valid_data],['test',test_data],['test',test_data],['codebase',codebase_data]]:
            with open('{}/{}.jsonl'.format(language,tag),'w') as f1, open("{}/{}.txt".format(language,tag)) as f2:
                for line in f2:
                    line=line.strip()
                    if line in data:
                        js=data[line]
                        if tag in ['valid','test']:
                            js['original_string']=''
                            js['code']=''
                            js['code_tokens']=[]
                        if tag=='codebase':
                            js['docstring']=''
                            js['docstring_tokens']=[]
                        f1.write(json.dumps(js)+'\n')
                    
