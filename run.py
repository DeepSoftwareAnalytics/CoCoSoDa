# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from unittest import removeResult
import torch.nn.functional as F
import argparse
import logging
import os
import pickle
import random
import torch
import json
from random import choice
import numpy as np
from itertools import cycle
from model import Model,Multi_Loss_CoCoSoDa
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                  RobertaConfig, RobertaModel, RobertaTokenizer)

logger = logging.getLogger(__name__)
from tqdm import tqdm
import multiprocessing
cpu_cont = 16

from parser import DFG_python,DFG_java,DFG_ruby,DFG_go,DFG_php,DFG_javascript
from parser import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index)
from tree_sitter import Language, Parser
import sys
sys.path.append("dataset")
from utils import save_json_data, save_pickle_data
dfg_function={
    'python':DFG_python,
    'java':DFG_java,
    'ruby':DFG_ruby,
    'go':DFG_go,
    'php':DFG_php,
    'javascript':DFG_javascript
}

parsers={}        
for lang in dfg_function:
    LANGUAGE = Language('parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE) 
    parser = [parser,dfg_function[lang]]    
    parsers[lang]= parser
    

ruby_special_token = ['keyword', 'identifier', 'separators', 'simple_symbol', 'constant', 'instance_variable',
 'operator', 'string_content', 'integer', 'escape_sequence', 'comment', 'hash_key_symbol',
  'global_variable', 'heredoc_beginning', 'heredoc_content', 'heredoc_end', 'class_variable',]

java_special_token = ['keyword', 'identifier', 'type_identifier',  'separators', 'operator', 'decimal_integer_literal',
 'void_type', 'string_literal', 'decimal_floating_point_literal', 
 'boolean_type', 'null_literal', 'comment', 'hex_integer_literal', 'character_literal']

go_special_token = ['keyword', 'identifier', 'separators', 'type_identifier', 'int_literal', 'operator', 
'field_identifier', 'package_identifier', 'comment',  'escape_sequence', 'raw_string_literal',
'rune_literal', 'label_name', 'float_literal']

javascript_special_token =['keyword', 'separators', 'identifier', 'property_identifier', 'operator', 
'number', 'string_fragment', 'comment', 'regex_pattern', 'shorthand_property_identifier_pattern', 
'shorthand_property_identifier', 'regex_flags', 'escape_sequence', 'statement_identifier']

php_special_token =['text', 'php_tag', 'name', 'operator', 'keyword', 'string', 'integer', 'separators', 'comment', 
'escape_sequence', 'ERROR',  'boolean', 'namespace', 'class', 'extends']

python_special_token =['keyword', 'identifier', 'separators', 'operator', '"', 'integer', 
'comment', 'none', 'escape_sequence']


special_token={
    'python':python_special_token,
    'java':java_special_token,
    'ruby':ruby_special_token,
    'go':go_special_token,
    'php':php_special_token,
    'javascript':javascript_special_token
}

all_special_token = []
for key, value in special_token.items():
    all_special_token = list(set(all_special_token ).union(set(value)))

def lalign(x, y, alpha=2):
    x = torch.tensor(x)
    y= torch.tensor(y)
    return (x - y).norm(dim=1).pow(alpha).mean()
    # code2nl_pos = torch.einsum('nc,nc->n', [x, y]).unsqueeze(-1)

    # return code2nl_pos.mean()

def lunif(x, t=2):
    x = torch.tensor(x)
    sq_pdist = torch.pdist(x, p=2).pow(2)
    return sq_pdist.mul(-t).exp().mean().log()



def cal_r1_r5_r10(ranks):
    r1,r5,r10= 0,0,0
    data_len= len(ranks)
    for item in ranks:
        if item >=1:
            r1 +=1
            r5 += 1 
            r10 += 1
        elif item >=0.2:
            r5+= 1
            r10+=1
        elif item >=0.1:
            r10 +=1
    result = {"R@1":round(r1/data_len,3), "R@5": round(r5/data_len,3),  "R@10": round(r10/data_len,3)}
    return result

#remove comments, tokenize code and extract dataflow                                        
def extract_dataflow(code, parser,lang):
    #remove comments
    try:
        code=remove_comments_and_docstrings(code,lang)
    except:
        pass    
    #obtain dataflow
    if lang=="php":
        code="<?php"+code+"?>"    
    try:
        tree = parser[0].parse(bytes(code,'utf8'))    
        root_node = tree.root_node  
        tokens_index=tree_to_token_index(root_node)     
        code=code.split('\n')
        code_tokens=[index_to_code_token(x,code) for x in tokens_index]  
        index_to_code={}
        for idx,(index,code) in enumerate(zip(tokens_index,code_tokens)):
            index_to_code[index]=(idx,code)  
        try:
            DFG,_=parser[1](root_node,index_to_code,{}) 
        except:
            DFG=[]
        DFG=sorted(DFG,key=lambda x:x[1])
        indexs=set()
        for d in DFG:
            if len(d[-1])!=0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG=[]
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg=new_DFG
    except:
        dfg=[]
    return code_tokens,dfg

#remove comments, tokenize code and extract dataflow                                        
def tokenizer_source_code(code, parser,lang):
    #remove comments
    try:
        code=remove_comments_and_docstrings(code,lang)
    except:
        pass    
    #obtain dataflow
    if lang=="php":
        code="<?php"+code+"?>"    
    try:
        tree = parser[0].parse(bytes(code,'utf8'))    
        root_node = tree.root_node  
        tokens_index=tree_to_token_index(root_node)     
        code=code.split('\n')
        code_tokens=[index_to_code_token(x,code) for x in tokens_index]  
    except:
        dfg=[]
    return code_tokens

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 code_tokens,
                 code_ids,
                #  position_idx,
                #  dfg_to_code,
                #  dfg_to_dfg,                 
                 nl_tokens,
                 nl_ids,
                 url,

    ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        # self.position_idx=position_idx
        # self.dfg_to_code=dfg_to_code
        # self.dfg_to_dfg=dfg_to_dfg        
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.url=url


class TypeAugInputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 code_tokens,
                 code_ids,
                #  position_idx,
                 code_type,
                 code_type_ids,                 
                 nl_tokens,
                 nl_ids,
                 url,

    ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        # self.position_idx=position_idx
        self.code_type=code_type
        self.code_type_ids=code_type_ids        
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.url=url

def convert_examples_to_features(js):
    js,tokenizer,args=js
    #code
    if args.lang == "java_mini":
        parser=parsers["java"]
    else:
        parser=parsers[js["language"]]
    # code
    code_tokens=tokenizer_source_code(js['original_string'],parser,args.lang)
    code_tokens=" ".join(code_tokens[:args.code_length-2])
    code_tokens=tokenizer.tokenize(code_tokens)[:args.code_length-2]
    code_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    code_ids =  tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = args.code_length - len(code_ids)
    code_ids+=[tokenizer.pad_token_id]*padding_length   

    #nl
    nl=' '.join(js['docstring_tokens'])
    nl_tokens=tokenizer.tokenize(nl)[:args.nl_length-2]
    nl_tokens =[tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]
    nl_ids =  tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.nl_length - len(nl_ids)
    nl_ids+=[tokenizer.pad_token_id]*padding_length  

    return InputFeatures(code_tokens,code_ids,nl_tokens,nl_ids,js['url'])


def convert_examples_to_features_aug_type(js):
    js,tokenizer,args=js
    #code
    if args.lang == "java_mini":
        parser=parsers["java"]
    else:
        parser=parsers[js["language"]]
    # code
    token_type_role = js[ 'bpe_token_type_role']
    code_token = [item[0] for item in token_type_role]
    # code = ' '.join(code_token[:args.code_length-4])
    # code_tokens = tokenizer.tokenize(code)[:args.code_length-4]
    code_tokens = code_token[:args.code_length-4]
    code_tokens =[tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+code_tokens+[tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = args.code_length - len(code_ids)
    code_ids += [tokenizer.pad_token_id]*padding_length

    # code type
    code_type_token = [item[-1] for item in token_type_role]
    # code_type= ' '.join(code_type_token[:args.code_length-4])
    # code_type_tokens = tokenizer.tokenize(code_type)[:args.code_length-4]
    code_type_tokens = code_type_token[:args.code_length-4]
    code_type_tokens =[tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+code_type_tokens+[tokenizer.sep_token]
    code_type_ids = tokenizer.convert_tokens_to_ids(code_type_tokens)
    padding_length = args.code_length - len(code_type_ids)
    code_type_ids += [tokenizer.pad_token_id]*padding_length

    #nl
    nl=' '.join(js['docstring_tokens'])
    nl_tokens = tokenizer.tokenize(nl)[:args.nl_length-4]
    nl_tokens = [tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+nl_tokens+[tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.nl_length - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id]*padding_length 

    return TypeAugInputFeatures(code_tokens,code_ids,code_type_tokens,code_type_ids,nl_tokens,nl_ids,js['url'])
  

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None,pool=None):
        self.args=args
        prefix=file_path.split('/')[-1][:-6]
        cache_file=args.output_dir+'/'+prefix+'.pkl'
        n_debug_samples = args.n_debug_samples
        # if 'codebase' in file_path:
        #     n_debug_samples = 100000
        if 'train' in file_path:
            self.split = "train"
        else:
            self.split = "other"
        if os.path.exists(cache_file):
            self.examples=pickle.load(open(cache_file,'rb'))
            if args.debug:
                self.examples= self.examples[:n_debug_samples]
        else:
            self.examples = []
            data=[]
            if args.debug:
                with open(file_path, encoding="utf-8") as f:
                    for line in f:
                        line=line.strip()
                        js=json.loads(line)
                        data.append((js,tokenizer,args))
                        if len(data) >= n_debug_samples:
                            break
            else:
                with open(file_path, encoding="utf-8") as f:
                    for line in f:
                        line=line.strip()
                        js=json.loads(line)
                        data.append((js,tokenizer,args))
            
            if self.args.data_aug_type == "replace_type":
                self.examples=pool.map(convert_examples_to_features_aug_type, tqdm(data,total=len(data)))
            else:
                self.examples=pool.map(convert_examples_to_features, tqdm(data,total=len(data)))
            
        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("code_tokens: {}".format([x.replace('\u0120','_') for x in example.code_tokens]))
                logger.info("code_ids: {}".format(' '.join(map(str, example.code_ids))))             
                logger.info("nl_tokens: {}".format([x.replace('\u0120','_') for x in example.nl_tokens]))
                logger.info("nl_ids: {}".format(' '.join(map(str, example.nl_ids))))          
                
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item): 
        if self.args.data_aug_type == "replace_type":
            return (torch.tensor(self.examples[item].code_ids),
                    torch.tensor(self.examples[item].code_type_ids),
                    torch.tensor(self.examples[item].nl_ids))
        else:
            return (torch.tensor(self.examples[item].code_ids),
                    torch.tensor(self.examples[item].nl_ids))

        
def convert_examples_to_features_unixcoder(js,tokenizer,args):
    """convert examples to token ids"""
    code = ' '.join(js['code_tokens']) if type(js['code_tokens']) is list else ' '.join(js['code_tokens'].split())
    code_tokens = tokenizer.tokenize(code)[:args.code_length-4]
    code_tokens =[tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+code_tokens+[tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = args.code_length - len(code_ids)
    code_ids += [tokenizer.pad_token_id]*padding_length
    
    nl = ' '.join(js['docstring_tokens']) if type(js['docstring_tokens']) is list else ' '.join(js['doc'].split())
    nl_tokens = tokenizer.tokenize(nl)[:args.nl_length-4]
    nl_tokens = [tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+nl_tokens+[tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.nl_length - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id]*padding_length    
    
    return InputFeatures(code_tokens,code_ids,nl_tokens,nl_ids,js['url'] if "url" in js else js["retrieval_idx"])

class TextDataset_unixcoder(Dataset):
    def __init__(self, tokenizer, args, file_path=None, pooler=None):
        self.examples = []
        data = []
        n_debug_samples = args.n_debug_samples
        with open(file_path) as f:
            if "jsonl" in file_path:
                for line in f:
                    line = line.strip()
                    js = json.loads(line)
                    if 'function_tokens' in js:
                        js['code_tokens'] = js['function_tokens']
                    data.append(js)
                    if args.debug  and len(data) >= n_debug_samples:
                            break
            elif "codebase"in file_path or "code_idx_map" in file_path:
                js = json.load(f)
                for key in js:
                    temp = {}
                    temp['code_tokens'] = key.split()
                    temp["retrieval_idx"] = js[key]
                    temp['doc'] = ""
                    temp['docstring_tokens'] = ""
                    data.append(temp)
                    if  args.debug  and len(data) >= n_debug_samples:
                            break
            elif "json" in file_path:
                for js in json.load(f):
                    data.append(js)
                    if args.debug and len(data) >= n_debug_samples:
                            break 
        # if "test" in file_path:
        #     data = data[-200:]
        for js in data:
            self.examples.append(convert_examples_to_features_unixcoder(js,tokenizer,args))
                
        if "train" in file_path:
            # self.examples = self.examples[:128]
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("code_tokens: {}".format([x.replace('\u0120','_') for x in example.code_tokens]))
                logger.info("code_ids: {}".format(' '.join(map(str, example.code_ids))))
                logger.info("nl_tokens: {}".format([x.replace('\u0120','_') for x in example.nl_tokens]))
                logger.info("nl_ids: {}".format(' '.join(map(str, example.nl_ids))))                             
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):   
        return (torch.tensor(self.examples[i].code_ids),torch.tensor(self.examples[i].nl_ids))
            
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # all gpus
    torch.backends.cudnn.deterministic = True


def mask_tokens(inputs,tokenizer,mlm_probability):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, mlm_probability).to(inputs.device)
    special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
                           labels.tolist()] # for masking special token
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool).to(inputs.device), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0) # masked padding
        
    masked_indices = torch.bernoulli(probability_matrix).bool() # will decide who will be masked
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool().to(inputs.device) & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool().to(inputs.device) & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long).to(inputs.device)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def replace_with_type_tokens(inputs,replaces,tokenizer,mlm_probability):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, mlm_probability).to(inputs.device)
    special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
                           labels.tolist()] # for masking special token
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool).to(inputs.device), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0) # masked padding
        
    masked_indices = torch.bernoulli(probability_matrix).bool() # will decide who will be masked
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool().to(inputs.device) & masked_indices
    inputs[indices_replaced] = replaces[indices_replaced] 

    return inputs, labels

def replace_special_token_with_type_tokens(inputs, speical_token_ids, tokenizer, mlm_probability):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape,0.0).to(inputs.device)   
    probability_matrix.masked_fill_(labels.eq(speical_token_ids).to(inputs.device), value=mlm_probability)
    masked_indices = torch.bernoulli(probability_matrix).bool() # will decide who will be masked
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool().to(inputs.device) & masked_indices
    inputs[indices_replaced] =  speical_token_ids

    return inputs, labels

def replace_special_token_with_mask(inputs, speical_token_ids, tokenizer, mlm_probability):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape,0.0).to(inputs.device)   
    probability_matrix.masked_fill_(labels.eq(speical_token_ids).to(inputs.device), value=mlm_probability)
    masked_indices = torch.bernoulli(probability_matrix).bool() # will decide who will be masked
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool().to(inputs.device) & masked_indices
    inputs[indices_replaced] =tokenizer.convert_tokens_to_ids(tokenizer.mask_token) 

    return inputs, labels

def train(args, model, tokenizer,pool):

    """ Train the model """
    if args.data_aug_type ==  "replace_type" :
        train_dataset=TextDataset(tokenizer, args, args.train_data_file, pool)
    else:
        # if "unixcoder" in args.model_name_or_path or "coco" in args.model_name_or_path :
        train_dataset=TextDataset_unixcoder(tokenizer, args, args.train_data_file, pool)
        # else:
        #     train_dataset=TextDataset(tokenizer, args, args.train_data_file, pool)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=4,drop_last=True)

    model.to(args.device)
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader)*args.num_train_epochs)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Num quene = %d", args.moco_k)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//args.n_gpu)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", len(train_dataloader)*args.num_train_epochs)
    
    model.zero_grad()
    model.train()
    tr_num,tr_loss,best_mrr=0,0,-1
    loss_fct = CrossEntropyLoss()
    # if args.model_type ==  "multi-loss-cocosoda" :
    if args.model_type in  ["no_aug_cocosoda", "multi-loss-cocosoda"]  :
        if args.do_continue_pre_trained:
            logger.info("do_continue_pre_trained")
        elif args.do_fine_tune:
            logger.info("do_fine_tune")
    special_token_list = special_token[args.lang]
    special_token_id_list = tokenizer.convert_tokens_to_ids(special_token_list)

    for idx in range(args.num_train_epochs): 
        for step,batch in enumerate(train_dataloader):
           
                          #get inputs
            code_inputs = batch[0].to(args.device)    
            nl_inputs = batch[1].to(args.device)
            #get code and nl vectors
            code_vec = model(code_inputs=code_inputs)
            nl_vec = model(nl_inputs=nl_inputs)

            tr_num+=1  
            #calculate scores and loss
            scores = torch.einsum("ab,cb->ac",nl_vec,code_vec)
            
            loss = loss_fct(scores*20, torch.arange(code_inputs.size(0), device=scores.device))

            tr_loss += loss.item()
           
            if (step+1)% args.eval_frequency==0:
                logger.info("epoch {} step {} loss {}".format(idx,step+1,round(tr_loss/tr_num,5)))
                tr_loss=0
                tr_num=0
            
            #backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step() 
 
        results = evaluate(args, model, tokenizer,args.eval_data_file, pool, eval_when_training=True)
        for key, value in results.items():
            logger.info("  %s = %s", key, round(value,4))    
            
        #save best model
        if results['eval_mrr']>best_mrr:
            best_mrr=results['eval_mrr']
            logger.info("  "+"*"*20)  
            logger.info("  Best mrr:%s",round(best_mrr,4))
            logger.info("  "+"*"*20)                          

            checkpoint_prefix = 'checkpoint-best-mrr'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)                        
            model_to_save = model.module if hasattr(model,'module') else model
            output_dir = os.path.join(output_dir, '{}'.format('model.bin')) 
            torch.save(model_to_save.state_dict(), output_dir)
            logger.info("Saving model checkpoint to %s", output_dir)

def  multi_lang_continue_pre_train(args, model, tokenizer,pool):
    """ Train the model """
    #get training dataset
    if "unixcoder" in args.model_name_or_path:
        train_datasets = []
        for train_data_file in args.couninue_pre_train_data_files:
            train_dataset=TextDataset_unixcoder(tokenizer, args, train_data_file, pool)
            train_datasets.append(train_dataset)
    else:
        train_datasets = []
        for train_data_file in args.couninue_pre_train_data_files:
            train_dataset=TextDataset(tokenizer, args, train_data_file, pool)
            train_datasets.append(train_dataset)

    train_samplers = [RandomSampler(train_dataset) for train_dataset in train_datasets]
    # https://blog.csdn.net/weixin_44966641/article/details/124878064
    train_dataloaders = [cycle(DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,drop_last=True)) for train_dataset,train_sampler in zip(train_datasets,train_samplers)]
    t_total = args.max_steps

    #get optimizer and scheduler
    # Prepare optimizer and schedule (linear warmup and decay)https://huggingface.co/transformers/v3.3.1/training.html
    model.to(args.device)
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps,num_training_steps=t_total)

    # Train!
    training_data_length = sum ([len(item) for item in train_datasets])
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", training_data_length)
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Num quene = %d", args.moco_k)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//args.n_gpu)
    logger.info("  Total train batch size  = %d", args.train_batch_size)

    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last, map_location="cpu"))
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last, map_location="cpu"))    
    if args.local_rank == 0:
        torch.distributed.barrier()         
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank%args.gpu_per_node],
                                                          output_device=args.local_rank%args.gpu_per_node,
                                                          find_unused_parameters=True)
 
    loss_fct = CrossEntropyLoss()
    set_seed(args.seed)  # Added here for reproducibility (even between python 2 and 3)
    probs=[len(x) for x in train_datasets]
    probs=[x/sum(probs) for x in probs]
    probs=[x**0.7 for x in probs]
    probs=[x/sum(probs) for x in probs]
    # global_step = args.start_step
    model.zero_grad()
    model.train()

    global_step = args.start_step
    step=0
    tr_loss, logging_loss,avg_loss,tr_nb, best_mrr = 0.0, 0.0,0.0,0,-1
    tr_num=0
    special_token_list = all_special_token 
    special_token_id_list = tokenizer.convert_tokens_to_ids(special_token_list)
    while True: 
        
        train_dataloader=np.random.choice(train_dataloaders, 1, p=probs)[0]
        # train_dataloader=train_dataloader[0]
        step+=1
        batch=next(train_dataloader)
        # source_ids= batch.to(args.device)
        model.train()
        # loss = model(source_ids)
        code_inputs = batch[0].to(args.device)  
        code_transformations_ids = code_inputs.clone()
        nl_inputs = batch[1].to(args.device)
        nl_transformations_ids= nl_inputs.clone()
        
        if step%4 == 0:
            code_transformations_ids[:, 3:], _ = mask_tokens(code_inputs.clone()[:, 3:] ,tokenizer,args.mlm_probability)
            nl_transformations_ids[:, 3:], _ = mask_tokens(nl_inputs.clone()[:, 3:] ,tokenizer,args.mlm_probability)
        elif step%4 == 1:
            code_types = code_inputs.clone()
            code_transformations_ids[:, 3:], _ = replace_with_type_tokens(code_inputs.clone()[:, 3:] ,code_types.clone()[:, 3:],tokenizer,args.mlm_probability)
        elif step%4 == 2:
            random.seed( step)
            choice_token_id  = choice(special_token_id_list)
            code_transformations_ids[:, 3:], _ = replace_special_token_with_type_tokens(code_inputs.clone()[:, 3:], choice_token_id, tokenizer,args.mlm_probability)
        elif step%4 == 3:
            random.seed( step)
            choice_token_id  = choice(special_token_id_list)
            code_transformations_ids[:, 3:], _ = replace_special_token_with_mask(code_inputs.clone()[:, 3:], choice_token_id, tokenizer,args.mlm_probability)
        

        tr_num+=1   
        inter_output, inter_target, _, _= model(source_code_q=code_inputs, source_code_k=code_transformations_ids, 
                                    nl_q=nl_inputs , nl_k=nl_transformations_ids )
        
        
        
        # loss_fct = CrossEntropyLoss()
        loss = loss_fct(20*inter_output, inter_target)

        if args.n_gpu > 1:
            loss = loss.mean() # mean() to average on multi-gpu parallel training

            
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        tr_loss += loss.item()
        if (step+1)% args.eval_frequency==0:
            logger.info("step {} loss {}".format(step+1,round(tr_loss/tr_num,5)))
            tr_loss=0
            tr_num=0

        if (step + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()  
            global_step += 1
            output_flag=True
            avg_loss=round((tr_loss - logging_loss) /(global_step- tr_nb),6)

            if global_step %100 == 0:
                logger.info(" global steps (step*gradient_accumulation_steps ): %s loss: %s", global_step, round(avg_loss,6))
            if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                logging_loss = tr_loss
                tr_nb=global_step

            if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                checkpoint_prefix = 'checkpoint-mrr'
                # results = evaluate(args, model, tokenizer,pool=pool,eval_when_training=True)
                results = evaluate(args, model, tokenizer,args.eval_data_file, pool, eval_when_training=True)

                # for key, value in results.items():
                #     logger.info("  %s = %s", key, round(value,6))
                logger.info("  %s = %s", 'eval_mrr', round(results['eval_mrr'],6))

                if results['eval_mrr']>best_mrr:
                    best_mrr=results['eval_mrr']
                    logger.info("  "+"*"*20)  
                    logger.info("  Best mrr:%s",round(best_mrr,4))
                    logger.info("  "+"*"*20)                          

                    output_dir = os.path.join(args.output_dir, '{}'.format('checkpoint-best-mrr'))                        
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)                        
                    model_to_save = model.module if hasattr(model,'module') else model
                    output_dir = os.path.join(output_dir, '{}'.format('model.bin')) 
                    torch.save(model_to_save.state_dict(), output_dir)
                    logger.info("Saving model checkpoint to %s", output_dir)



                # Save model checkpoint
                output_dir = os.path.join(args.output_dir, '{}-{}-{}'.format(checkpoint_prefix, global_step,round(results['eval_mrr'],6)))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = model.module.code_encoder_q  if hasattr(model,'module') else model.code_encoder_q   # Take care of distributed/parallel training
                model_to_save.save_pretrained(output_dir)
                torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                logger.info("Saving model checkpoint to %s", output_dir)

                # _rotate_checkpoints(args, checkpoint_prefix)

                last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                if not os.path.exists(last_output_dir):
                    os.makedirs(last_output_dir)
                model_to_save.save_pretrained(last_output_dir)
                idx_file = os.path.join(last_output_dir, 'idx_file.txt')
                with open(idx_file, 'w', encoding='utf-8') as idxf:
                    idxf.write(str(0) + '\n')

                torch.save(optimizer.state_dict(), os.path.join(last_output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(last_output_dir, "scheduler.pt"))
                logger.info("Saving optimizer and scheduler states to %s", last_output_dir)

                step_file = os.path.join(last_output_dir, 'step_file.txt')
                with open(step_file, 'w', encoding='utf-8') as stepf:
                    stepf.write(str(global_step) + '\n')

            if args.max_steps > 0 and global_step > args.max_steps:
                break


def evaluate(args, model, tokenizer,file_name,pool, eval_when_training=False):
    # if "unixcoder" in args.model_name_or_path or "coco" in args.model_name_or_path :
    dataset_class = TextDataset_unixcoder
    # else:
        # dataset_class = TextDataset
    query_dataset = dataset_class(tokenizer, args, file_name, pool)
    query_sampler = SequentialSampler(query_dataset)
    query_dataloader = DataLoader(query_dataset, sampler=query_sampler, batch_size=args.eval_batch_size,num_workers=4)
    
    code_dataset = dataset_class(tokenizer, args, args.codebase_file, pool)
    code_sampler = SequentialSampler(code_dataset)
    code_dataloader = DataLoader(code_dataset, sampler=code_sampler, batch_size=args.eval_batch_size,num_workers=4)    

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation on %s *****"%args.lang)
    logger.info("  Num queries = %d", len(query_dataset))
    logger.info("  Num codes = %d", len(code_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    
    model.eval()
    model_eval = model.module if hasattr(model,'module') else model
    code_vecs=[] 
    nl_vecs=[]
    for batch in query_dataloader:  
        nl_inputs = batch[-1].to(args.device)
        with torch.no_grad():
            if args.model_type ==  "base" :
                nl_vec = model(nl_inputs=nl_inputs) 

            elif args.model_type in  ["cocosoda" ,"no_aug_cocosoda", "multi-loss-cocosoda"]:
                outputs = model_eval.nl_encoder_q(nl_inputs, attention_mask=nl_inputs.ne(1))
                if args.agg_way == "avg":
                    outputs = outputs [0]
                    nl_vec = (outputs*nl_inputs.ne(1)[:,:,None]).sum(1)/nl_inputs.ne(1).sum(-1)[:,None] # None作为ndarray或tensor的索引作用是增加维度，
                elif args.agg_way == "cls_pooler":
                    nl_vec =outputs [1]
                elif args.agg_way == "avg_cls_pooler":
                     nl_vec =outputs [1] +  (outputs[0]*nl_inputs.ne(1)[:,:,None]).sum(1)/nl_inputs.ne(1).sum(-1)[:,None] 
                nl_vec  = torch.nn.functional.normalize( nl_vec, p=2, dim=1)
                if args.do_whitening:
                    nl_vec=whitening_torch_final(nl_vec)


            
            nl_vecs.append(nl_vec.cpu().numpy()) 

    for batch in code_dataloader:
        with torch.no_grad():
            code_inputs = batch[0].to(args.device)
            if args.model_type ==  "base" :
                code_vec = model(code_inputs=code_inputs)
            elif args.model_type in  ["cocosoda" ,"no_aug_cocosoda", "multi-loss-cocosoda"]:
                # code_vec =  model_eval.code_encoder_q(code_inputs, attention_mask=code_inputs.ne(1))[1]
                outputs = model_eval.code_encoder_q(code_inputs, attention_mask=code_inputs.ne(1))
                if args.agg_way == "avg":
                    outputs = outputs [0]
                    code_vec  = (outputs*code_inputs.ne(1)[:,:,None]).sum(1)/code_inputs.ne(1).sum(-1)[:,None] # None作为ndarray或tensor的索引作用是增加维度，
                elif args.agg_way == "cls_pooler":
                    code_vec=outputs [1]
                elif args.agg_way == "avg_cls_pooler":
                     code_vec=outputs [1] +  (outputs[0]*code_inputs.ne(1)[:,:,None]).sum(1)/code_inputs.ne(1).sum(-1)[:,None] 
                code_vec  = torch.nn.functional.normalize(code_vec, p=2, dim=1)
                if args.do_whitening:
                    code_vec=whitening_torch_final(code_vec)
        
            
            
            code_vecs.append(code_vec.cpu().numpy())  

    model.train()    
    code_vecs=np.concatenate(code_vecs,0)
    nl_vecs=np.concatenate(nl_vecs,0)

    scores=np.matmul(nl_vecs,code_vecs.T)
    
    sort_ids=np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1]    
    
    nl_urls=[]
    code_urls=[]
    for example in query_dataset.examples:
        nl_urls.append(example.url)
        
    for example in code_dataset.examples:
        code_urls.append(example.url)
        
    ranks=[]
    for url, sort_id in zip(nl_urls,sort_ids):
        rank=0
        find=False
        for idx in sort_id[:1000]:
            if find is False:
                rank+=1
            if code_urls[idx]==url:
                find=True
        if find:
            ranks.append(1/rank)
        else:
            ranks.append(0)
    if args.save_evaluation_reuslt:
        evaluation_result = {"nl_urls":nl_urls, "code_urls":code_urls,"sort_ids":sort_ids[:,:10],"ranks":ranks}
        save_pickle_data(args.save_evaluation_reuslt_dir, "evaluation_result.pkl",evaluation_result)
    result = cal_r1_r5_r10(ranks)
    result["eval_mrr"]  = round(float(np.mean(ranks)),3)
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    # soda
    parser.add_argument('--data_aug_type',default="replace_type",choices=["replace_type", "random_mask" ,"other"], help="the ways of soda",required=False)
    parser.add_argument('--aug_type_way',default="random_replace_type",choices=["random_replace_type", "replace_special_type" ,"replace_special_type_with_mask"], help="the ways of soda",required=False)
    parser.add_argument('--print_align_unif_loss', action='store_true', help='print_align_unif_loss', required=False)
    parser.add_argument('--do_ineer_loss', action='store_true', help='print_align_unif_loss', required=False)
    parser.add_argument('--only_save_the_nl_code_vec', action='store_true', help='print_align_unif_loss', required=False)
    parser.add_argument('--do_zero_short', action='store_true', help='print_align_unif_loss', required=False)
    parser.add_argument('--agg_way',default="avg",choices=["avg", "cls_pooler","avg_cls_pooler" ], help="base is codebert/graphcoder/unixcoder",required=False)
    parser.add_argument('--weight_decay',default=0.01, type=float,required=False)
    parser.add_argument('--do_single_lang_continue_pre_train', action='store_true', help='do_single_lang_continue_pre_train', required=False)
    parser.add_argument('--save_evaluation_reuslt', action='store_true', help='save_evaluation_reuslt', required=False)
    parser.add_argument('--save_evaluation_reuslt_dir', type=str, help='save_evaluation_reuslt', required=False)
    parser.add_argument('--epoch', type=int, default=50,
                        help="random seed for initialization")
    # new continue pre-training
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--loaded_model_filename", type=str, required=False,
                        help="loaded_model_filename")
    parser.add_argument("--loaded_codebert_model_filename", type=str, required=False,
                        help="loaded_model_filename")
    parser.add_argument('--do_multi_lang_continue_pre_train', action='store_true', help='do_multi_lang_continue_pre_train', required=False)
    parser.add_argument("--couninue_pre_train_data_files", default=["dataset/ruby/train.jsonl",  "dataset/java/train.jsonl",], type=str, nargs='+', required=False,
                        help="The input training data files (some json files).")
    # parser.add_argument("--couninue_pre_train_data_files", default=["dataset/go/train.jsonl",  "dataset/java/train.jsonl",
    # "dataset/javascript/train.jsonl",  "dataset/php/train.jsonl",  "dataset/python/train.jsonl",  "dataset/ruby/train.jsonl",], type=list, required=False,
    #                     help="The input training data files (some json files).")
    parser.add_argument('--do_continue_pre_trained', action='store_true', help='debug mode', required=False)
    parser.add_argument('--do_fine_tune', action='store_true', help='debug mode', required=False)
    parser.add_argument('--do_whitening', action='store_true', help='do_whitening https://github.com/Jun-jie-Huang/WhiteningBERT', required=False)
    parser.add_argument("--time_score", default=1, type=int,help="cosine value * time_score")   
    parser.add_argument("--max_steps", default=100, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--num_warmup_steps", default=0, type=int, help="num_warmup_steps")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")    
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    # new moco
    parser.add_argument('--moco_type',default="encoder_queue",choices=["encoder_queue","encoder_momentum_encoder_queue" ], help="base is codebert/graphcoder/unixcoder",required=False)

    
    # debug
    parser.add_argument('--use_best_mrr_model', action='store_true', help='cosine_space', required=False)
    parser.add_argument('--debug', action='store_true', help='debug mode', required=False)
    parser.add_argument('--n_debug_samples', type=int, default=100, required=False)
    parser.add_argument("--max_codeblock_num", default=10, type=int,
                        help="Optional NL input sequence length after tokenization.")    
    parser.add_argument('--hidden_size', type=int, default=768, required=False)
    parser.add_argument("--eval_frequency", default=1, type=int, required=False)
    parser.add_argument("--mlm_probability", default=0.1, type=float, required=False)

    # model type
    parser.add_argument('--do_avg', action='store_true', help='avrage hidden status', required=False)
    parser.add_argument('--model_type',default="base",choices=["base", "cocosoda","multi-loss-cocosoda","no_aug_cocosoda"], help="base is codebert/graphcoder/unixcoder",required=False)
    # moco
    # moco specific configs:
    parser.add_argument('--moco_dim', default=768, type=int,
                        help='feature dimension (default: 768)')
    parser.add_argument('--moco_k', default=32, type=int,
                        help='queue size; number of negative keys (default: 65536), which is divided by 32, etc.')
    parser.add_argument('--moco_m', default=0.999, type=float,
                        help='moco momentum of updating key encoder (default: 0.999)')
    parser.add_argument('--moco_t', default=0.07, type=float,
                        help='softmax temperature (default: 0.07)')

    # options for moco v2
    parser.add_argument('--mlp', action='store_true',help='use mlp head')

    ## Required parameters
    parser.add_argument("--train_data_file", default="dataset/java/train.jsonl", type=str, required=False,
                        help="The input training data file (a json file).")
    parser.add_argument("--output_dir", default="saved_models/pre-train", type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_data_file", default="dataset/java/valid.jsonl", type=str,
                        help="An optional input evaluation data file to evaluate the MRR(a jsonl file).")
    parser.add_argument("--test_data_file", default="dataset/java/test.jsonl", type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")
    parser.add_argument("--codebase_file", default="dataset/java/codebase.jsonl", type=str,
                        help="An optional input test data file to codebase (a jsonl file).")  
    
    parser.add_argument("--lang", default="java", type=str,
                        help="language.")  
    
    parser.add_argument("--model_name_or_path", default="microsoft/graphcodebert-base", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="microsoft/graphcodebert-base", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="microsoft/graphcodebert-base", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    
    parser.add_argument("--nl_length", default=50, type=int,
                        help="Optional NL input sequence length after tokenization.")    
    parser.add_argument("--code_length", default=100, type=int,
                        help="Optional Code input sequence length after tokenization.") 
    parser.add_argument("--data_flow_length", default=0, type=int,
                        help="Optional Data Flow input sequence length after tokenization.",required=False) 
    
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")  
    
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=4, type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument('--seed', type=int, default=3407,
                        help="random seed for initialization")  
        
    #print arguments
    args = parser.parse_args()
    return  args                     

def create_model(args,model,tokenizer, config=None):
    # logger.info("args.data_aug_type %s"%args.data_aug_type)
    # replace token with type
    if args.data_aug_type in ["replace_type" , "other"] and not args.only_save_the_nl_code_vec:
        special_tokens_dict = {'additional_special_tokens': all_special_token}
        logger.info(" new token %s"%(str(special_tokens_dict)))
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))
  
    if (args.loaded_model_filename) and ("pytorch_model.bin" in args.loaded_model_filename):
        logger.info("reload pytorch model from {}".format(args.loaded_model_filename))
        model.load_state_dict(torch.load(args.loaded_model_filename),strict=False) 
        # model.from_pretrain
    if args.model_type ==  "base" :
        model= Model(model)
    elif args.model_type ==  "multi-loss-cocosoda":
        model= Multi_Loss_CoCoSoDa(model,args, args.mlp)
    if (args.loaded_model_filename) and ("pytorch_model.bin" not in args.loaded_model_filename) :
        logger.info("reload model from {}".format(args.loaded_model_filename))
        model.load_state_dict(torch.load(args.loaded_model_filename)) 
        # model.load_state_dict(torch.load(args.loaded_model_filename,strict=False)) 
        # model.from_pretrained(args.loaded_model_filename)  
    if (args.loaded_codebert_model_filename) :
        logger.info("reload pytorch model from {}".format(args.loaded_codebert_model_filename))
        model.load_state_dict(torch.load(args.loaded_codebert_model_filename),strict=False)   
    logger.info(model.model_parameters())


    return model

def main():
    
    args = parse_args()
    #set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )
    #set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s",device, args.n_gpu)
    
    pool = multiprocessing.Pool(cpu_cont)

    # Set seed
    set_seed(args.seed)

    #build model

    if "codet5" in   args.model_name_or_path:
        config = T5Config.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
        tokenizer =  RobertaTokenizer.from_pretrained(args.tokenizer_name)
        model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
        model = model.encoder
    else:
        config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
        tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
        model = RobertaModel.from_pretrained(args.model_name_or_path) 
    model=create_model(args,model,tokenizer,config)

    logger.info("Training/evaluation parameters %s", args)
    args.start_step = 0

    model.to(args.device)
    
    # Training
    if args.do_multi_lang_continue_pre_train:
        multi_lang_continue_pre_train(args, model, tokenizer, pool)
        output_tokenizer_dir = os.path.join(args.output_dir,"tokenzier")                      
        if not os.path.exists(output_tokenizer_dir):
            os.makedirs( output_tokenizer_dir)    
        tokenizer.save_pretrained( output_tokenizer_dir)
    if args.do_train:
        train(args, model, tokenizer, pool)
     
    
    # Evaluation
    results = {}

    if args.do_eval:
        checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        if (not args.only_save_the_nl_code_vec) and (not args.do_zero_short) :
            model.load_state_dict(torch.load(output_dir),strict=False)      
        model.to(args.device)
        result=evaluate(args, model, tokenizer,args.eval_data_file, pool)
        logger.info("***** Eval valid results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key],4)))
            
    if args.do_test:

        logger.info("runnning test")
        checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        if (not args.only_save_the_nl_code_vec) and (not args.do_zero_short) :
            model.load_state_dict(torch.load(output_dir),strict=False)      
        model.to(args.device)
        result=evaluate(args, model, tokenizer,args.test_data_file, pool)
        logger.info("***** Eval test results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key],4)))
        save_json_data(args.output_dir, "result.jsonl", result)
    return results


if __name__ == "__main__":
    main()

