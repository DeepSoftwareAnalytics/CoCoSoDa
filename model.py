import torch
import torch.nn as nn
from prettytable import PrettyTable
from torch.nn.modules.activation import Tanh
import copy
import logging
logger = logging.getLogger(__name__)
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                  RobertaConfig, RobertaModel, RobertaTokenizer)
def whitening_torch_final(embeddings):
    mu = torch.mean(embeddings, dim=0, keepdim=True)
    cov = torch.mm((embeddings - mu).t(), embeddings - mu)
    u, s, vt = torch.svd(cov)
    W = torch.mm(u, torch.diag(1/torch.sqrt(s)))
    embeddings = torch.mm(embeddings - mu, W)
    return embeddings

class BaseModel(nn.Module): 
    def __init__(self, ):
        super().__init__()
        
    def model_parameters(self):
        table = PrettyTable()
        table.field_names = ["Layer Name", "Output Shape", "Param #"]
        table.align["Layer Name"] = "l"
        table.align["Output Shape"] = "r"
        table.align["Param #"] = "r"
        for name, parameters in self.named_parameters():
            if parameters.requires_grad:
                table.add_row([name, str(list(parameters.shape)), parameters.numel()])
        return table
class Model(BaseModel):   
    def __init__(self, encoder):
        super(Model, self).__init__()
        self.encoder = encoder
      
    def forward(self, code_inputs=None, nl_inputs=None): 
        # code_inputs [bs, seq] 
        if code_inputs is not None:
            outputs = self.encoder(code_inputs,attention_mask=code_inputs.ne(1))[0] #[bs, seq_len, dim]
            outputs = (outputs*code_inputs.ne(1)[:,:,None]).sum(1)/code_inputs.ne(1).sum(-1)[:,None] # None作为ndarray或tensor的索引作用是增加维度，
            return torch.nn.functional.normalize(outputs, p=2, dim=1)
        else:
            outputs = self.encoder(nl_inputs,attention_mask=nl_inputs.ne(1))[0]
            outputs = (outputs*nl_inputs.ne(1)[:,:,None]).sum(1)/nl_inputs.ne(1).sum(-1)[:,None]
            return torch.nn.functional.normalize(outputs, p=2, dim=1)
        
 
class Multi_Loss_CoCoSoDa( BaseModel):

    def __init__(self, base_encoder, args, mlp=False):
        super(Multi_Loss_CoCoSoDa, self).__init__()

        self.K = args.moco_k
        self.m = args.moco_m
        self.T = args.moco_t
        dim= args.moco_dim

        # create the encoders
        # num_classes is the output fc dimension
        self.code_encoder_q = base_encoder
        self.code_encoder_k = copy.deepcopy(base_encoder)
        self.nl_encoder_q = base_encoder
        # self.nl_encoder_q = RobertaModel.from_pretrained("roberta-base")   
        self.nl_encoder_k = copy.deepcopy(self.nl_encoder_q)
        self.mlp = mlp
        self.time_score= args.time_score
        self.do_whitening = args.do_whitening
        self.do_ineer_loss = args.do_ineer_loss 
        self.agg_way = args.agg_way
        self.args = args

        for param_q, param_k in zip(self.code_encoder_q.parameters(), self.code_encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.nl_encoder_q.parameters(), self.nl_encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the code queue
        torch.manual_seed(3047)
        torch.cuda.manual_seed(3047)
        self.register_buffer("code_queue", torch.randn(dim,self.K ))
        self.code_queue = nn.functional.normalize(self.code_queue, dim=0)
        self.register_buffer("code_queue_ptr", torch.zeros(1, dtype=torch.long))
        # create the masked code queue
        self.register_buffer("masked_code_queue", torch.randn(dim, self.K ))
        self.masked_code_queue = nn.functional.normalize(self.masked_code_queue, dim=0)
        self.register_buffer("masked_code_queue_ptr", torch.zeros(1, dtype=torch.long))


        # create the nl queue
        self.register_buffer("nl_queue", torch.randn(dim, self.K ))
        self.nl_queue = nn.functional.normalize(self.nl_queue, dim=0)
        self.register_buffer("nl_queue_ptr", torch.zeros(1, dtype=torch.long))
        # create the masked nl  queue
        self.register_buffer("masked_nl_queue", torch.randn(dim, self.K ))
        self.masked_nl_queue= nn.functional.normalize(self.masked_nl_queue, dim=0)
        self.register_buffer("masked_nl_queue_ptr", torch.zeros(1, dtype=torch.long))




    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        % key encoder的Momentum update
        """
        for param_q, param_k in zip(self.code_encoder_q.parameters(), self.code_encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.nl_encoder_q.parameters(), self.nl_encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        if self.mlp:
            for param_q, param_k in zip(self.code_encoder_q_fc.parameters(), self.code_encoder_k_fc.parameters()):
                param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
            for param_q, param_k in zip(self.nl_encoder_q_fc.parameters(), self.nl_encoder_k_fc.parameters()):
                param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, option='code'):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]
        if option == 'code':
            code_ptr = int(self.code_queue_ptr)
            assert self.K % batch_size == 0  # for simplicity

            # replace the keys at ptr (dequeue and enqueue)
            try:
                self.code_queue[:, code_ptr:code_ptr + batch_size] = keys.T
            except:
                print(code_ptr)
                print(batch_size)
                print(keys.shape)
                exit(111)
            code_ptr = (code_ptr + batch_size) % self.K  # move pointer  ptr->pointer

            self.code_queue_ptr[0] = code_ptr
        
        elif option == 'masked_code':
            masked_code_ptr = int(self.masked_code_queue_ptr)
            assert self.K % batch_size == 0  # for simplicity

            # replace the keys at ptr (dequeue and enqueue)
            try:
                self.masked_code_queue[:, masked_code_ptr:masked_code_ptr + batch_size] = keys.T
            except:
                print(masked_code_ptr)
                print(batch_size)
                print(keys.shape)
                exit(111)
            masked_code_ptr = (masked_code_ptr + batch_size) % self.K  # move pointer  ptr->pointer

            self.masked_code_queue_ptr[0] = masked_code_ptr
        
        elif option == 'nl':

            nl_ptr = int(self.nl_queue_ptr)
            assert self.K % batch_size == 0  # for simplicity

            # replace the keys at ptr (dequeue and enqueue)
            self.nl_queue[:, nl_ptr:nl_ptr + batch_size] = keys.T
            nl_ptr = (nl_ptr + batch_size) % self.K  # move pointer  ptr->pointer

            self.nl_queue_ptr[0] = nl_ptr
        elif option == 'masked_nl':

            masked_nl_ptr = int(self.masked_nl_queue_ptr)
            assert self.K % batch_size == 0  # for simplicity

            # replace the keys at ptr (dequeue and enqueue)
            self.masked_nl_queue[:, masked_nl_ptr:masked_nl_ptr + batch_size] = keys.T
            masked_nl_ptr = (masked_nl_ptr + batch_size) % self.K  # move pointer  ptr->pointer

            self.masked_nl_queue_ptr[0] = masked_nl_ptr

    

    def forward(self, source_code_q, source_code_k, nl_q,nl_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        if not self.args.do_multi_lang_continue_pre_train:
            # logger.info(".do_multi_lang_continue_pre_train")
            outputs = self.code_encoder_q(source_code_q, attention_mask=source_code_q.ne(1))[0]
            code_q  = (outputs*source_code_q.ne(1)[:,:,None]).sum(1)/source_code_q.ne(1).sum(-1)[:,None] # None作为ndarray或tensor的索引作用是增加维度，
            code_q  = torch.nn.functional.normalize(code_q, p=2, dim=1)
            # compute query features for nl
            outputs= self.nl_encoder_q(nl_q, attention_mask=nl_q.ne(1))[0]  # queries: NxC   bs*feature_dim
            nl_q = (outputs*nl_q.ne(1)[:,:,None]).sum(1)/nl_q.ne(1).sum(-1)[:,None]
            nl_q = torch.nn.functional.normalize(nl_q, p=2, dim=1)
            code2nl_logits = torch.einsum("ab,cb->ac", code_q,nl_q )
            # loss = self.loss_fct(scores*20, torch.arange(code_inputs.size(0), device=scores.device))
            code2nl_logits /= self.T
            # label
            code2nl_label = torch.arange(code2nl_logits.size(0), device=code2nl_logits.device)
            return code2nl_logits,code2nl_label, None, None
        if self.agg_way == "avg":
            # compute query features for source code
            outputs = self.code_encoder_q(source_code_q, attention_mask=source_code_q.ne(1))[0]
            code_q  = (outputs*source_code_q.ne(1)[:,:,None]).sum(1)/source_code_q.ne(1).sum(-1)[:,None] # None作为ndarray或tensor的索引作用是增加维度，
            code_q  = torch.nn.functional.normalize(code_q, p=2, dim=1)
            # compute query features for nl
            outputs= self.nl_encoder_q(nl_q, attention_mask=nl_q.ne(1))[0]  # queries: NxC   bs*feature_dim
            nl_q = (outputs*nl_q.ne(1)[:,:,None]).sum(1)/nl_q.ne(1).sum(-1)[:,None]
            nl_q = torch.nn.functional.normalize(nl_q, p=2, dim=1)

            # compute key features
            with torch.no_grad():  # no gradient to keys
                self._momentum_update_key_encoder()  # update the key encoder

                # shuffle for making use of BN
                # im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

                # masked code
                outputs = self.code_encoder_k(source_code_k, attention_mask=source_code_k.ne(1))[0]  # keys: NxC
                code_k  = (outputs*source_code_k.ne(1)[:,:,None]).sum(1)/source_code_k.ne(1).sum(-1)[:,None] # None作为ndarray或tensor的索引作用是增加维度，
                code_k  = torch.nn.functional.normalize( code_k, p=2, dim=1)
                # masked nl
                outputs = self.nl_encoder_k(nl_k, attention_mask=nl_k.ne(1))[0]   # keys: bs*dim
                nl_k = (outputs*nl_k.ne(1)[:,:,None]).sum(1)/nl_k.ne(1).sum(-1)[:,None]
                nl_k = torch.nn.functional.normalize(nl_k, p=2, dim=1)

        elif self.agg_way == "cls_pooler":
            # logger.info(self.agg_way )
            # compute query features for source code
            outputs = self.code_encoder_q(source_code_q, attention_mask=source_code_q.ne(1))[1]
            code_q  = torch.nn.functional.normalize(code_q, p=2, dim=1)
            # compute query features for nl
            outputs= self.nl_encoder_q(nl_q, attention_mask=nl_q.ne(1))[1]  # queries: NxC   bs*feature_dim
            nl_q = torch.nn.functional.normalize(nl_q, p=2, dim=1)

            # compute key features
            with torch.no_grad():  # no gradient to keys
                self._momentum_update_key_encoder()  # update the key encoder

                # shuffle for making use of BN
                # im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

                # masked code
                outputs = self.code_encoder_k(source_code_k, attention_mask=source_code_k.ne(1))[1]  # keys: NxC
                code_k  = torch.nn.functional.normalize( code_k, p=2, dim=1)
                # masked nl
                outputs = self.nl_encoder_k(nl_k, attention_mask=nl_k.ne(1))[1]   # keys: bs*dim
                nl_k = torch.nn.functional.normalize(nl_k, p=2, dim=1)

        elif self.agg_way == "avg_cls_pooler":
            # logger.info(self.agg_way )
            outputs = self.code_encoder_q(source_code_q, attention_mask=source_code_q.ne(1))
            code_q_cls = outputs[1]
            outputs = outputs[0]
            code_q_avg  = (outputs*source_code_q.ne(1)[:,:,None]).sum(1)/source_code_q.ne(1).sum(-1)[:,None] # None作为ndarray或tensor的索引作用是增加维度，
            code_q = code_q_cls + code_q_avg 
            code_q  = torch.nn.functional.normalize(code_q, p=2, dim=1)
            # compute query features for nl
            outputs= self.nl_encoder_q(nl_q, attention_mask=nl_q.ne(1))
            nl_q_cls = outputs[1]
            outputs= outputs[0]  # queries: NxC   bs*feature_dim
            nl_q_avg = (outputs*nl_q.ne(1)[:,:,None]).sum(1)/nl_q.ne(1).sum(-1)[:,None]
            nl_q = nl_q_avg  + nl_q_cls 
            nl_q = torch.nn.functional.normalize(nl_q, p=2, dim=1)

            # compute key features
            with torch.no_grad():  # no gradient to keys
                self._momentum_update_key_encoder()  # update the key encoder

                # shuffle for making use of BN
                # im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

                # masked code

                outputs = self.code_encoder_k(source_code_k, attention_mask=source_code_k.ne(1))
                code_k_cls = outputs[1]  # keys: NxC
                outputs = outputs[0]
                code_k_avg  = (outputs*source_code_k.ne(1)[:,:,None]).sum(1)/source_code_k.ne(1).sum(-1)[:,None] # None作为ndarray或tensor的索引作用是增加维度，
                code_k =  code_k_cls + code_k_avg
                code_k  = torch.nn.functional.normalize( code_k, p=2, dim=1)
                # masked nl
                outputs = self.nl_encoder_k(nl_k, attention_mask=nl_k.ne(1))
                nl_k_cls = outputs[1]   # keys: bs*dim
                outputs = outputs[0]
                nl_k_avg = (outputs*nl_k.ne(1)[:,:,None]).sum(1)/nl_k.ne(1).sum(-1)[:,None]
                nl_k =  nl_k_cls + nl_k_avg
                nl_k = torch.nn.functional.normalize(nl_k, p=2, dim=1)
            
        # ## do_whitening
        # if self.do_whitening:
        #     code_q = whitening_torch_final(code_q)
        #     code_k = whitening_torch_final(code_k)
        #     nl_q = whitening_torch_final(nl_q)
        #     nl_k = whitening_torch_final(nl_k)


        ## code vs nl
        code2nl_pos = torch.einsum('nc,bc->nb', [code_q, nl_q])
        # negative logits: NxK
        code2nl_neg = torch.einsum('nc,ck->nk', [code_q, self.nl_queue.clone().detach()])
        # logits: Nx(n+K)
        code2nl_logits = torch.cat([self.time_score*code2nl_pos, code2nl_neg], dim=1)
        # apply temperature
        code2nl_logits /= self.T
        # label
        code2nl_label = torch.arange(code2nl_logits.size(0), device=code2nl_logits.device)

        ## code vs masked nl
        code2maskednl_pos = torch.einsum('nc,bc->nb', [code_q, nl_k])
        # negative logits: NxK
        code2maskednl_neg = torch.einsum('nc,ck->nk', [code_q, self.masked_nl_queue.clone().detach()])
        # logits: Nx(n+K)
        code2maskednl_logits = torch.cat([self.time_score*code2maskednl_pos, code2maskednl_neg], dim=1)
        # apply temperature
        code2maskednl_logits /= self.T
        # label
        code2maskednl_label = torch.arange(code2maskednl_logits.size(0), device=code2maskednl_logits.device)

        ## nl vs code
        # nl2code_pos = torch.einsum('nc,nc->n', [nl_q, code_k]).unsqueeze(-1)
        nl2code_pos = torch.einsum('nc,bc->nb', [nl_q, code_q])
        # negative logits: bsxK
        nl2code_neg = torch.einsum('nc,ck->nk', [nl_q, self.code_queue.clone().detach()])
        # nl2code_logits: bsx(n+K)
        nl2code_logits = torch.cat([self.time_score*nl2code_pos, nl2code_neg], dim=1)
        # apply temperature
        nl2code_logits /= self.T
        # label
        nl2code_label = torch.arange(nl2code_logits.size(0), device=nl2code_logits.device)

        ## nl vs masked code
        # nl2code_pos = torch.einsum('nc,nc->n', [nl_q, code_k]).unsqueeze(-1)
        nl2maskedcode_pos = torch.einsum('nc,bc->nb', [nl_q, code_k])
        # negative logits: bsxK
        nl2maskedcode_neg = torch.einsum('nc,ck->nk', [nl_q, self.masked_code_queue.clone().detach()])
        # nl2code_logits: bsx(n+K)
        nl2maskedcode_logits = torch.cat([self.time_score*nl2maskedcode_pos, nl2maskedcode_neg], dim=1)
        # apply temperature
        nl2maskedcode_logits /= self.T
        # label
        nl2maskedcode_label = torch.arange(nl2maskedcode_logits.size(0), device=nl2maskedcode_logits.device)
        
        #logit 4*bsx(1+K)
        inter_logits = torch.cat((code2nl_logits, code2maskednl_logits, nl2code_logits ,nl2maskedcode_logits ), dim=0)

        # labels: positive key indicators
        # inter_labels = torch.zeros(inter_logits.shape[0], dtype=torch.long).cuda()
        inter_labels =  torch.cat((code2nl_label, code2maskednl_label, nl2code_label, nl2maskedcode_label), dim=0)

        if self.do_ineer_loss:
            # logger.info("do_ineer_loss")
            ## code vs masked code
            code2maskedcode_pos = torch.einsum('nc,bc->nb', [code_q, code_k])
            # negative logits: NxK
            code2maskedcode_neg = torch.einsum('nc,ck->nk', [code_q, self.masked_code_queue.clone().detach()])
            # logits: Nx(n+K)
            code2maskedcode_logits = torch.cat([self.time_score*code2maskedcode_pos, code2maskedcode_neg], dim=1)
            # apply temperature
            code2maskedcode_logits /= self.T
            # label
            code2maskedcode_label = torch.arange(code2maskedcode_logits.size(0), device=code2maskedcode_logits.device)


            ## nl vs masked nl
            # nl2code_pos = torch.einsum('nc,nc->n', [nl_q, code_k]).unsqueeze(-1)
            nl2maskednl_pos = torch.einsum('nc,bc->nb', [nl_q, nl_k])
            # negative logits: bsxK
            nl2maskednl_neg = torch.einsum('nc,ck->nk', [nl_q, self.masked_nl_queue.clone().detach()])
            # nl2code_logits: bsx(n+K)
            nl2maskednl_logits = torch.cat([self.time_score*nl2maskednl_pos, nl2maskednl_neg], dim=1)
            # apply temperature
            nl2maskednl_logits /= self.T
            # label
            nl2maskednl_label = torch.arange(nl2maskednl_logits.size(0), device=nl2maskednl_logits.device)
            

            #logit 6*bsx(1+K)
            inter_logits = torch.cat((inter_logits, code2maskedcode_logits, nl2maskednl_logits), dim=0)

            # labels: positive key indicators
            # inter_labels = torch.zeros(inter_logits.shape[0], dtype=torch.long).cuda()
            inter_labels =  torch.cat(( inter_labels, code2maskedcode_label, nl2maskednl_label ), dim=0)


        # dequeue and enqueue
        self._dequeue_and_enqueue(code_q, option='code')
        self._dequeue_and_enqueue(nl_q, option='nl')
        self._dequeue_and_enqueue(code_k, option='masked_code')
        self._dequeue_and_enqueue(nl_k, option='masked_nl')

        return inter_logits, inter_labels, code_q, nl_q 

