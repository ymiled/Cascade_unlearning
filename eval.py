import argparse
from argparse import ArgumentParser
import os
import json
import pandas as pd
import numpy as np

# ML pipeline
from datasets import load_dataset, Dataset, DatasetDict, load_from_disk
import torch
from torch.utils.data import DataLoader
torch.multiprocessing.set_sharing_strategy('file_system')
from transformers import AutoTokenizer, GPT2Tokenizer
from transformers import AutoModelForCausalLM, BloomForCausalLM

# local imports
from local_dataset_utilities import DS

def tokenize_text(batch):
    return tokenizer(batch["text"], truncation=True, padding=True, max_length=4096)

def get_dataset(config, ind_mask):
    print(f'Processing {config.dataset_name} data ...')
    # fix random seed for reproducibility
    dataset_size = ind_mask.shape[0]
    pth = f"{config.path_prefix_data}/datasets/{config.dataset_name}_n{dataset_size}"
    dataset = load_from_disk(pth)
    
    ''' get train / test splits '''
    all_indices = np.arange(dataset_size)
    print(ind_mask.shape)
    train_indices = all_indices[ind_mask]
    test_indices = all_indices[~ind_mask]
    dataset_train = dataset.select(train_indices)
    dataset_test = dataset.select(test_indices)
    print('train size', len(dataset_train['text']))
    print('test size', len(dataset_test['text']))
    dataset = DatasetDict({"train": dataset_train, "test": dataset_test})
    
    return dataset


def eval_model(model, tokenized_text, n_additional):
    '''Returns model outputs for given batch'''
    model.cuda().eval()
    with torch.no_grad():
        # print('shape of input', tokenized_text["input_ids"].shape)
        # print('length of input', length)
        length = tokenized_text["input_ids"][0].shape[0]
        out = model.generate(input_ids=tokenized_text["input_ids"].cuda(),
                             attention_mask=tokenized_text["attention_mask"].cuda(),
                             max_length=length+n_additional,
                             return_dict_in_generate=True,
                             output_scores=True)

    return out


def get_icl_vary_context(batch, batch_eval, batch_context, batch_label, icl_unlearn_labels, config):
    ''' When batch_size == 1, sets up context of the form: <ex 1> \n <ex 2> \n ... <ex querry> \\ '''
    assert config.batch_sizes[0] == 1
    if batch_label == 'forget':
        batch_eval = batch
    length = batch['input_ids'][0].shape[0]
    length_eval = batch_eval['input_ids'][0].shape[0]
    # context for query position
    txt_eval = tokenizer.decode(batch_eval['input_ids'][0][0:(length_eval-1)], skip_special_tokens=True)
    # content for position 1
    txt = tokenizer.decode(batch['input_ids'][0][0:(length-1)], skip_special_tokens=True)
    # content for remaining context positions
    txts_ctxt = []
    for j in range(batch_context['input_ids'].shape[0]):
        length_ctx = batch_context['input_ids'][j].shape[0]
        txt_ctxt = tokenizer.decode(batch_context['input_ids'][j][0:(length_ctx-1)])
        txts_ctxt.append(txt_ctxt)
    # print('txts context:', txts_ctxt)
    # print('txt eval:', txt_eval)
    # print('txt:', txt)
    # setup full context
    for j in range(config.n_ctxt+1):
        if j == 0:
            # instantiate appended post
            appended_post = txt + ' ' + icl_unlearn_labels[j] + "\n"
        elif j < (config.n_ctxt):
            # structure: <ex 1> \n <ex 2> \n ... <ex k> \n <ask ex 1>
            appended_post += txts_ctxt[j-1] + ' ' + icl_unlearn_labels[j] + "\n" 
        else:
            appended_post += txt_eval
        # print(f'iteration {j} out of {config.n_ctxt}:', appended_post)
    return appended_post
    
    
def get_icl_vary_context_batchsize_gg1(batch, batch_eval, batch_context, batch_label, icl_unlearn_labels, idx_to_evaluate, config):
    ''' When batch_size > 1, sets up context of the form: <ex 1> \n <ex 2> \n ... <ex querry> \\ '''
    batch_size = batch['input_ids'].shape[0]
    if batch_label == 'forget':
        batch_eval = batch
    length = batch['input_ids'][0].shape[0]
    length_eval = batch_eval['input_ids'][0].shape[0]
    # context for query position
    txt_eval = tokenizer.decode(batch_eval['input_ids'][idx_to_evaluate][0:(length_eval-1)], skip_special_tokens=True)
    # content for first batch_size positions
    txts_instruction = []
    for b in range(batch_size):
        txts_instruction.append(tokenizer.decode(batch['input_ids'][b][0:(length-1)], skip_special_tokens=True))
    # content for remaining context positions
    txts_ctxt = []
    for j in range(batch_context['input_ids'].shape[0]):
        length_ctx = batch_context['input_ids'][j].shape[0]
        if j == 0:
            txt_ctxt = tokenizer.decode(batch_context['input_ids'][j][0:(length_ctx-1)])
        else:
            txt_ctxt = tokenizer.decode(batch_context['input_ids'][j][0:(length_ctx-1)], skip_special_tokens=True)
        txts_ctxt.append(txt_ctxt)
    # print('txts context:', txts_ctxt)
    # print('txt eval:', txt_eval)
    # print('txt:', txt)
    # setup full context

    # structure: <ex 1> \n <ex 2> \n ... <ex k> \n <ask ex 1>
    # instantiate appended post
    appended_post = txts_instruction[0] + ' ' + icl_unlearn_labels[0] + "\n"
    # loop over remaining forget points
    for b in range(1, batch_size): 
        appended_post += txts_instruction[b] + ' ' + icl_unlearn_labels[b] + "\n"
    for j in range(config.n_ctxt):
        if j == config.n_ctxt - 1:
            appended_post += txt_eval
        else:
            appended_post += txts_ctxt[j] + ' ' + icl_unlearn_labels[batch_size + j] + "\n" 
        # print(f'iteration {j} out of {config.n_ctxt}:', appended_post)
    return appended_post

def get_icl_only_forget_points(batch, batch_eval, batch_context, batch_label, icl_unlearn_labels, config):
    assert config.batch_sizes[0] == 1
    if batch_label == 'forget':
        batch_eval = batch
    length = batch['input_ids'][0].shape[0]
    length_eval = batch_eval['input_ids'][0].shape[0]

    # context for query position
    txt_eval = tokenizer.decode(batch_eval['input_ids'][0][0:(length_eval-1)], skip_special_tokens=True)

    # content for position 1
    txt = tokenizer.decode(batch['input_ids'][0][0:(length-1)], skip_special_tokens=True)
    
    appended_post = txt + ' ' + icl_unlearn_labels[0] + "\n" + txt_eval

    return appended_post


def get_icl_only_forget_points_batchsize_gg1(batch, batch_eval, batch_context, batch_label, icl_unlearn_labels, idx_to_evaluate, config):
    batch_size = batch['input_ids'].shape[0]
    if batch_label == 'forget':
        batch_eval = batch
    length = batch['input_ids'][0].shape[0]
    length_eval = batch_eval['input_ids'][0].shape[0]

    # context for query position
    txt_eval = tokenizer.decode(batch_eval['input_ids'][idx_to_evaluate][0:(length_eval-1)], skip_special_tokens=True)

    # content for first batch_size positions
    txts_instruction = []
    for b in range(batch_size):
        txts_instruction.append(tokenizer.decode(batch['input_ids'][b][0:(length-1)]), skip_special_tokens=True)    

    appended_post = txts_instruction[0] + ' ' + icl_unlearn_labels[0] + "\n"
    
    # loop over remaining forget points
    for b in range(1, batch_size): 
        appended_post += txts_instruction[b] + ' ' + icl_unlearn_labels[b] + "\n"
    
    appended_post += txt_eval
    return appended_post


def prepare_text(batch,
                 batch_label,
                 batch_other,
                 batch_context,
                 config,
                 n_subtract_tokens,
                 label,
                 icl_unlearn_label,
                 mode,
                 idx_to_evaluate:int=0,
                 verbose=False):

    ''' converts batch from dataloader into tokenized txt '''
    if mode == 'ICL':
        if config.batch_sizes[0] == 1 and config.n_ctxt > 1:
            if config.ctxt_style == "vary":
                appended_post = get_icl_vary_context(batch, batch_other, batch_context, batch_label, icl_unlearn_label, config)
            else:
                raise ValueError(f"Other context styles are currently not supported. Please choose >vary<.")
        elif config.batch_sizes[0] > 1 and config.n_ctxt > 1:
            if config.ctxt_style == "vary":
                appended_post = get_icl_vary_context_batchsize_gg1(batch, batch_other, batch_context, batch_label, icl_unlearn_label, idx_to_evaluate, config)
            else:
                raise ValueError(f"Other context styles are currently not supported. Please choose >vary<.")
        
        elif config.batch_sizes[0] == 1 and config.n_ctxt == 1 and config.ctxt_style == "only_forget_points":
            appended_post = get_icl_only_forget_points(batch, batch_other, batch_context, batch_label, icl_unlearn_label, config)
        elif config.batch_sizes[0] > 1 and config.n_ctxt == 1 and config.ctxt_style == "only_forget_points":
            appended_post = get_icl_only_forget_points_batchsize_gg1(batch, batch_other, batch_context, batch_label, icl_unlearn_label, idx_to_evaluate, config)
        
        else:
            raise ValueError(f"This configuration is not supported. You chose a batch size of {config.batch_sizes[0]} and n_ctxt of {config.n_ctxt}.")
    else:
        # use idx_to_evaluate-th point of batch for evaluation
        length = batch['input_ids'][idx_to_evaluate].shape[0]
        if batch_label == 'forget':
            appended_post = tokenizer.decode(batch['input_ids'][idx_to_evaluate][0:(length-n_subtract_tokens)])
        else:
            length_other = batch_other['input_ids'][idx_to_evaluate].shape[0]
            appended_post = tokenizer.decode(batch_other['input_ids'][idx_to_evaluate][0:(length_other-n_subtract_tokens)])
    if config.verbose:
        print('IDX to eval:', idx_to_evaluate)
        print('BATCH LABEL:', batch_label)
        print(f'CONTEXT FOR {mode}:', appended_post)
        print('------------------------------------')
    tokenized_text = tokenizer(appended_post, return_tensors="pt")
    return tokenized_text

def get_id(word: str):
    ''' returns the id corresponding to a word/str '''
    tokenized_text = tokenizer(word, return_tensors="pt")
    id = tokenized_text['input_ids'][0][0]
    return id

def compute_token_preds(out):
    pred_token = tokenizer.decode(out['sequences'][0][-1]).strip()
    return pred_token

def get_stable_logit_loss(out, label, ids, restricted_labels, label_mapping, eps=1e-35):
    '''
    Computing stable logit loss from Section VI (https://arxiv.org/abs/2112.03570)
    '''

    if label == 0:
        Y = ids[0].reshape(1,-1)
    elif label == 1:
        Y = ids[1].reshape(1,-1)
    elif label == 2:
        Y = ids[2].reshape(1,-1)
    elif label == 3:
        Y = ids[3].reshape(1,-1)
    else:
        raise ValueError(f"Label {label} is not supported. Please choose one of: 0, 1, 2, 3.")
    probs = torch.nn.functional.softmax(out['scores'][0][-1], dim=0).reshape(1,-1)
    # print('print prob at pred', probs)
    class_log_probs = torch.log(probs[torch.arange(probs.shape[0]), Y] + eps)  # compute log(f(x)_y)
    m, n = probs.shape
    del_ind = Y
    mask = torch.ones((m, n), dtype=bool)
    mask[range(m), del_ind] = False
    probs_complement = probs[mask].reshape(m, n-1)
    complement_class_sum = torch.sum(probs_complement, axis=1)                 # compute log(\sum_{y'} f(x)_{y'})
    score = class_log_probs - torch.log(complement_class_sum + eps)            # compute log(f(x)_y) - log(\sum_{y'} f(x)_{y'})
    # get class probs and complement class probs for analysis purposes
    # print('class log probs shape:', class_log_probs)
    # print('complement_class_sum shape:', complement_class_sum)
    return score[0][0].detach().cpu().numpy(), class_log_probs[0][0].detach().cpu().numpy(), torch.log(complement_class_sum + eps)[0].detach().cpu().numpy()

def setup_context_loader(train_dataset, n_ctxt):
    # first position is for point we want to unlearn; all other positions will be filled by samples from context_loader.
    context_loader = DataLoader(dataset=train_dataset,
                                batch_size=n_ctxt,
                                shuffle=True,
                                num_workers=2,
                                drop_last=True)
    return context_loader

def setup_dict(config, batch_size):
    results = {}
    methods = config.unlearning_methods + ['base']
    regimes = ['forget', 'test', 'train']
    for t in ['losses', 'nxt_token_preds']:
        for method in methods:
            for regime in regimes:
                for be in range(batch_size):
                    results[f'{t}_{method}_{regime}_{be}'] = []
    # for analysis purposes
    if 'icl' in methods:
        for regime in regimes:
            for be in range(batch_size):
                results[f'confs_icl_first_{regime}_{be}'] = []
                results[f'confs_icl_others_{regime}_{be}'] = []
    # to compute model performance
    for regime in regimes:
        for be in range(batch_size):
            results[f'labels_{regime}_{be}'] = []
    
    return results
    
def evals(k,
          model,
          config,
          train_dataset,
          forget_loader,
          train_loader,
          test_loader,
          n_subtract_tokens: int=1,
          n_additional: int=1,
          model_chckpt: str="finetuned_models/checkpoint-1547"):

    '''Evaluates the performance of various unlearning strategies:
       options are unlearn = {'icl', 'ga'} '''


    # infer batch size
    for idx, batch in enumerate(forget_loader):
        batch_size = batch['input_ids'].shape[0]
        if idx == 0:
            break
    # Setup dictionary to collect results
    results = setup_dict(config, batch_size)

    '''Get the correct label ids: The white space for these ones is IMPORTANT'''
    if config.dataset_name in ['sst2', 'amazon_polarity', 'yelp_polarity']:
        restricted_labels = ['positive', 'negative']
        id_pos = get_id(" positive")
        print('positive id:', {id_pos})
        id_neg = get_id(" negative")
        print('negative id:', {id_neg})

        ids = [id_pos, id_neg]

        label_mapping = {
            0: 'negative',
            1: 'positive'
        }

    elif config.dataset_name == 'ag_news':
        restricted_labels = ['World', 'Sports', 'Business', 'Science']
        id_0 = get_id(" World")
        print('World id:', {id_0})
        id_1 = get_id(" Sports")
        print('Sports id:', {id_1})
        id_2 = get_id(" Business")
        print('Business id:', {id_2})
        id_3 = get_id(" Science")
        print('Science id:', {id_3})

        ids = [id_0, id_1, id_2, id_3]

        label_mapping = {
            0: 'World',
            1: 'Sports',
            2: 'Business',
            3: 'Science'
        }
    
    else: 
        raise ValueError(f"Dataset {config.dataset_name} is not supported. Please choose one of: sst2, yelp_polarity, amazon_polarity, ag_news.")


    '''Setup loaders'''
    context_loader = setup_context_loader(train_dataset, config.n_ctxt)
    # print elements of the context loader
    it = iter(train_loader)
    jt = iter(test_loader)
    kt = iter(context_loader)


    '''Start (unlearning) eval loop'''
    print(f"Using the following label flipping strategey: {config.label_flipping_method}")
    print(f'Looping over {len(forget_loader)} samples...')
    for idx, batch in enumerate(forget_loader):
        '''Samples train / test points for evaluation'''
        batch_train = next(it)
        batch_test = next(jt)
        try:
            '''Samples the context batch'''
            batch_context = next(kt)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            context_loader = setup_context_loader(train_dataset, config.n_ctxt)
            kt = iter(context_loader)
            batch_context = next(kt)
        assert batch_size >= config.unlearn_batch_size
        if idx % 100 == 0:
            print(f'Evaluating sample: {idx}')
        
        ''' Get the right set of labels '''
        labels = []
        icl_unlearn_label = []
        ''' case: batch_size > 1 '''
        if batch_size > 1:
            if config.ctxt_style == "vary" or config.ctxt_style == "only_forget_points":
                ''' first batch_size context positions '''
                for b in range(batch_size):
                    correct_label = label_mapping[batch['label'][b].item()]
                    if correct_label in restricted_labels:
                        labels.append(correct_label)
                        random_flip = np.random.choice([x for x in restricted_labels if x != correct_label])
                        icl_unlearn_label.append(random_flip)
                    else:
                        raise ValueError(f"Label {batch['label'][b]} is not supported. Please choose one of: 0, 1, 2, 3.")
                
                if config.ctxt_style == "vary":
                    ''' remaining context positions '''
                    # append the correct labels for the remaining context positions
                    for j in range(config.n_ctxt-1):
                        correct_label = label_mapping[batch_context['label'][j].item()]
                        if correct_label in restricted_labels:
                            icl_unlearn_label.append(correct_label)
                        else:
                            raise ValueError(f"Label {batch_context['label'][j]} is not supported. Please choose one of: 0, 1, 2, 3.")

                # print(f'Labels for context style: {config.ctxt_style}')
                # print('icl unlearn labels:', icl_unlearn_label)
                # print('labels:', labels)
            
            else:
                raise ValueError(f"Other context styles are currently not supported. Please choose >vary<.")
        
        # case: batch_size == 1
        else:
            if config.ctxt_style == "vary" or config.ctxt_style == "only_forget_points":
                ''' first context position '''
                correct_label = label_mapping[batch['label'][0].item()]
                if correct_label in restricted_labels:
                    labels.append(correct_label)
                    random_flip = np.random.choice([x for x in restricted_labels if x != correct_label])
                    icl_unlearn_label.append(random_flip)
                else:
                    raise ValueError(f"Label {batch['label'][0]} is not supported. Please choose one of: 0, 1, 2, 3.")               
                
                if config.ctxt_style == "vary":
                    ''' remaining context positions '''
                    # append the correct labels for the remaining context positions
                    print('batch context label:', batch_context['label'])
                    for j in range(config.n_ctxt-1):

                        correct_label = label_mapping[batch_context['label'][j].item()]
                        if correct_label in restricted_labels:
                            icl_unlearn_label.append(correct_label)
                        else:
                            raise ValueError(f"Label {batch_context['label'][j]} is not supported. Please choose one of: 0, 1, 2, 3.")

                # print(f'Labels for context style: {config.ctxt_style}')
                # print('icl unlearn labels:', icl_unlearn_label)
                # print('labels:', labels)
            
            else:
                raise ValueError(f"Other context styles are currently not supported. Please choose either >vary< or >only_forget_points< .")

        print(f'Labels for context style: {config.ctxt_style}')
        print(f'icl unlearn labels: {icl_unlearn_label}')
        print(f'labels: {labels}')
        
        print('-----------------------------------------------------------')
        print('Counter at:', idx)

        ''' Baseline predictions '''
        print('Baseline predictions ...')
        batchers = [None, batch_test, batch_train]
        loader_labels = ['forget', 'test', 'train']
        for sample_idx in range(batch_size):
            for l_idx, batch_other in enumerate(batchers):
                batch_label = loader_labels[l_idx]
                tokenized_text_base = prepare_text(batch,
                                                   batch_label,
                                                   batch_other,
                                                   batch_context,
                                                   config,
                                                   n_subtract_tokens,
                                                   labels,
                                                   icl_unlearn_label,
                                                   verbose=config.verbose,
                                                   idx_to_evaluate=sample_idx,
                                                   mode='no')
                
                
                out_baseline = eval_model(model,
                                              tokenized_text_base,
                                              n_additional)

                if batch_label == 'forget':
                    lab = batch['label'][sample_idx]
                else:
                    lab = batch_other['label'][sample_idx]


                results[f'losses_base_{batch_label}_{sample_idx}'].append(get_stable_logit_loss(out_baseline,
                                                                                                lab,
                                                                                                ids,
                                                                                                restricted_labels,
                                                                                                label_mapping
                                                                                                )[0])


                results[f'nxt_token_preds_base_{batch_label}_{sample_idx}'].append(compute_token_preds(out_baseline))
                results[f'labels_{batch_label}_{sample_idx}'].append(lab)

                # print(f'sample {sample_idx} in batch {batch_label}')
                # print(f'len of tokenized text: {len(tokenized_text_base["input_ids"][0])}')
                # print(f'tokenized text for baseline: {tokenizer.decode(tokenized_text_base["input_ids"][0])}')
                # print(f'predicted token: {compute_token_preds(out_baseline)}')
                # print(f'ground truth label: {lab}')

                if compute_token_preds(out_baseline) not in restricted_labels:
                    print(f'PROBLEM: out_baseline is not in {restricted_labels}')
                    print(f'sample {sample_idx} in batch {batch_label}')
                    print(f'len of tokenized text: {len(tokenized_text_base["input_ids"][0])}')
                    print(f'tokenized text for baseline: {tokenizer.decode(tokenized_text_base["input_ids"][0], skip_special_tokens=True)}')
                    print(f'predicted token: {compute_token_preds(out_baseline)}')
                    print(f'ground truth label: {lab}')

        
        ''' In context unlearning evaluation '''
        if 'icl' in config.unlearning_methods:
            print('ICL unlearning ...')
            batchers = [None, batch_test, batch_train]
            batch_labels = ['forget', 'test', 'train']
            # loop over batch size elements
            for sample_idx in range(batch_size):
                # loop over forget test & train points
                for l_idx, batch_other in enumerate(batchers):
                    batch_label = batch_labels[l_idx]
                    # prepare tokens to get ICL unlearning predictions
                    tokenized_text = prepare_text(batch,
                                                  batch_label,
                                                  batch_other,
                                                  batch_context,
                                                  config,
                                                  n_subtract_tokens,
                                                  labels,
                                                  icl_unlearn_label,
                                                  verbose=config.verbose,
                                                  idx_to_evaluate=sample_idx,
                                                  mode='ICL')
                    out_icl = eval_model(model,
                                         tokenized_text,
                                         n_additional)

                    if batch_label == 'forget':
                        lab = batch['label'][sample_idx]
                    else:
                        lab = batch_other['label'][sample_idx]

                    results[f'nxt_token_preds_icl_{batch_label}_{sample_idx}'].append(compute_token_preds(out_icl))
                    results[f'losses_icl_{batch_label}_{sample_idx}'].append(get_stable_logit_loss(out_icl,
                                                                                                   lab,
                                                                                                    ids,
                                                                                                    restricted_labels,
                                                                                                    label_mapping)[0])                        
                    results[f'confs_icl_first_{batch_label}_{sample_idx}'].append(get_stable_logit_loss(out_icl,
                                                                                                        lab,
                                                                                                        ids,
                                                                                                        restricted_labels,
                                                                                                        label_mapping)[1])
                    results[f'confs_icl_others_{batch_label}_{sample_idx}'].append(get_stable_logit_loss(out_icl,
                                                                                                         lab,
                                                                                                         ids,
                                                                                                         restricted_labels,
                                                                                                         label_mapping)[2])
                    
                    # print(f'sample {sample_idx} in batch {batch_label}')
                    # print(f'len of tokenized text: {len(tokenized_text["input_ids"][0])}')
                    # print(f'tokenized text for icl: {tokenizer.decode(tokenized_text["input_ids"][0])}')
                    # print(f'predicted token: {results[f"nxt_token_preds_icl_{batch_label}_{sample_idx}"][-1]}')
                    # print(f'ground truth label: {lab}')

                    if compute_token_preds(out_icl) not in restricted_labels:
                        print(f'PROBLEM: out_icl is not in {restricted_labels}')
                        print(f'sample {sample_idx} in batch {batch_label}')
                        print(f'len of tokenized text: {len(tokenized_text["input_ids"][0])}')
                        print(f'tokenized text for icl: {tokenizer.decode(tokenized_text["input_ids"][0], skip_special_tokens=True)}')
                        print(f'predicted token: {results[f"nxt_token_preds_icl_{batch_label}_{sample_idx}"][-1]}')
                        print(f'ground truth label: {lab}')

        
            
        ''' For testing purposes '''
        if idx == config.n_samples:
            break
        
    ''' Save all results '''
    res = pd.DataFrame.from_dict(results)
    res.to_csv(f'./results/{config.dataset_name}/eval_skip_cancel_decode_ctxts/ubs{batch_size}/nctxt{config.n_ctxt}/results_{config.dataset_name}_{config.model_name}_model{k}_{config.unlearning_methods[0]}_n{config.n_samples}_mepochs{config.model_epochs}_uepochs{config.n_unlearn_epochs}_bs{batch_size}_{config.ctxt_style}_nctxt{config.n_ctxt}_lfm{config.label_flipping_method}.csv')

    return res



if __name__ == '__main__':
    # Parsing Arguments
    parser = ArgumentParser()
    parser.add_argument("--config", default=None, type=str, help="Config file.")
    parser.add_argument("--batch_sizes", default=None, type=int)
    parser.add_argument("--n_ctxt", default=None, type=int, help="Int: Length of context when batch_size=1.")
    parser.add_argument("--ctxt_style", default=None, type=str, help="Str: Context style must be >vary< or >only_forget_points<.")
    parser.add_argument("--K_models", default=None, type=int, help="Int: How many models to run the evaluation over. Should usually be 1 when running several evaluations in parallel. Make sure you understand this setup.")
    parser.add_argument("--rng_offset", default=None, type=int, help="Int: Number of run indicating which model will be used.")
    parser.add_argument("--lfm", default=None, type=str, help="Str: One of >first-k<, >last-k<, or >flipp_all<.")
    parser.add_argument("--model_path", default=None, type=str, help="Str: One of: >gpt2<, >gpt2-medium< or >bigscience/bloom-560m<.")
    parser.add_argument('--dataset_name', default=None, type=str, help="Str: Whichh dataset to use for the evaluation. Options are >sst2<, >yelp_polarity< or >amazon_polarity<")
    parser.add_argument('--lr', default=5e-5, type=float, help="Float: Learning rate for gradient ascent unlearning.")
    
    arg_ = parser.parse_args()
    if arg_.config is None:
        raise NameError("Include a >config< file in the argument please.")
    if arg_.batch_sizes is None:
        raise NameError("Include a >batch< size in the argument please.")
    if arg_.ctxt_style is None:
        raise NameError("Include a >ctxt_style< in the argument please. One of >vary< or >only_forget_points<.")
    if arg_.K_models is None:
        raise NameError("Include >K_models< in the argument please.")
    if arg_.lr is None:
        raise NameError("Please include >lr< in the argument please.")
    if arg_.rng_offset is None:
        raise NameError("Include >rng_offset< in the argument please.")
    if arg_.lfm is None:
        raise NameError("Include >lfm< in the argument please. One of >first-k<, >last-k<, >random< or >flip_all<.")
    if arg_.model_path is None:
        raise NameError("Include a >model_path< in the argument please: Onf of >gpt2<, >gpt2-medium<, >bigscience/bloom-560m< or >bigscience/bloom-1b1<.")
    if arg_.dataset_name is None:
        raise NameError("Include a >dataset_name< in the argument please. One of >sst2<, >yelp_polarity< or >amazon_polarity<.")

    # Getting configurations
    config_path = arg_.config
    with open(config_path) as config_file:
        config = json.load(config_file)
    config = argparse.Namespace(**config)

    if "model_epochs" not in config:
        config.model_epochs = 1
    if "unlearning_methods" not in config:
        config.unlearning_methods = ["icl"]
    if "n_samples" not in config:
        config.n_samples = 12500
    if "n_unlearn_epochs" not in config:
        config.n_unlearn_epochs = 1
    if "verbose" not in config:
        config.verbose = False
    if 'path_prefix_model' not in config:
        config.path_prefix_model = 'path/to/model'
    if 'path_prefix_data' not in config:
        config.path_prefix_data = 'path/to/data'
        
    assert len(config.unlearning_methods) == 1

    config.ga_in_memory = False
    config.label_flipping_method = arg_.lfm
    config.K_models = arg_.K_models
    config.rng_offset = arg_.rng_offset
    config.batch_sizes = [arg_.batch_sizes]
    config.model_name_or_path = arg_.model_path
    config.lr = float(arg_.lr)
    print(f'Learning rate: {config.lr}')
    config.n_ctxt = arg_.n_ctxt
    config.ctxt_style = arg_.ctxt_style
    if 'bloom-560m' in config.model_name_or_path:
    	config.model_name = 'bloom-560m'
    elif 'bloom-1b1' in config.model_name_or_path:
        config.model_name = 'bloom-1b1'
    elif 'OLMo-2-0425-1B' in config.model_name_or_path:
        config.model_name = 'OLMo-2-0425-1B'
    else:
        config.model_name = config.model_name_or_path
    config.ctxt_style = arg_.ctxt_style
    config.dataset_name = arg_.dataset_name

    # load cleaned data set    
    ''' eval loop over shadow models with index >config.rng_offset + k< when K_models=1 '''
    for k in range(config.K_models):
        if config.rng_offset > 0:
            k = config.rng_offset + k
        print(f'Evaluation for model: {k} ...')
        for batch_size in config.batch_sizes:
            # select correct in / out splits
            ind_mask_k = pd.read_csv(f"finetuned_models/{config.dataset_name}_indices_epochs{config.model_epochs}_unlearnbs{batch_size}_kmodel{k}.csv", 
                                     index_col=False).to_numpy()
            
            dataset = get_dataset(config, ind_mask_k.reshape(-1))
                                   
            # get the right tokenizer
            if 'bloom-560m' in config.model_name_or_path or 'bloom-1b1' in config.model_name_or_path:
                tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path,
                                                          max_length=config.max_length, )
            elif 'OLMo-2-0425-1B' in config.model_name_or_path:
                tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path,
                                                            max_length=config.max_length, 
                                                            padding_side='left')
                # tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path,
                #                                           max_length=config.max_length,
                #                                           padding_side='left')
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer = GPT2Tokenizer.from_pretrained(config.model_name_or_path,
                                                          max_length=config.max_length,
                                                          padding_side='left')
                tokenizer.pad_token = '<pad>'
            
            print("Tokenizer input max length:", tokenizer.model_max_length, flush=True)
            print("Tokenizer vocabulary size:", tokenizer.vocab_size, flush=True)
            print("Tokenizing ...", flush=True)
            tokenized = dataset.map(tokenize_text, batched=True, batch_size=None)
            tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            del dataset
            
            train_dataset = DS(tokenized, partition_key="train")
            test_dataset = DS(tokenized, partition_key="test")
    
            # make sure we load the correct model    
            print('Loading model from checkpoint ...')
            if config.path_prefix_model is None:
                model_chckpt = f"finetuned_models/{config.dataset_name}_epochs{config.model_epochs}_unlearnbs{batch_size}_kmodel{k}_{config.model_name}"
            else:
                model_chckpt = f"{config.path_prefix_model}/finetuned_models/{config.dataset_name}_epochs{config.model_epochs}_unlearnbs{batch_size}_kmodel{k}_{config.model_name}"
            
            if 'bloom-560m' in config.model_name_or_path or 'bloom-1b1' in config.model_name_or_path:
                model = BloomForCausalLM.from_pretrained(model_chckpt)
            elif 'OLMo-2-0425-1B' in config.model_name_or_path:
                model = AutoModelForCausalLM.from_pretrained(model_chckpt)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_chckpt)
            print(f'Batch_size: {batch_size} - n_unlearn_epoch: {config.n_unlearn_epochs}')
    
            forget_loader = DataLoader(
                dataset=train_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
                drop_last=True)
    
            train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2,
                drop_last=True)
    
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
                drop_last=True)
    
            results_unlearn = evals(k,
                                    model,
                                    config,
                                    train_dataset,
                                    forget_loader,
                                    train_loader,
                                    test_loader,
                                    n_subtract_tokens=1,
                                    n_additional=1,
                                    model_chckpt=model_chckpt)