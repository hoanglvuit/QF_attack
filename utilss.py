import torch
import copy
import random
from torch.nn import functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
import gc 
from open_clip import tokenizer
from torch import autocast
import base64
from openai import OpenAI

random.seed(28)
torch.manual_seed(28)
cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
def get_text_embeds_without_uncond(prompt, tokenizer, text_encoder):
    # Tokenize text and get embeddings
    text_input = tokenizer(
      prompt, padding='max_length', max_length=tokenizer.model_max_length,
      truncation=True, return_tensors='pt')
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.cuda())[0]
        # text_embeddings = text_encoder(text_input.input_ids)[0]
    return text_embeddings

def get_char_table():
    char_table=['·','~','!','@','#','$','%','^','&','*','(',')','=','-','*','+','.','<','>','?',',','\'',';',':','|','\\','/']
    for i in range(ord('a'),ord('z')+1):
        char_table.append(chr(i))
    for i in range(0,10):
        char_table.append(str(i))
    return char_table

# greedy algorithm
def cos_mask(a,b,mask):
    return cos(a*mask, b*mask)

def select_top_candidates_softmax(candidates, scores, k=None, temperature=None):
    scores = np.array(scores)
    exp_scores = np.exp(scores / temperature)  
    probabilities = exp_scores / np.sum(exp_scores) 
    
    selected_indices = np.random.choice(len(candidates), size=k, replace=False, p=probabilities)
    selected_candidates = [candidates[i] for i in selected_indices]
    selected_scores = [scores[i] for i in selected_indices]
    
    return selected_candidates,selected_scores

def search(target_embedding=None, target_sentence = None,sentence=None, pool=None, mask=None, tokenizer=None, text_encoder=None,num = None, pro=None):
    pool_score = []
    query_time = 0
    for candidate in pool : 
        pertub_sentence = sentence + ' ' + candidate 
        query_time += 1 
        if mask == None : 
            temp_score = cos_embedding_text(target_embedding, pertub_sentence, tokenizer=tokenizer, text_encoder=text_encoder)
        else : 
            temp_score = cos_embedding_text(target_embedding, pertub_sentence, mask, tokenizer=tokenizer, text_encoder=text_encoder)
        pool_score.append((temp_score,candidate)) 
    if target_sentence :
      sorted_pool = sorted(pool_score,reverse=True)
    else : 
      sorted_pool = sorted(pool_score,reverse=False)
    candidates = [x[1] for x in sorted_pool]
    scores = [x[0] for x in sorted_pool] 
    if pro != None : 
        top_candidates,top_scores = select_top_candidates_softmax(candidates, scores, k=num, temperature=pro)
    else : 
        top_candidates = candidates[:num]
        top_scores = scores[:num]
    return top_candidates,top_scores,query_time
def beamsearch(target_sentence = None,sentence=None, char_list=None, length=None,  mask=None, tokenizer=None, text_encoder=None, num = [500,50,10,3], pro = [0.03,0.01, 0.005,None]):
    query_time = 1
    scores = None
    if target_sentence != None : 
        sentence_embedding = get_text_embeds_without_uncond([target_sentence], tokenizer, text_encoder)
    else : 
        sentence_embedding = get_text_embeds_without_uncond([sentence], tokenizer, text_encoder)
    candidates = char_list 
    iteration = length - 1 
    for i in range(iteration) : 
        pool = []
        for candidate in candidates : 
            for char in char_list : 
                pool.append(candidate + char) 
        if target_sentence != None : 
          candidates, scores,times = search(sentence_embedding,target_sentence,sentence,pool,mask,tokenizer,text_encoder,num[i],pro[i])
        else : 
          candidates, scores,times = search(sentence_embedding,sentence,pool,mask,tokenizer,text_encoder,num[i])
        query_time += times
    print(query_time)
    print(scores)
    return candidates
        
# genetic algorithm
def tournament_selection(pool_score, flag = None) : 
    pool = []
    if  flag != None :
        for _ in range(2) : 
            random.shuffle(pool_score) 
            for i in range(0,len(pool_score),4) : 
                sub_pool = pool_score[i:i+4] 
                sorted_sub_pool = sorted(sub_pool,reverse=True) 
                pool.append(sorted_sub_pool[0][1])
    else : 
        for _ in range(2) : 
            random.shuffle(pool_score) 
            for i in range(0,len(pool_score),4) : 
                sub_pool = pool_score[i:i+4] 
                sorted_sub_pool = sorted(sub_pool) 
                pool.append(sorted_sub_pool[0][1])
    return pool 
def get_generation(string1, string2, char_list):
    if len(string1) != len(string2):
        print("length of string1 and string2 should be the same")
        return None
    string1, string2 = cross_generation(string1, string2)
    string1, string2 = vari_generation(string1, string2, char_list)
    return string1, string2
    
def cross_generation(string1, string2):
    cross_loc = random.randint(1, len(string1)-1)
    string1_seg1 = string1[0:cross_loc]
    string2_seg1 = string2[0:cross_loc]
    string1_list = list(string1)
    string2_list = list(string2)
    for i in range(len(string1_seg1)):
        string1_list[i] = string2_seg1[i]
        string2_list[i] = string1_seg1[i]
    string1 = ''.join(string1_list)
    string2 = ''.join(string2_list)
    return string1, string2

def vari_generation(string1, string2, char_list):
    vari_loc = random.randint(0, len(string1)-1)
    vari_char = random.randint(0,len(char_list)-1)
    string1_list = list(string1)
    string2_list = list(string2)
    string1_list[vari_loc] = char_list[vari_char]
    string2_list[vari_loc] = char_list[vari_char]
    string1 = ''.join(string1_list)
    string2 = ''.join(string2_list)
    return string1, string2

def POPOP(target_sentence = None,sentence=None, char_list=None, length=None, generation_num = 50, generateion_scale = 100, mask=None, tokenizer=None, text_encoder=None,tournament = False):
    generation_list = init_pool(char_list, length,generateion_scale)
    query_time = 0
    res = []
    score_list={}
    for _ in range(generation_num):
        pool = []
        indices = np.arange(generateion_scale)
        np.random.shuffle(indices)
        for i in range(0,len(generation_list),2) : 
            candidate = generation_list[indices[i]] 
            mate = generation_list[indices[i+1]]
            g1, g2 = get_generation(candidate, mate, char_list)
            pool.append(g1)
            pool.append(g2)  
            pool.append(candidate)
            pool.append(mate)

        generation_list, times  = select(target_sentence = target_sentence,sentence=sentence, pool=pool, generateion_scale =generateion_scale , score_list=score_list, mask=mask, tokenizer=tokenizer, text_encoder=text_encoder,tournament =tournament)
        query_time += times

    if target_sentence != None : 
        res = sorted(score_list.items(),key = lambda x:x[1],reverse = True)[0:3]
    else : 
        res = sorted(score_list.items(),key = lambda x:x[1],reverse = False)[0:3]
    print(query_time)
    return res
def select(target_sentence = None,sentence=None, pool=None, generateion_scale=None, mask=None, score_list=None, tokenizer=None, text_encoder=None,tournament = False):
    query_time = 1
    if target_sentence : 
        text_embedding = get_text_embeds_without_uncond([target_sentence], tokenizer, text_encoder)
    else : 
        text_embedding = get_text_embeds_without_uncond([sentence], tokenizer, text_encoder)
    pool_score = []
    if score_list == None:
        score_list = {}
    for candidate in pool:
        if candidate in score_list.keys():
            temp_score = score_list[candidate]
            pool_score.append((temp_score, candidate))
            continue
        query_time += 1 
        candidate_text = sentence + ' ' + candidate
        if mask == None:
            temp_score = cos_embedding_text(text_embedding, candidate_text, tokenizer=tokenizer, text_encoder=text_encoder)
        else:
            temp_score = cos_embedding_text(text_embedding, candidate_text, mask, tokenizer=tokenizer, text_encoder=text_encoder)
        score_list[candidate]=temp_score
        # print('genetic prompt:',candidate,temp_score)
        pool_score.append((temp_score, candidate))

    # tournament selection 
    if tournament == True : 
        return tournament_selection(pool_score,target_sentence), query_time
    # Have guide ? 
    if target_sentence != None : 
        sorted_pool = sorted(pool_score,reverse=True) 
    else : 
        sorted_pool = sorted(pool_score)
    
    selected_generation = [x[1] for x in sorted_pool]
    return selected_generation[0:generateion_scale], query_time

def cos_mask(a,b,mask):
    return cos(a*mask, b*mask)
    
def cos_embedding_text(embading, text, mask=None, tokenizer=None, text_encoder=None):    
    change_embading = get_text_embeds_without_uncond([text], tokenizer, text_encoder)
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    if mask==None:
        return cos(embading.view(-1), change_embading.view(-1)).item()
    else:
        return cos(embading.view(-1)*mask, change_embading.view(-1)*mask).item()
    
def init_pool(char_list, length, num = 10):
    pool=[]
    for i in range(num):
        pool.append(''.join(random.sample(char_list, length)))
    return pool


def object_key_1(sentence_list, object_word, thres = 10, tokenizer=None, text_encoder=None, use_avr = False):
    extra_words=object_word
    diff_list=[]
    total_diff=0
    for i in sentence_list:
        sen_embed = get_text_embeds_without_uncond(i, tokenizer=tokenizer, text_encoder=text_encoder)
        crafted_embed = get_text_embeds_without_uncond(i.replace(object_word,''), tokenizer=tokenizer, text_encoder=text_encoder)
        diff_list.append(crafted_embed-sen_embed)
        total_diff += crafted_embed-sen_embed
    average_diff = total_diff/len(diff_list)

    total_sign=0
    for vec in diff_list:
        vec[vec>0]=1
        vec[vec<0]=-1
        total_sign+=vec
    total_sign[abs(total_sign)<=thres]=0
    total_sign[abs(total_sign)>thres]=1
    average_diff[abs(average_diff)<=thres]=0
    average_diff[abs(average_diff)>thres]=1
    if use_avr : 
        total_sign = average_diff
    print('Ratio of mask', total_sign[total_sign>0].shape[0]/total_sign.view(-1).shape[0])
    return total_sign
def object_key_2(sentence_list, object_word, thres = 10, tokenizer=None, text_encoder=None,use_avr = False):
    extra_words=object_word
    diff_list=[]
    total_diff=0
    for i in sentence_list:
        sen_embed = get_text_embeds_without_uncond(i, tokenizer=tokenizer, text_encoder=text_encoder)
        crafted_embed = get_text_embeds_without_uncond(extra_words, tokenizer=tokenizer, text_encoder=text_encoder)
        diff_list.append(crafted_embed-sen_embed)
        total_diff += crafted_embed-sen_embed
    average_diff = total_diff/len(diff_list)

    total_sign=0
    for vec in diff_list:
        vec[vec>0]=1
        vec[vec<0]=-1
        total_sign+=vec
    total_sign[abs(total_sign)<=thres]=0
    total_sign[abs(total_sign)>thres]=1
    average_diff[abs(average_diff)<=thres]=0
    average_diff[abs(average_diff)>thres]=1
    if use_avr :
        total_sign = average_diff
    print('Ratio of mask', total_sign[total_sign>0].shape[0]/total_sign.view(-1).shape[0])
    return total_sign


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid
def show_image_groups_with_prompts(image_groups, prompts, score_list, figsize=(25, 100)):
    """
    Display groups of images with their corresponding prompts and scores.
    Optimized for large image groups (around 50 images per group).
    
    Args:
        image_groups: List of lists containing images
        prompts: List of prompts corresponding to each group
        score_list: List of scores for each group
        figsize: Tuple of (width, height) for the figure
    """
    # Number of groups and images per group
    num_groups = len(image_groups)
    images_per_group = len(image_groups[0])
    
    # Calculate optimal grid layout
    images_per_row = min(10, images_per_group)  # Cap at 10 images per row
    rows_per_group = math.ceil(images_per_group / images_per_row)
    total_rows = rows_per_group * num_groups
    
    # Adjust figsize based on the number of rows and columns
    adjusted_height = max(15, 5 * total_rows)  # Minimum height of 15
    adjusted_width = max(20, 2.5 * images_per_row)  # Minimum width of 20
    fig = plt.figure(figsize=(adjusted_width, adjusted_height))
    
    # Create grid for all images
    grid = plt.GridSpec(total_rows, images_per_row, hspace=0.4, wspace=0.2)
    
    for group_idx, (group, prompt, score) in enumerate(zip(image_groups, prompts, score_list)):
        # Calculate row offset for current group
        row_offset = group_idx * rows_per_group
        
        # Add prompt and score as a title for the group
        prompt_text = f"Group {group_idx + 1}\nPrompt: {prompt}\nScore: {score}"
        plt.figtext(0.02, 1 - (row_offset + rows_per_group/2) / total_rows,
                   prompt_text,
                   verticalalignment='center',
                   fontsize=10,
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', pad=3),
                   wrap=True)
        
        # Plot images in the group
        for img_idx, img in enumerate(group):
            # Calculate position in grid
            current_row = row_offset + (img_idx // images_per_row)
            current_col = img_idx % images_per_row
            
            # Create subplot and display image
            ax = plt.subplot(grid[current_row, current_col])
            ax.imshow(img)
            ax.axis('off')
            
            # Add small image number
            ax.text(0.02, 0.98, f'{img_idx + 1}',
                   transform=ax.transAxes,
                   fontsize=8,
                   color='white',
                   bbox=dict(facecolor='black', alpha=0.5, pad=1),
                   verticalalignment='top')
    
    # Adjust layout
    plt.subplots_adjust(left=0.2, right=0.98, top=0.98, bottom=0.02)
    plt.show()
def generate_images(prompts,pipe,generator) : 
    torch.cuda.empty_cache()
    gc.collect()
    images = [] 
    for prompt in prompts : 
        with autocast('cuda') : 
            image = pipe([prompt],generator = generator,num_inference_steps=50,num_images_per_prompt = 10).images
            images.append(image)
    return images
def clip_score(prompts,images): 
    simi = 0 
    score_list = []
    for i in range(len(prompts)): 
        image_input = torch.tensor(np.stack([preprocess(img) for img in images[i]])).to(device)
        text_tokens = tokenizer.tokenize([prompts[i]]*5).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_input).float()
            text_features = model.encode_text(text_tokens).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
        similarity = np.mean(similarity)
        simi += similarity 
        score_list.append(similarity)
    return simi / len(prompts) , score_list

def compare_sentences(sentence1,sentence2,mask=None,tokenizer=None,text_encoder=None) : 
    text_embedding1 = get_text_embeds_without_uncond([sentence1], tokenizer, text_encoder)
    text_embedding2 = get_text_embeds_without_uncond([sentence2], tokenizer, text_encoder)
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    if mask != None : 
        result = cos(text_embedding1.view(-1) *mask, text_embedding2.view(-1)*mask).item()
    else : 
        result = cos(text_embedding1.view(-1) , text_embedding2.view(-1)).item()
    return result
def get_min_character(key_word,tokenizer,text_encoder) : 
    char_collection = list(key_word) 
    min_score = 1
    min_word = ''
    for ind,char in enumerate(char_collection) :
        s_word = ''.join(char_collection[:ind] + char_collection[ind+1:]) 
        score = compare_sentences(key_word,s_word,None,tokenizer,text_encoder)
        if score < min_score : 
            min_word = char
            min_score = score
    return min_word

