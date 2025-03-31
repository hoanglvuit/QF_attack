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
import base64
from torch import autocast
import base64
from openai import OpenAI
from google import genai
from google.genai import types
import re
import os


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

def get_char_table(c = None):
    char_table=['·','~','!','@','#','$','%','^','*','(',')','=','-','*','.','<','>','?',',','\'',';',':','|','\\','/']
    for i in range(ord('a'),ord('z')+1):
        char_table.append(chr(i))
    for i in range(0,10):
        char_table.append(str(i))
    if c != None : 
        char_table = [i for i in char_table if i not in [c]]
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

def search(target_embedding=None,sentence=None, pool=None, mask=None, tokenizer=None, text_encoder=None,num = None, pro=None,optimize_max=True):
    pool_score = []
    print(num,pro)
    query_time = 0
    for candidate in pool : 
        pertub_sentence = sentence + ' ' + candidate 
        query_time += 1 
        if mask == None : 
            temp_score = cos_embedding_text(target_embedding, pertub_sentence, tokenizer=tokenizer, text_encoder=text_encoder)
        else : 
            temp_score = cos_embedding_text(target_embedding, pertub_sentence, mask, tokenizer=tokenizer, text_encoder=text_encoder)
        pool_score.append((temp_score,candidate)) 
    if optimize_max == True :
      sorted_pool = sorted(pool_score,reverse=True)
    else : 
      sorted_pool = sorted(pool_score,reverse=False)
    candidates = [x[1] for x in sorted_pool]
    scores = [x[0] for x in sorted_pool] 
    if pro != None : 
        top_candidates,top_scores = select_top_candidates_softmax(candidates[:num*4], scores[:num*4], k=num, temperature=pro)
    else : 
        top_candidates = candidates[:num]
        top_scores = scores[:num]
    return top_candidates,top_scores,query_time
def beamsearch(target_sentence = None,sentence=None, char_list=None, length=None,  mask=None, tokenizer=None, text_encoder=None, num = [140,140,140,3], pro = [None,None, None,None],optimize_max=True):
    query_time = 1
    scores = None
    sentence_embedding = get_text_embeds_without_uncond([target_sentence], tokenizer, text_encoder)
    candidates = char_list 
    iteration = length - 1 
    for i in range(iteration) : 
        pool = []
        for candidate in candidates : 
            for char in char_list : 
                pool.append(candidate + char) 
        candidates, scores,times = search(sentence_embedding,sentence,pool,mask,tokenizer,text_encoder,num[i],pro[i],optimize_max)
        query_time += times
    
    return list(zip(candidates,scores)) , query_time
        
# genetic algorithm
def tournament_selection(pool_score, optimize_max = True) : 
    pool = []
    if  optimize_max == True :
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
    cross_loc = random.randint(0, len(string1)-1)
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
    vari_char = random.randint(0, len(char_list)-1)
    string1_list = list(string1)
    string2_list = list(string2)
    string1_list[vari_loc] = char_list[vari_char]
    string2_list[vari_loc] = char_list[vari_char]
    string1 = ''.join(string1_list)
    string2 = ''.join(string2_list)
    return string1, string2

def POPOP(target_sentence = None,sentence=None, char_list=None, length=None, generation_num = 50, generateion_scale = 100, mask=None, tokenizer=None, text_encoder=None,tournament = True, optimize_max = True):
    generation_list = init_pool(char_list, length,generateion_scale)
    query_time = 0
    res = []
    score_list={}
    for _ in range(generation_num):
        pool = []
        indices = np.arange(generateion_scale)
        random.shuffle(indices)
        for i in range(0,len(generation_list),2) : 
            candidate = generation_list[indices[i]] 
            mate = generation_list[indices[i+1]]
            g1, g2 = get_generation(candidate, mate, char_list)
            pool.append(g1)
            pool.append(g2)  
            pool.append(candidate)
            pool.append(mate)

        generation_list, times  = select(target_sentence = target_sentence,sentence=sentence, pool=pool, generateion_scale =generateion_scale , score_list=score_list, mask=mask, tokenizer=tokenizer, text_encoder=text_encoder,tournament =tournament,optimize_max= optimize_max)
        query_time += times

    if target_sentence != None : 
        res = sorted(score_list.items(),key = lambda x:x[1],reverse = True)[0:3]
    else : 
        res = sorted(score_list.items(),key = lambda x:x[1],reverse = False)[0:3]
    print(query_time)
    return res,query_time
def select(target_sentence = None,sentence=None, pool=None, generateion_scale=None, mask=None, score_list=None, tokenizer=None, text_encoder=None,tournament = False,optimize_max=True):
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
        return tournament_selection(pool_score,optimize_max), query_time
    # Have guide ? 
    if optimize_max == True : 
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
# def object_key_2(sentence_list, object_word, thres = 10, tokenizer=None, text_encoder=None,use_avr = False):
#     extra_words=object_word
#     diff_list=[]
#     total_diff=0
#     for i in sentence_list:
#         sen_embed = get_text_embeds_without_uncond(i, tokenizer=tokenizer, text_encoder=text_encoder)
#         crafted_embed = get_text_embeds_without_uncond(extra_words, tokenizer=tokenizer, text_encoder=text_encoder)
#         diff_list.append(crafted_embed-sen_embed)
#         total_diff += crafted_embed-sen_embed
#     average_diff = total_diff/len(diff_list)

#     total_sign=0
#     for vec in diff_list:
#         vec[vec>0]=1
#         vec[vec<0]=-1
#         total_sign+=vec
#     total_sign[abs(total_sign)<=thres]=0
#     total_sign[abs(total_sign)>thres]=1
#     average_diff[abs(average_diff)<=thres]=0
#     average_diff[abs(average_diff)>thres]=1
#     if use_avr :
#         total_sign = average_diff
#     print('Ratio of mask', total_sign[total_sign>0].shape[0]/total_sign.view(-1).shape[0])
#     return total_sign



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
def generate_images(prompts,pipe,generator,num_image = 10) : 
    torch.cuda.empty_cache()
    gc.collect()
    images = [] 
    for prompt in prompts : 
        with autocast('cuda') : 
            image = pipe([prompt],generator = generator,num_inference_steps=50,num_images_per_prompt = num_image).images
            images.append(image)
    return images
def CLIP_score(folder, prompt, model, preprocess, tokenizer):
    image_folder = folder
    image_files = [f for f in os.listdir(image_folder) if f.endswith(".png")]
    
    # Di chuyển mọi xử lý lên GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # Xử lý hình ảnh và đưa lên GPU ngay lập tức
    images = torch.stack([preprocess(Image.open(os.path.join(image_folder, img))) for img in image_files])
    images = images.to(device)
    
    # Tokenize prompt và đưa lên GPU
    text = tokenizer([prompt]).to(device) 
    
    # Encode images và text
    with torch.no_grad(), torch.autocast(device):
        image_features = model.encode_image(images)
        text_features = model.encode_text(text)
        
        # Chuẩn hóa các đặc trưng trên GPU
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Tính toán điểm tương đồng trên GPU rồi mới chuyển về CPU
        similarity = (text_features @ image_features.T).cpu().numpy()
    

    return np.mean(similarity)

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
    char_collection = list(set(key_word)) 
    min_score = 1
    min_word = ''
    for char in char_collection :
        s_word = key_word.replace(char,'')
        score = compare_sentences(key_word,s_word,None,tokenizer,text_encoder)
        if score < min_score : 
            min_word = char
            min_score = score
    print('word: ',key_word,'min_word: ',min_word,'min_score: ',min_score) 
    return min_word
# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def SR_evaluation(obj_1, obj_2, folder, client, batch_size=30):
    """
    obj_1, obj_2 : string 
    folder: chứa ảnh .png
    client: OpenAI API client
    batch_size: số lượng ảnh gửi mỗi lần để tránh vượt giới hạn
    """
    image_files = [f for f in os.listdir(folder) if f.endswith(".png")]
    image_files = [os.path.join(folder, img) for img in image_files]
    
    total_count = 0  # Biến để cộng dồn kết quả

    ins = f"How many images contain both {obj_1} and {obj_2}? Respond with only a number."
    print(ins)
    num_batches = math.ceil(len(image_files) / batch_size)

    for batch_idx in range(num_batches):
        batch_images = image_files[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        
        messages = [
            {"role": "system", "content": "Only respond with a single integer, no text."},
            {"role": "user", "content": [{"type": "text", "text": ins}]}
        ]
        
        for image_path in batch_images:
            base64_image = encode_image(image_path)
            messages[1]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}", "detail": "low"},
            })

        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0,
            top_p=0,
        )

        batch_count = int(completion.choices[0].message.content.strip())
        total_count += batch_count  # Cộng dồn kết quả từng batch

    return total_count/len(image_files)

def generate_sentences(object_1,object_2,gemini_key) : 
    client_gemini = genai.Client(api_key=gemini_key)
    prompt = f"Generate 50 sentences for text-to-image that have an simple object (only 1 word such as '{object_1}') and  'and {object_2}' at the end"
    print(prompt)
    response = client_gemini.models.generate_content(
        model="gemini-2.0-flash", contents=prompt,
        config=types.GenerateContentConfig(
        temperature=0,
        topP = 1,
        topK=1,      
        )
    )
    sentence_list =  [s.rstrip('.') for s in re.findall(r'\d+\.\s+(.*)', response.text)]
    extra_key = " and " + object_2 
    return sentence_list,extra_key 

def find_mask(target_sentence,sentence,sentence_list,extra_key,tokenizer,text_encoder,start = 1.1,end = 2) : 
    mask = None 
    score = None 
    thresholds = np.arange(start,end,0.05) 
    for thres in thresholds : 
        mask= object_key_1(sentence_list, extra_key, thres=thres, tokenizer=tokenizer, text_encoder=text_encoder,use_avr = True)
        mask_list = mask.tolist()
        mask = mask.view(-1)
        print(np.sum(mask_list))
        result = compare_sentences(target_sentence,sentence,mask,tokenizer,text_encoder) 
        if result < 0.4  or np.sum(mask_list) < 3000: 
            score = result 
            print(thres)
            break       
    return mask,score





# import numpy as np
# from Bio import pairwise2
# from sklearn.manifold import MDS
# from sklearn.cluster import KMeans
# from difflib import SequenceMatcher


# def longest_common_substring(s1, s2):
#     matcher = SequenceMatcher(None, s1, s2)
#     match = matcher.find_longest_match(0, len(s1), 0, len(s2))
#     return len(s1[match.a: match.a + match.size])

# # Hàm tính khoảng cách dựa trên global alignment
# def compute_distance(seq1, seq2):
#     alignment = longest_common_substring(seq1,seq2)
#     max_score = len(seq1) * 1 
#     distance = max_score - alignment 
#     return distance

# def k_mean(pool_score,n=5,k=20,num=0) : 
#     strings = [can[1] for can in pool_score]
#     scores = [can[0] for can in pool_score]
#     B = []
#     n = len(strings)
#     distance_matrix = np.zeros((n, n))
    
#     for i in range(n):
#         for j in range(i, n):
#             distance = compute_distance(strings[i], strings[j])
#             distance_matrix[i, j] = distance
#             distance_matrix[j, i] = distance
    
#     # Giảm chiều dữ liệu với MDS
#     mds = MDS(n_components=n, dissimilarity="precomputed", random_state=42)
#     coordinates = mds.fit_transform(distance_matrix)
    
#     # Phân cụm bằng K-means
#     kmeans = KMeans(n_clusters=k) 
#     clusters = kmeans.fit_predict(coordinates)
#     A = []
#     for i in range(k) : 
#         A+= list(np.array(strings)[clusters == i][:int(num/k)])
#         B+= list(np.array(scores)[clusters == i][:int(num/k)])
#     return A ,B
    
    