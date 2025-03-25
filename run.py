from diffusers import StableDiffusionPipeline
from torch import autocast
from utilss import *
import numpy as np
import torch
from transformers import CLIPTextModel, CLIPTokenizer
import argparse
from google import genai
from google.genai import types
import re
import torch
if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Define hyperparameters.')
    parser.add_argument('--model', type=str, default='sd1.5', help='Model name:sd1.5;sd2.1;sd3.5,sdxl')
    parser.add_argument('--oo', type=str, default='a cat', help='Original Object')
    parser.add_argument('--to', type=str, default='a book', help='Target Object')
    parser.add_argument('--objective',type=str,default='maxte',help='objective name: maxte,maxnote') 
    parser.add_argument('--algorithm',type=str,default='popop',help='popop,beamsearch')
    parser.add_argument('--ge_key',type=str,default='****',help='gemini_key')
    parser.add_argument('--save_path',type=str,help='image folder') 
args = parser.parse_args()


# Load clip
len_prompt = 5
tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
text_encoder = CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14')
text_encoder = text_encoder.to('cuda')

# create sentence 
sentence = args.oo 
target_sentence = args.oo + ' and ' + args.to 
print(sentence)
print(target_sentence)

# get search space 
c = get_min_character(args.to,tokenizer,text_encoder)
char_table = get_char_table(c) 

# ori score 1 
oris_1 = compare_sentences(target_sentence,sentence,None,tokenizer,text_encoder) 
print('Cosine similarity between o and t:', oris_1) 

# generate s pair 
sentence_list , extra_key = generate_sentences(args.OO,args.TO,args.ge_key) 
mask,oris_2 = find_mask(target_sentence,sentence,sentence_list,extra_key,tokenizer,text_encoder) 
print(oris_2)

   
   
   










  

