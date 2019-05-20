# use this file to classify using perceptron classifier 
# Expected: generate percepoutput.txt

import sys
import glob
import os
import collections
import re
import numpy as np
import json
import random

all_files = glob.glob(os.path.join(sys.argv[2], '*/*/*/*.txt'))
#print(all_files)

def pre_processing(text_line, curr_features):
    text_line = text_line.replace('\n','')
    text_line = text_line.replace('\t','')
    
    cleaned_line = re.sub('[^a-z\s]+',' ',text_line)
    
    cleaned_line = re.sub('(\s+)',' ',cleaned_line)
    
    stop_words = set(['1','2','3','4','5','6','7','8','9','0','it', 'hers', 'between', 'yourself', 'but', 'the','again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are','his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'up', 'to', 'ours', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than','hotel','stay','i','we', 'these', 'your','while', 'above', 'both','where', 'too', 'only','had', 'she', 'all','do', 'its', 'yours', 'such','chicago','day','ourselves','no', 'when', 'at', 'any','who', 'as', 'from','b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 'u', 'v', 'w', 'x', 'y', 'z'])
    
    final_string = ""
    
    for word in cleaned_line.split():        
        if word in stop_words:
            continue
        else:
            final_string = final_string+" "+word
            
            if word in curr_features:
                curr_features[word] += 1
            else:
                curr_features[word] = 1
    
    return curr_features

def classify():
    fopen = open('percepoutput.txt','w')
    labels = {'positive':1, 'negative':-1, 'truthful':1, 'deceptive':-1}
    model_dict = open(sys.argv[1]).read()
    vanilla_dict = json.loads(model_dict)
    #print(vanilla_dict)
    for f in all_files:
        fhandle = open(f,'r')
        file = fhandle.read()
        fhandle.close()
        current_review_features = {}
        current_review_features = pre_processing(file, {})
        activation_pn = vanilla_dict['pn']['__bais__']
        activation_td = vanilla_dict['td']['__bais__']
        for word in current_review_features:
            if word in vanilla_dict['pn']:
                activation_pn += vanilla_dict['pn'][word]*current_review_features[word]
            if word in vanilla_dict['td']:
                activation_td += vanilla_dict['td'][word]*current_review_features[word]
        
        string = ""
        if activation_td>=0:
            string += "truthful"
        elif activation_td<0:
            string += "deceptive"
        if activation_pn>=0:
            string += " positive "
        elif activation_pn<0:
            string += " negative "
        string+=str(f)
        string+="\n"
        
        #print(string)
        fopen.write(string)
    
    fopen.close() 
    
classify()