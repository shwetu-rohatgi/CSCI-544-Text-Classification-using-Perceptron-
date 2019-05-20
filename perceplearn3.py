# use this file to learn perceptron classifier 
# Expected: generate vanillamodel.txt and averagemodel.txt
import sys
import glob
import os
import collections
import re
import numpy as np
import json
import random

all_files = glob.glob(os.path.join(sys.argv[1], '*/*/*/*.txt'))

#print (all_files)
train_by_class = collections.defaultdict(list)

allfileList = []
for f in all_files:
    class1, class2, fold, fname = f.split('/')[-4:]
    if class1=='positive_polarity' and class2=='deceptive_from_MTurk':
        review_tuple = (f, 'positive', 'deceptive')
    elif class1=='positive_polarity' and class2=='truthful_from_TripAdvisor':
        review_tuple = (f, 'positive', 'truthful')
    elif class1=='negative_polarity' and class2=='deceptive_from_MTurk':
        review_tuple = (f, 'negative', 'deceptive')
    elif class1=='negative_polarity' and class2=='truthful_from_Web':
        review_tuple = (f, 'negative', 'truthful')
    
    allfileList.append(review_tuple)    


def pre_processing(text_line, curr_features):
    text_line = text_line.replace('\n','')
    text_line = text_line.replace('\t','')
    
    cleaned_line = re.sub('[^a-z\s]+',' ',text_line)
    
    cleaned_line = re.sub('(\s+)',' ',cleaned_line)
    
    stop_words = set(['1','2','3','4','5','6','7','8','9','0','it', 'hers', 'between', 'yourself', 'but', 'the','again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are','his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'up', 'to', 'ours', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than','hotel','stay','i','we', 'these', 'your','while', 'above', 'both','where', 'too', 'only','had', 'she', 'all','do', 'its', 'yours', 'such','chicago','day','ourselves','no', 'when', 'at', 'any','who', 'as', 'from', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 'u', 'v', 'w', 'x', 'y', 'z'])
    
    final_string = ""
    
    for word in cleaned_line.split():        
        if word in stop_words:
            continue
        else:
            if word in curr_features:
                curr_features[word] += 1
            else:
                curr_features[word] = 1
    
    return curr_features


#word_review_count =[0,0,0,0] #{'positive': 0, 'negative': 0, 'truthful': 0, 'deceptive': 0}

def train_perceptron():
    labels = {'positive':1, 'negative':-1, 'truthful':1, 'deceptive':-1}
    pn_weight_vector = {'__bais__' : 0}
    td_weight_vector = {'__bais__' : 0}
    
    avg_pn_weight_vector = {'__bais__' : 0}
    avg_td_weight_vector = {'__bais__' : 0}
    
    u_avg_pn_weight_vector = {'__bais__' : 0}
    u_avg_td_weight_vector = {'__bais__' : 0}
    #weight_vector = {'__bais__' : 0}
    
    c=1
    
    maxIter = 100
    for i in range(maxIter):
        random.shuffle(allfileList) 
        for review_file in allfileList:
            f = open(review_file[0],'r')
            current_review_features = {}
            for line in f:
                current_review_features = pre_processing(line, {})
                
                #weight vector intialized with zero
                for k in current_review_features.keys():
                    if k not in pn_weight_vector:
                        pn_weight_vector[k] = 0
                        avg_pn_weight_vector[k] = 0
                        u_avg_pn_weight_vector[k] = 0
                    if k not in td_weight_vector:
                        td_weight_vector[k] = 0
                        avg_td_weight_vector[k] = 0
                        u_avg_td_weight_vector[k] = 0
            
            pn_activation = pn_weight_vector['__bais__']
            td_activation = td_weight_vector['__bais__']
            avg_pn_activation = avg_pn_weight_vector['__bais__']
            avg_td_activation = avg_td_weight_vector['__bais__']
            for k in current_review_features:
                pn_activation += current_review_features[k] * pn_weight_vector[k]
                td_activation += current_review_features[k] * td_weight_vector[k]
                
                avg_pn_activation += current_review_features[k] * avg_pn_weight_vector[k]
                avg_td_activation += current_review_features[k] * avg_td_weight_vector[k]
                
            if pn_activation*labels[review_file[1]] <= 0:
                for k in current_review_features:
                    pn_weight_vector[k] += labels[review_file[1]] * current_review_features[k]
                    pn_weight_vector['__bais__'] += labels[review_file[1]]
                    
            if td_activation*labels[review_file[2]] <= 0:
                for k in current_review_features:
                    td_weight_vector[k] += labels[review_file[2]] * current_review_features[k]
                    td_weight_vector['__bais__'] += labels[review_file[2]]
                    
            if avg_pn_activation*labels[review_file[1]] <= 0:
                for k in current_review_features:
                    avg_pn_weight_vector[k] += labels[review_file[1]] * current_review_features[k]
                    avg_pn_weight_vector['__bais__'] += labels[review_file[1]]
                    u_avg_pn_weight_vector[k] += labels[review_file[1]] * c * current_review_features[k]
                    u_avg_pn_weight_vector['__bais__'] += labels[review_file[1]] * c
                
            if avg_td_activation*labels[review_file[2]] <= 0:
                for k in current_review_features:
                    avg_td_weight_vector[k] += labels[review_file[2]] * current_review_features[k]
                    avg_td_weight_vector['__bais__'] += labels[review_file[2]]
                    u_avg_td_weight_vector[k] += labels[review_file[2]] * c * current_review_features[k]
                    u_avg_td_weight_vector['__bais__'] += labels[review_file[2]] * c
                
            c = c+1
    
    for keys in avg_pn_weight_vector:
        if keys in u_avg_pn_weight_vector:
            avg_pn_weight_vector[keys] -= u_avg_pn_weight_vector[keys]/(c*1.0)
            
    for keys in avg_td_weight_vector:
        if keys in u_avg_td_weight_vector:
            avg_td_weight_vector[keys] -= u_avg_td_weight_vector[keys]/(c*1.0)
            
    vanilla = {'pn': pn_weight_vector, 'td': td_weight_vector}
    averaged = {'pn': avg_pn_weight_vector, 'td': avg_td_weight_vector}
    fhandle = open('vanillamodel.txt','w')
    fhandle.write(json.dumps(vanilla, indent=2))
    fhandle.close()
    fhandle = open('averagedmodel.txt','w')
    fhandle.write(json.dumps(averaged, indent=2))
    fhandle.close()
#Running my classifier
train_perceptron()