from collections import OrderedDict
import numpy as np
import torch
import sys
import os
import random
from tqdm import tqdm
sys.path.append('.')
from torch.utils.data import Dataset

import re
import pickle

class OntologyDataset(Dataset):
    def __init__(self, params, save_path):
        self.concept2id, self.role2id, self.individual2id, self.concepts, self.roles, self.individuals = self.load_entities(params["conceptPath"],params["rolePath"],params["individualPath"],save_path)
        print(self.concept2id)
        self.conceptSize, self.roleSize = len(self.concept2id), len(self.role2id)
        self.params = params
        self.left, self.right, self.neg = self.load_axiomdata(params["normalizationPath"])
        self.mode = 1
        
        
    def __len__(self):
        return self.left[self.mode].shape[0]
    
    def __getitem__(self, idx):
        return self.left[self.mode][idx],self.right[self.mode][idx], self.neg[self.mode][idx]
    
    
    def load_axiomdata(self, axiom_path):
        lines = open(axiom_path,"r",encoding="utf-8").read().splitlines()
        left, right, neg, corpus = [],[], [],[]
        atype = []
        left_, right_, neg_ = [[] for i in range(7)],[[] for i in range(7)],[[] for i in range(7)]
        cnt = []
      
        for line in lines:
            a = line.strip().split("|")
            sp = a[1].split("\t")
            idx = int(a[0])-1
            cnt.append(len(left_[idx]))
            
            if a[0] == "1":
                left_[idx].append(self.concept2id[sp[0]])
                right_[idx].append(self.concept2id[sp[1]])
                # corpus.append(self.uri_label[sp[0]] + ["subClassOf"] + self.uri_label[sp[1]])
            elif a[0] == "2":
                left_[idx].append([self.concept2id[sp[0]], self.concept2id[sp[1]]])
                right_[idx].append(self.concept2id[sp[2]])
                # corpus.append(self.uri_label[sp[0]]+["intersectionWith"]+self.uri_label[sp[1]]+["subClassOf"]+self.uri_label[sp[2]])
            elif a[0] == "3":
                left_[idx].append(self.concept2id[sp[0]])
                right_[idx].append([self.concept2id[sp[1]], self.concept2id[sp[2]]])
                # corpus.append(self.uri_label[sp[0]]+["subClassOf"]+self.uri_label[sp[1]]+["unionOf"]+self.uri_label[sp[2]])
            elif (a[0] == "4") or (a[0] == "5"):
                left_[idx].append(self.concept2id[sp[0]])
                right_[idx].append([self.role2id[sp[1]], self.concept2id[sp[2]]])
                # if a[0] == "4":
                #     corpus.append(self.uri_label[sp[0]]+["subClassOf","exists"]+self.uri_label[sp[1]]+self.uri_label[sp[2]])
                # else:
                #     corpus.append(self.uri_label[sp[0]]+["subClassOf","forall"]+self.uri_label[sp[1]]+self.uri_label[sp[2]])
            else:
                left_[idx].append([self.role2id[sp[0]], self.concept2id[sp[1]]])
                right_[idx].append(self.concept2id[sp[2]])
                # if a[0] == "6":
                #     corpus.append(["exists"]+self.uri_label[sp[0]]+self.uri_label[sp[1]]+["subClassOf"]+self.uri_label[sp[2]])
                # else:
                #     corpus.append(["forall"]+self.uri_label[sp[0]]+self.uri_label[sp[1]]+["subClassOf"]+self.uri_label[sp[2]])
            sp = a[2].split(" ")
            neg_[idx].append([int(i) for i in sp])
            atype.append(idx)
        
        for i in range(7):
            left.append(torch.LongTensor(left_[i]))
            right.append(torch.LongTensor(right_[i]))
            neg.append(torch.LongTensor(neg_[i]))
        
        return left, right, neg
     
    



    @staticmethod
    def load_entities(concepts_path, roles_path, individuals_path, save_path):
        # if os.path.exists(save_path+".c2id.pkl"):
        #     concept2id = pickle.load(open(save_path+".c2id.pkl","rb"))
        #     role2id = pickle.load(open(save_path+".r2id.pkl","rb"))
        #     individual2id = pickle.load(open(save_path+".i2id.pkl","rb"))
        #     return concept2id, role2id, individual2id, list(concept2id.keys()), list(role2id.keys()), list(individual2id.keys())
        # else:
        concepts = list(open(concepts_path,"r",encoding="utf-8").read().splitlines())
        roles = list(open(roles_path,"r",encoding="utf-8").read().splitlines())
        individuals = list(open(individuals_path,"r",encoding="utf-8").read().splitlines())
        concept2id = {}
            
        if "http://www.w3.org/2002/07/owl#Thing" in concepts:
            concepts.remove("http://www.w3.org/2002/07/owl#Thing")
        if "http://www.w3.org/2002/07/owl#Nothing" in concepts:
            concepts.remove("http://www.w3.org/2002/07/owl#Nothing")
        if "owl:Thing" in concepts:
            concepts.remove("owl:Thing")
        if "owl:Nothing" in concepts:
            concepts.remove("owl:Nothing")
        
        concepts = list(concepts)
    
        roles = list(roles)

        for i,c in enumerate(concepts):
            concept2id[c] = i
        concept2id["owl:Nothing"] = len(concepts)
        concept2id["http://www.w3.org/2002/07/owl#Nothing"] = len(concepts)
        concept2id["owl:Thing"] = len(concepts)+1
        concept2id["http://www.w3.org/2002/07/owl#Thing"] = len(concepts)+1
        
        concepts.append("http://www.w3.org/2002/07/owl#Nothing")
        concepts.append("http://www.w3.org/2002/07/owl#Thing")
        
        concepts.append("owl:Nothing")
        concepts.append("owl:Thing")
        
        role2id = {r:i for i,r in enumerate(roles)}
        individual2id = {r:i for i,r in enumerate(individuals)}
        
        pickle.dump(concept2id,open(save_path+".c2id.pkl","wb"))
        pickle.dump(role2id,open(save_path+".r2id.pkl","wb"))
        pickle.dump(individual2id,open(save_path+".i2id.pkl","wb"))
        return concept2id, role2id, individual2id, concepts, roles, individuals