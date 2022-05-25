import numpy as np
import re
import os 

np.random.seed(42)
class MaskABox():
    def __init__(self, abox_path, c2id, r2id, i2id, alpha=0.9, save=True, save_path="input\Family.owl", mask_rate=0.2, only_cEmb=False):
        self.alpha = alpha
        self.c2id, self.r2id, self.i2id = c2id, r2id, i2id
        self.alpha = alpha
        
        self.true_cEmb, self.true_rEmb = self.load_abox_axioms(abox_path)
        self.masked_cEmb, self.masked_rEmb = self.mask_abox(mask_rate=mask_rate, only_cEmb=only_cEmb)
        
        
        if os.path.exists(save_path+".true_cEmb.npy"):
            self.true_cEmb = np.load(open(save_path+".true_cEmb.npy","rb"))
            self.true_rEmb = np.load(open(save_path+".true_rEmb.npy","rb"))
            self.masked_cEmb = np.load(open(save_path+".masked_cEmb.npy","rb"))
            self.masked_rEmb = np.load(open(save_path+".masked_rEmb.npy","rb"))
        else:
            if save:
                np.save(open(save_path+".true_cEmb.npy","wb"),self.true_cEmb)
                np.save(open(save_path+".true_rEmb.npy","wb"),self.true_rEmb)
                np.save(open(save_path+".masked_cEmb.npy","wb"),self.masked_cEmb)
                np.save(open(save_path+".masked_rEmb.npy","wb"),self.masked_rEmb)
        
        
    def MSE(self,A,B):
        mse = np.power(A-B,2)/len(A)
        return mse
    
    
    def load_abox_axioms(self, abox_path):
        # Negative assertions assigning is not realized here. Because most ontology engineering only assert positive assertions.
        # i.e. a:¬C and (a,b) has no relation r are not asserted in most of the cases.
        # So either fill cEmb and rEmb with 0.5 or 0 will not affect the results.
        cEmb, rEmb = np.zeros((len(self.c2id)-2,len(self.i2id))), np.zeros((len(self.r2id), len(self.i2id), len(self.i2id)))
        print("cEmb shape: ", cEmb.shape)
        print("rEmb shape: ", rEmb.shape)
        
        # Optional
        cEmb.fill(0.5)
        rEmb.fill(0.5)
        re_c = r'(.*)\(([^,]*)\)'
        re_r = r'(.*)\(([^,]*),([^,]*)\)'
        with open(abox_path, "r") as f:
            for l in f.readlines():
                k = re.match(re_c,l.strip())
                if k == None:
                    k = re.match(re_r,l.strip())
                p = list(k.groups())
                p = [t.strip() for t in p]
                
                if len(p) == 2:
                    cEmb[self.c2id[p[0]], self.i2id[p[1]]] = 1
                else:
                    if p[0] not in self.r2id: continue
                    rEmb[self.r2id[p[0]], self.i2id[p[1]], self.i2id[p[2]]] = 1
                    
                
                
        cEmb[-1] = np.ones((1,len(self.i2id)))
        cEmb[-2] = np.zeros((1,len(self.i2id)))
        return cEmb, rEmb
    
    def mask_abox(self, mask_rate = 0.2, only_cEmb = False):
        masked_cEmb, masked_rEmb = np.array(self.true_cEmb, copy=True), np.array(self.true_rEmb, copy=True)
        coords = []
        for i in range(len(self.c2id)-4):
            for j in range(len(self.i2id)):
                coords.append([i,j])

        size = len(coords)
        print("cEmb masked size: ", int(mask_rate*size))
        for i in np.random.choice(size, int(mask_rate*size), replace=False):
            x,y = coords[i]
            masked_cEmb[x,y] = np.random.uniform(1-self.alpha,self.alpha)
        
        if only_cEmb == False:
            coords = set()
                        
            size = len(self.r2id)*len(self.i2id)
            print("rEmb masked size: ", int(mask_rate*size))
            for i in np.random.choice(len(self.r2id)*len(self.i2id), int(mask_rate*size), replace=False):
                x,y,z = np.random.randint(0,high=len(self.r2id)), np.random.randint(0,high=len(self.i2id)), np.random.randint(0,high=len(self.i2id))
                while (x,y,z) in coords:
                    x,y,z = np.random.randint(0,high=len(self.r2id)), np.random.randint(0,high=len(self.i2id)), np.random.randint(0,high=len(self.i2id))
                coords.add((x,y,z))
                v = np.random.uniform(1-self.alpha,self.alpha)
                while (v == 1-self.alpha) or (v == self.alpha):
                    v = np.random.uniform(1-self.alpha,self.alpha)
                masked_rEmb[x,y,z] = v
        return masked_cEmb, masked_rEmb
    
    
class CQAnswering():
    def __init__(self, query_path, answer_path, c2id, r2id, i2id):
        self.query_answer = self.load_query_answer(query_path, answer_path, c2id, r2id, i2id)
        self.id2c = {i:c for c,i in c2id.items()}
        self.id2r = {i:r for r,i in r2id.items()}
        self.id2i = {i:ind for ind,i in i2id.items()}
        
    def load_data(self, path):
        data = {1:[],2:[],3:[]}
        with open(path,"r") as f:
            lines = f.readlines()
            for l in lines:
                mode = int(l[0])
                indi = l[2:].strip().split(" ")
                data[mode].append(indi)
        return data
    
    
    def load_query_answer(self, query_path, answer_path, c2id, r2id, i2id):
        query_data = self.load_data(query_path)
        answer_data = self.load_data(answer_path)
        qa = {1:[],2:[],3:[]}
        for i, ds in query_data.items():
            for j, d in enumerate(ds):
                if i == 1:
                    q = [c2id[cur] for cur in d]
                else:
                    if d[-2] not in r2id: continue
                    q = [c2id[cur] for cur in d[:-2]] + [r2id[d[-2]]] + [c2id[d[-1]]]
                
                a = [i2id[cur] for cur in answer_data[i][j]]
                qa[i].append((q,a))
        return qa
    
    def get_score(self, model, cEmb, rEmb, alpha=0.8):
        def conjunct(a,b):
            if b.shape[0] == 1:
                return model.t_norm(a,b)[0]
            
            return model.t_norm(a, conjunct(b[0,:],b[1:,:]))[0]
        
        def precision_recall(predict, true):
            inter = set(predict).intersection(set(true))
            if (len(predict) == 0) or (len(true) == 0):
                return 0,0
            return len(inter)/len(predict), len(inter)/len(true)
            
        precision, recall = 0.0, 0.0
        cnt = 0
        for i, qas in self.query_answer.items():
            for qa in qas:
                q,a = qa
                if i == 1:
                    predicted_a = conjunct(cEmb[q[0]], cEmb[q[1:]])
                    str_q = [self.id2c[c] for c in q]
                elif i == 2:
                    predicted_a = conjunct(model.exist(rEmb[q[-2]].unsqueeze(0),cEmb[q[-1]]),cEmb[q[:-2]])
                    str_q = [self.id2c[c] for c in q[:-2]] + [self.id2r[q[-2]]] + [self.id2c[q[-1]]]
                else:
                    predicted_a = conjunct(model.forall(rEmb[q[-2]],cEmb[q[-1]]),cEmb[q[:-2]])
                predicted_a = predicted_a.detach().numpy()
                predicted_set = np.where(predicted_a>=alpha)[0]
                
                
                str_pa = [self.id2i[i] for i in predicted_set]
                str_a = [self.id2i[i] for i in a]
                # print("Query: "+" ".join(str_q))
                # print("Predicted answers: "+" ".join(str_pa))
                # print("True answers: "+" ".join(str_a))
                p,r = precision_recall(predicted_set,a)
                precision += p
                recall += r
                cnt += 1
                # print("p: {}, r: {}".format(p,r))
        if cnt == 0:
            return 0,0
        return precision/cnt, recall/cnt
        