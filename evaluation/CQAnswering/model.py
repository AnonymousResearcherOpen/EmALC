
import numpy as np
import torch
import torch.nn as nn

    
class DFALC(nn.Module):
    def __init__(self, params, conceptSize, roleSize, cEmb_init, rEmb_init,  device, name="Godel"):
        super().__init__()
        self.params = params
        self.conceptSize, self.roleSize = conceptSize, roleSize
        self.device = device
        self.cEmb = nn.Parameter(torch.tensor(cEmb_init))
        self.rEmb = nn.Parameter(torch.tensor(rEmb_init))
        self.relu = torch.nn.ReLU()
        # self.c_mask, self.r_mask = self.get_mask()
        self.logic_name = name
        self.epsilon = 1e-2
        self.p=2


    def to_sparse(self, A):
        return torch.sparse_coo_tensor(np.where(A!=0),A[np.where(A!=0)],A.shape)
    
    def index_sparse(self, A, idx):
        return torch.where(A.indices[0] in idx)
    
    def pi_0(self, x):
        return (1-self.epsilon)*x+self.epsilon
    
    def pi_1(self, x):
        return (1-self.epsilon)*x
    
    
    def neg(self, x, negf):
        negf = negf.unsqueeze(1)
        # print("negf: ",negf.shape)
        # print("x: ",x.shape)
        negf2 = negf*(-2) + 1
        # print("negf2: ",negf2)
        # print("negf2: ",negf2.shape)
        
        return negf2*x
        
    def t_norm(self, x, y):
        if self.logic_name == "Godel":
            return torch.minimum(x,y)
        elif self.logic_name == "LTN":
            return self.pi_0(x)*self.pi_0(y)
        # elif self.logic_name == "Product":
        #     return x*y
        
    def t_cnorm(self, x, y):
        if self.logic_name == "Godel":
            return torch.maximum(x,y)
        elif self.logic_name == "LTN":
            a = self.pi_1(x)
            b = self.pi_1(y)
            return a+b-a*b
        # elif self.logic_name == "Product":
        #     return x+y-x*y

    def forall(self, r, x):
        if self.logic_name == "Godel":
            return torch.min(self.t_cnorm(1-r,x.unsqueeze(1).expand(r.shape)),2).values
        elif self.logic_name == "LTN":
            return 1-torch.pow(torch.mean(torch.pow(1-self.pi_1(self.t_cnorm(r,x.unsqueeze(1).expand(r.shape))),self.p),2),1/self.p)
        # elif self.logic_name == "Product":
        #     return torch.prod(torch.max(-b,0),2)
    
    def exist(self, r, x):
        if self.logic_name == "Godel":
            return torch.max(self.t_norm(r,x.unsqueeze(1).expand(r.shape)),2).values
        elif self.logic_name == "LTN":
            return torch.pow(torch.mean(torch.pow(self.pi_0(self.t_norm(r,x.unsqueeze(1).expand(r.shape))),self.p),2),1/self.p)
    
    def L2(self, x, dim=1):
        return torch.sqrt(torch.sum((x)**2, dim))
    
    def L2_dist(self, x, y, dim=1):
        return torch.sqrt(torch.sum((x-y)**2, dim))
    
    def L1(self,x,dim=1):
        return torch.sum(torch.abs(x),dim)
    
    def L1_dist(self,x,y,dim=1):
        return torch.sum(torch.abs(x-y),dim)
    
    def HierarchyLoss(self, lefte, righte):
        return torch.mean(self.L1(self.relu(lefte-righte)))


        
    def forward(self, batch, atype, device):
        left, right, negf = batch
        # print("here negf: ", negf.shape)
        # print('here left: ',left.shape)
        # print("here right: ", right.shape)
        
        loss, lefte, righte, b_c_mask, b_r_mask = None, None, None, None, None
        
        self.cEmb[-1,:].detach().masked_fill_(self.cEmb[-1,:].gt(0.0),1.0)
        self.cEmb[-2,:].detach().masked_fill_(self.cEmb[-2,:].lt(1),0.0)
        
        
        if atype == 0:
            lefte = self.neg(self.cEmb[left],-negf[:,0])
            righte = self.neg(self.cEmb[right],negf[:,1])
            shape = lefte.shape
            # b_c_mask = self.c_mask[left] 
            
        elif atype == 1:
            righte = self.neg(self.cEmb[right], negf[:,2])
            shape = righte.shape
            lefte = self.t_norm(self.neg(self.cEmb[left[:,0]],negf[:,0]), self.neg(self.cEmb[left[:,1]],negf[:,1]))

        elif atype == 2:
            lefte = self.neg(self.cEmb[left], negf[:,0])
            shape = lefte.shape
            righte = self.t_norm(self.neg(self.cEmb[right[:,0]],negf[:,1]), self.neg(self.cEmb[right[:,1]],negf[:,2]))

        elif atype == 3:
            lefte = self.neg(self.cEmb[left], negf[:,0])
            shape = lefte.shape
            righte = self.exist(self.rEmb[right[:,0]], self.neg(self.cEmb[right[:,1]],negf[:,1]))

        elif atype == 4:
            lefte = self.neg(self.cEmb[left], negf[:,0])
            shape = lefte.shape
            righte = self.forall(self.rEmb[right[:,0]],self.neg(self.cEmb[right[:,1]], negf[:,1]))
            
            
        elif atype == 5:
            righte = self.neg(self.cEmb[right], negf[:,1])
            shape = righte.shape
            lefte = self.exist(self.rEmb[left[:,0]],self.neg(self.cEmb[left[:,1]], negf[:,0]))

        elif atype == 6:
            righte = self.neg(self.cEmb[right], negf[:,1])
            shape = righte.shape
            lefte = self.forall(self.rEmb[left[:,0]],self.neg(self.cEmb[left[:,1]], negf[:,0]))

        # print("lefte: ", lefte)
        # print("righte: ", righte)
        # print("atype: ",atype)
        loss = self.HierarchyLoss(lefte, righte)
        
            
        return loss