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
        # self.rEmb.requires_grad = False
        self.relu = torch.nn.ReLU()
        self.gelu = torch.nn.GELU()
        # self.c_mask, self.r_mask = self.get_mask()
        self.logic_name = name
        self.epsilon = 1e-2
        self.p=2
        self.alpha=params["alpha"]

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
        
        loss, lefte, righte, loss1, loss2 = None, None, None, None, None
        
        # self.cEmb[-1,:].detach().masked_fill_(self.cEmb[-1,:].gt(0.0),1.0)
        # self.cEmb[-2,:].detach().masked_fill_(self.cEmb[-2,:].lt(1),0.0)
        
        
        if atype == 0:
            lefte = self.neg(self.cEmb[left],-negf[:,0])
            righte = self.neg(self.cEmb[right],negf[:,1])
            shape = lefte.shape
            
            
        elif atype == 1:
            righte = self.neg(self.cEmb[right], negf[:,2])
            shape = righte.shape
            lefte = self.t_norm(self.neg(self.cEmb[left[:,0]],negf[:,0]), self.neg(self.cEmb[left[:,1]],negf[:,1]))
            
            
        elif atype == 2:
            lefte = self.neg(self.cEmb[left], negf[:,0])
            shape = lefte.shape
            righte = self.t_cnorm(self.neg(self.cEmb[right[:,0]],negf[:,1]), self.neg(self.cEmb[right[:,1]],negf[:,2]))
            

        elif atype == 3:
            B = lefte = self.neg(self.cEmb[left], negf[:,0])
            shape = lefte.shape
            righte = self.exist(self.rEmb[right[:,0]], self.neg(self.cEmb[right[:,1]],negf[:,1]))
            A = self.neg(self.cEmb[right[:,1]], negf[:,1])
            expected_B = torch.bmm(self.rEmb[right[:,0]],(A).unsqueeze(2)).squeeze()
            expected_A = torch.bmm((B).unsqueeze(1), self.rEmb[right[:,0]]).squeeze()
            # expected_r = torch.bmm(B.unsqueeze(2),A.unsqueeze(1))
            loss1 = (-A)*((self.relu(expected_A-(self.alpha))*self.relu(self.alpha-A)).detach())#+torch.mean((-self.rEmb[right[:,0]])*((self.relu(expected_r - self.rEmb[right[:,0]])).detach()),1) #+ B*((self.relu(1-self.alpha-expected_B)*self.relu(B-(1-self.alpha))).detach()) #+torch.mean((1-self.rEmb[right[:,0]])*((self.relu(expected_r-self.alpha)*self.relu( self.alpha-self.rEmb[right[:,0]])).detach()),2) #(A)*((self.relu(expected_A-self.alpha)*(self.relu(B-(1-self.alpha)))).detach()) + (B)*((self.relu(expected_B-self.alpha)*(self.relu(A-(1-self.alpha)))).detach()) #+ torch.mean((1-expected_r)*((self.relu(expected_r - self.rEmb[right[:,0]])).detach()),2) + B*((self.relu(1-self.alpha-expected_B)*self.relu(B-(1-self.alpha))).detach())

        elif atype == 4:
            B = lefte = self.neg(self.cEmb[left], negf[:,0])
            shape = lefte.shape
            righte = self.forall(self.rEmb[right[:,0]],self.neg(self.cEmb[right[:,1]], negf[:,1]))
            A = self.neg(self.cEmb[right[:,1]], negf[:,1])
            expected_B = torch.bmm(self.rEmb[right[:,0]],(A).unsqueeze(2)).squeeze()
            expected_A = torch.bmm((B).unsqueeze(1), self.rEmb[right[:,0]]).squeeze()
            loss1 = (-A)*((self.relu(expected_A-self.alpha)*self.relu(self.alpha-A)).detach())#+righte*(self.relu(lefte-self.alpha)*self.relu(righte-self.alpha)).detach()#+  B*((self.relu(1-self.alpha-expected_B)*self.relu(B-(1-self.alpha))).detach())#(B)*((self.relu(expected_B-self.alpha)*(self.relu(A-(1-self.alpha)))).detach())
            
        elif atype == 5:
            B = righte = self.neg(self.cEmb[right], negf[:,1])
            shape = righte.shape
            lefte = self.exist(self.rEmb[left[:,0]],self.neg(self.cEmb[left[:,1]], negf[:,0]))
            A = self.neg(self.cEmb[left[:,1]], negf[:,0])
            
            expected_B = torch.bmm(self.rEmb[left[:,0]],A.unsqueeze(2)).squeeze()#.squeeze(2)
            loss1 = (-B)*((self.relu(expected_B-self.alpha)*self.relu(self.alpha-B)).detach())#+lefte*(self.relu(1-self.alpha-B)*self.relu(lefte-(1-self.alpha))).detach()
            # righte1 = torch.max(torch.minimum(self.rEmb[left[:,0]],lefte2.unsqueeze(1)),1).values
            # print("righte: ", righte.shape)
            # print("righte1:", righte1.shape)
            
            # print("righte2: ", righte2.shape)
            # print("hehre: ",((1-self.relu(righte-self.alpha))*self.relu(righte1-righte)).detach())
            # loss1 =  (1-righte)*(((1-self.relu(righte-self.alpha))*self.relu(righte1-righte)).detach()) #self.relu(torch.max(self.rEmb[left[:,0]],1).values-self.alpha)*
            # loss1 =  (-righte)*100*((self.relu(torch.max(self.rEmb[left[:,0]],2).values-self.alpha)*self.relu(righte1-self.alpha)*self.relu(self.alpha-righte)).detach())
            # loss1 = -lefte2*100*((self.relu(torch.max(self.rEmb[left[:,0]],1).values-0.9)*(1-self.relu(lefte2-0.9))*self.relu(righte2-0.9)).detach()) \
            #         -righte*100*((self.relu(lefte-0.9)*(1-self.relu(righte-0.9))).detach())
            
                #self.relu(torch.max(self.rEmb[left[:,0]],2).values-self.alpha)*
            
        elif atype == 6:
            B = righte = self.neg(self.cEmb[right], negf[:,1])
            shape = righte.shape
            lefte = self.forall(self.rEmb[left[:,0]],self.neg(self.cEmb[left[:,1]], negf[:,0]))
            A = self.neg(self.cEmb[left[:,1]], negf[:,0])
            expected_B = torch.bmm(self.rEmb[left[:,0]],A.unsqueeze(2)).squeeze()
            expected_A = torch.bmm(B.unsqueeze(1), self.rEmb[left[:,0]]).squeeze()
            # expected_r = torch.bmm(B.unsqueeze(2),A.unsqueeze(1))
            # righte1 = torch.max(torch.minimum(self.rEmb[left[:,0]],lefte2.unsqueeze(1)),1).values
            # righte2 = torch.max(torch.minimum(righte.unsqueeze(2),self.rEmb[left[:,0]]),2).values
            loss1 =  (-B)*((self.relu(expected_B-self.alpha)*self.relu(self.alpha-B)).detach())+(-A)*(self.relu(expected_A-self.alpha)*self.relu(self.alpha-A)).detach()#  +torch.mean((-self.rEmb[left[:,0]])*((self.relu(expected_r - self.rEmb[left[:,0]])).detach()),1)
                                                                                        
            #+ torch.mean((-expected_r)*((self.relu(expected_r - self.rEmb[left[:,0]])).detach()),2) #(self.relu(torch.max(self.rEmb[left[:,0]],1).values-self.alpha)*
            # loss1 = (1-lefte2)*(((1-self.relu(lefte2-self.alpha))*self.relu(righte2-lefte2)).detach())
            # loss1 = -lefte2*100*((self.relu(torch.max(self.rEmb[left[:,0]],1).values-0.9)*(1-self.relu(lefte2-0.9))*self.relu(righte2-0.9)).detach()) (-A)*((self.relu(expected_A-self.alpha)*self.relu(self.alpha-A)).detach()) +
        # print("lefte: ", lefte)
        # print("righte: ", righte)
        # print("r: ", torch.max(self.rEmb[left[:,0]],1).values)
        # print("lefte2: ", lefte2)
        # print("righte2: ", righte2)
        # print("righte1: ", righte1)
        # print("loss1: ", loss1)
        # print("atype: ",atype)
        loss = self.HierarchyLoss(lefte,righte)
        if atype < 3:
            loss1 = -righte*(self.relu(lefte-righte).detach())#-righte*(self.relu(lefte-self.alpha)*self.relu(lefte-righte)).detach()#+ lefte*(self.relu(1-self.alpha-righte)*self.relu(lefte-righte)).detach()#-righte*(self.relu(lefte-righte).detach())#+ lefte*(self.relu(1-self.alpha-righte).detach())
        return torch.mean(torch.sum(loss1,1))
        # if loss1 == None:
        #     return self.HierarchyLoss(lefte, righte)
        # return torch.mean(torch.sum(loss1,1))
        # print("loss: ", self.relu(lefte-righte)+loss1)
        
        
        


class DFALC2(nn.Module):
    def __init__(self, params, conceptSize, roleSize, cEmb_init, rEmb_init,  device, name="Godel"):
        super().__init__()
        self.params = params
        self.conceptSize, self.roleSize = conceptSize, roleSize
        self.device = device
        self.cEmb = nn.Parameter(torch.tensor(cEmb_init))
        self.rEmb = nn.Parameter(torch.tensor(rEmb_init))
        self.relu = torch.nn.ReLU()
        self.gelu = torch.nn.GELU()
        # self.c_mask, self.r_mask = self.get_mask()
        self.logic_name = name
        self.epsilon = 1e-2
        self.p=2
        self.alpha=0.8

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
        
        loss, lefte, righte, loss1 = None, None, None, None
        
        self.cEmb[-1,:].detach().masked_fill_(self.cEmb[-1,:].gt(0.0),1.0)
        self.cEmb[-2,:].detach().masked_fill_(self.cEmb[-2,:].lt(1),0.0)
        
        
        if atype == 0:
            lefte = self.neg(self.cEmb[left],-negf[:,0])
            righte = self.neg(self.cEmb[right],negf[:,1])
            shape = lefte.shape
            # b_c_mask = self.c_mask[left] 
            loss1 = -righte*(self.relu(lefte-righte).detach())
        elif atype == 1:
            righte = self.neg(self.cEmb[right], negf[:,2])
            shape = righte.shape
            lefte = self.t_norm(self.neg(self.cEmb[left[:,0]],negf[:,0]), self.neg(self.cEmb[left[:,1]],negf[:,1]))
            loss1 = -righte*(self.relu(lefte-righte).detach())
            
        elif atype == 2:
            lefte = self.neg(self.cEmb[left], negf[:,0])
            shape = lefte.shape
            righte = self.t_cnorm(self.neg(self.cEmb[right[:,0]],negf[:,1]), self.neg(self.cEmb[right[:,1]],negf[:,2]))
            loss1 = -righte*(self.relu(lefte-righte).detach())

        elif atype == 3:
            lefte = self.neg(self.cEmb[left], negf[:,0])
            shape = lefte.shape
            righte = self.exist(self.rEmb[right[:,0]], self.neg(self.cEmb[right[:,1]],negf[:,1]))
            lefte2 = self.neg(self.cEmb[right[:,1]], negf[:,1])
            righte2 = torch.bmm(lefte.unsqueeze(1), self.rEmb[right[:,0]]).squeeze()
            righte1 = torch.bmm(self.rEmb[right[:,0]],lefte2.unsqueeze(2)).squeeze()
            loss1 = (lefte2)*((self.relu(1-self.alpha-righte2)*self.relu(lefte2-(1-self.alpha))).detach()) + (righte-1)*((self.relu(1-self.alpha-righte1)*self.relu(righte-(1-self.alpha))).detach())

        elif atype == 4:
            lefte = self.neg(self.cEmb[left], negf[:,0])
            shape = lefte.shape
            righte = self.forall(self.rEmb[right[:,0]],self.neg(self.cEmb[right[:,1]], negf[:,1]))
            lefte2 = self.neg(self.cEmb[right[:,1]], negf[:,1])
            righte1 = torch.bmm(self.rEmb[right[:,0]],lefte2.unsqueeze(2)).squeeze()
            loss1 = (righte)*((self.relu(1-self.alpha-righte1)*self.relu(righte-(1-self.alpha))).detach())
            
        elif atype == 5:
            righte = self.neg(self.cEmb[right], negf[:,1])
            shape = righte.shape
            lefte = self.exist(self.rEmb[left[:,0]],self.neg(self.cEmb[left[:,1]], negf[:,0]))
            lefte2 = self.neg(self.cEmb[left[:,1]], negf[:,0])
            righte2 = torch.bmm(righte.unsqueeze(1), self.rEmb[left[:,0]]).squeeze()#.squeeze(2)
            # print("self.rEmb[left[:,0]]: ", self.rEmb[left[:,0]].shape)
            # print("lefte2: ", lefte2.unsqueeze(2).shape)
            righte1 = torch.bmm(self.rEmb[left[:,0]],lefte2.unsqueeze(2)).squeeze()#.squeeze(2)
            
            # print("righte: ", righte.shape)
            # print("righte1:", righte1.shape)
            
            # print("righte2: ", righte2.shape)
            # print("hehre: ",((1-self.relu(righte-self.alpha))*self.relu(righte1-righte)).detach())
            # loss1 =  (1-righte)*(((1-self.relu(righte-self.alpha))*self.relu(righte1-righte)).detach()) #self.relu(torch.max(self.rEmb[left[:,0]],1).values-self.alpha)*
            # loss1 =  (-righte)*100*((self.relu(torch.max(self.rEmb[left[:,0]],2).values-self.alpha)*self.relu(righte1-self.alpha)*self.relu(self.alpha-righte)).detach())
            # loss1 = -lefte2*100*((self.relu(torch.max(self.rEmb[left[:,0]],1).values-0.9)*(1-self.relu(lefte2-0.9))*self.relu(righte2-0.9)).detach()) \
            #         -righte*100*((self.relu(lefte-0.9)*(1-self.relu(righte-0.9))).detach())
            loss1 = (1-righte)*((self.relu(righte1-self.alpha)*self.relu(self.alpha-righte)).detach()) #self.relu(torch.max(self.rEmb[left[:,0]],2).values-self.alpha)*
            
        elif atype == 6:
            righte = self.neg(self.cEmb[right], negf[:,1])
            shape = righte.shape
            lefte = self.forall(self.rEmb[left[:,0]],self.neg(self.cEmb[left[:,1]], negf[:,0]))
            lefte2 = self.neg(self.cEmb[left[:,1]], negf[:,0])
            righte2 = torch.bmm(righte.unsqueeze(1), self.rEmb[left[:,0]]).squeeze()
            righte1 = torch.bmm(self.rEmb[left[:,0]],lefte2.unsqueeze(2)).squeeze()
            loss1 = (1-lefte2)*((self.relu(righte2-self.alpha)*self.relu(self.alpha-lefte2)).detach())-righte*((self.relu(righte1-self.alpha)*self.relu(self.alpha-righte)).detach()) #(self.relu(torch.max(self.rEmb[left[:,0]],1).values-self.alpha)*
            # loss1 = (1-lefte2)*(((1-self.relu(lefte2-self.alpha))*self.relu(righte2-lefte2)).detach())
            # loss1 = -lefte2*100*((self.relu(torch.max(self.rEmb[left[:,0]],1).values-0.9)*(1-self.relu(lefte2-0.9))*self.relu(righte2-0.9)).detach())
        # print("lefte: ", lefte)
        # print("righte: ", righte)
        # print("r: ", torch.max(self.rEmb[left[:,0]],1).values)
        # print("lefte2: ", lefte2)
        # print("righte2: ", righte2)
        # print("righte1: ", righte1)
        # print("loss1: ", loss1)
        # print("atype: ",atype)
        loss = self.HierarchyLoss(lefte,righte)
        if loss1 == None:
            return self.HierarchyLoss(lefte, righte)
        return torch.mean(torch.sum(loss1,1))
        
        print("loss: ", self.relu(lefte-righte)+loss1)