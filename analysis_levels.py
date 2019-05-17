import numpy as np
import ursa.similarity as S
import ursa.util as U
import ursa.regress as R
from sklearn.metrics.pairwise import cosine_similarity
import ursa.regress
import torch
import torch.optim

def edit_similarity_mean_pool():
    stack = np.load("../talkie/state_stack_flickr8k_val.npy")
    trans = np.load("../talkie/transcription_flickr8k_val.npy")
    ref = trans[700:]
    trans = trans[:700]
    edit_sim = U.pairwise(S.stringsim, trans)
    Y = ursa.regress.embed(trans, ref, S.stringsim)
    print("layer mean avg regress")
    for layer in range(4): 
         mean_pool = np.vstack([item[:,layer,:].mean(axis=0) for item in stack[:700] ])  
         mean_pool_sim = cosine_similarity(mean_pool)
         reg = R.Regress()
         reg.fit(mean_pool, Y)
         wa = S.WeightedAverage(1024)
         avg_pool = np.vstack([ wa(torch.tensor([item[:,layer,:]])).detach().numpy() for item in stack[:700] ])
         avg_pool_sim = cosine_similarity(avg_pool)
         print(layer,
               U.pearson_r(U.triu(mean_pool_sim), U.triu(edit_sim)),
               U.pearson_r(U.triu(avg_pool_sim), U.triu(edit_sim)),
               reg.report()['pearson_r']['mean'])
         
def weighted_average(epochs=1, device='cpu', factor=1):
    stackdata = np.load("../talkie/state_stack_flickr8k_val.npy")
    trans = np.load("../talkie/transcription_flickr8k_val.npy")
    trans_val = trans[500:]
    trans = trans[:500]    
    edit_sim = torch.tensor(U.pairwise(S.stringsim, trans)).float().to(device)
    edit_sim_val = torch.tensor(U.pairwise(S.stringsim, trans_val)).float().to(device)
    print("layer epoch train_loss val_loss")
    for layer in range(4):
        stack = [ torch.tensor([item[:, layer, :]]).float().to(device) for item in stackdata ]
        stack_val = stack[500:]
        stack = stack[:500]

        wa = S.WeightedAverage(1024, 1024//factor).to(device)
        optim = torch.optim.Adam(wa.parameters())
        minloss = 0; minepoch = None
        for epoch in range(1, 1+epochs):
            avg_pool = torch.cat([ wa(item) for item in stack])
            avg_pool_sim = S.cosine_matrix(avg_pool, avg_pool)
            avg_pool_val = torch.cat([ wa(item) for item in stack_val])
            avg_pool_sim_val = S.cosine_matrix(avg_pool_val, avg_pool_val)
        
            loss = -S.pearson_r(S.triu(avg_pool_sim), S.triu(edit_sim))
            loss_val = -S.pearson_r(S.triu(avg_pool_sim_val), S.triu(edit_sim_val))
            print(layer, epoch, -loss.item(), -loss_val.item())
            if loss_val.item() <= minloss:
                minloss = loss_val.item()
                minepoch = epoch
            optim.zero_grad()
            loss.backward()
            optim.step()
        print("Maximum correlation on val: {} at epoch {}".format(-minloss, minepoch))
        
        
def learned_similarity(epochs=1, device='cpu'):
    stackdata = np.load("../talkie/state_stack_flickr8k_val.npy")
    trans = np.load("../talkie/transcription_flickr8k_val.npy")
    trans_val = trans[500:]
    trans = trans[:500]    
    edit_sim = torch.tensor(U.pairwise(S.stringsim, trans)).float().to(device)
    edit_sim_val = torch.tensor(U.pairwise(S.stringsim, trans_val)).float().to(device)
    for layer in range(4):
        stack = [ torch.tensor(item[:, layer, :]).float().to(device) for (i,item) in enumerate(stackdata) ]
        stack_val = stack[500:]
        stack = stack[:500]
        model = S.LearnedSimilarity(1024).to(device)
        print(model)
        optim = torch.optim.Adam(model.parameters())
        minloss = 0; minepoch = None
        print("layer epoch train_loss val_loss")
        for epoch in range(epochs):
            sim = torch.cat([ model(u, v) for (i, u) in enumerate(stack) for (j, v) in enumerate(stack) if i < j ])
            print(sim.size())
            sim_val = torch.cat([ model(u, v) for (i, u) in enumerate(stack) for (j, v) in enumerate(stack_val) if i < j ])
            
            loss = -S.pearson_r(sim, S.triu(edit_sim))
            loss_val = -S.pearson_r(sim_val, S.triu(edit_sim_val))
            print(layer, epoch, -loss.item(), -loss_val.item())
            if loss_val.item() <= minloss:
                minloss = loss_val.item()
                minepoch = epoch
            optim.zero_grad()
            loss.backward()
            optim.step()
        print("Maximum correlation on val: {} at epoch {}".format(-minloss, minepoch))
        
        
        

