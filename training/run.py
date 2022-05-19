import numpy as np
import torch

from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from Dataset import OntologyDataset

import pickle
import csv
from model import DFALC
from Evaluation import MaskABox

import os
import click as ck
@ck.command()
@ck.option(
    '--info_path', default="input/", help='')
@ck.option(
    '--out_path', default="new_output/", help='')
@ck.option(
    '--save_path', default="new_output/", help='')
@ck.option(
    '--iter_path', default="ontologies/", help='')
@ck.option(
    '--mask_rate', default=0.0, help='')
@ck.option(
    '--alpha', default=0.8, help='')
@ck.option(
    '--learning_rate', default=2e-4, help='')
@ck.option(
    '--epoch_size', default=50000, help='')
@ck.option(
    '--batch_size', default=4, help='')
@ck.option(
    '--earlystopping', default=10, help='100 epochs as a unit')
@ck.option(
    '--train', default=True, help='')
@ck.option(
    '--model_name', default="Godel", help='')
@ck.option(
    '--device_name', default="cpu", help='')

def main(info_path, out_path, save_path, iter_path, mask_rate, alpha, learning_rate, epoch_size, batch_size, earlystopping, train, model_name, device_name):
    logs = [("step","loss")]
    device = torch.device(device_name)

    if train:
        out_path = os.path.join(out_path,"mask_" + str(mask_rate)+"/")
        save_path = os.path.join(save_path,"mask_" + str(mask_rate)+"/")
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        
            
        for file_idx, file_name in enumerate(os.listdir(iter_path)):
            print(file_name)
            if os.path.exists(os.path.join(out_path,file_name+".cEmb.pkl")):
                continue
            
            params = {
                "conceptPath": os.path.join(info_path,file_name+"_concepts.txt"),
                "allconceptPath": os.path.join(info_path,file_name+"_class.txt"),
                "rolePath": os.path.join(info_path,file_name+"_roles.txt"),
                "individualPath": os.path.join(info_path,file_name+"_individuals.txt"),
                "normalizationPath": os.path.join(info_path,file_name+"_normalization.txt"),
                "annotationPath": os.path.join(info_path,file_name+"_annotation.txt"),
                "aboxPath":os.path.join(info_path,file_name+"_abox.txt"),
                "batchSize": batch_size,
                "epochSize": epoch_size,
                "earlystopping": earlystopping,

            }

            data = OntologyDataset(params,save_path+file_name)
            
            eva = MaskABox(params["aboxPath"], data.concept2id, data.role2id, data.individual2id,alpha=alpha,mask_rate=mask_rate,save_path=save_path+file_name)
            eva_log = open(out_path+file_name+".evaluation.txt","w")
            eva_log.write("Initial masked cEmb MSE loss: {}\n".format(eva.MSE(eva.true_cEmb,eva.masked_cEmb)))
            eva_log.write("Initial masked rEmb MSE loss: {}\n".format(eva.MSE(eva.true_rEmb,eva.masked_rEmb)))
            print("Initial masked cEmb MSE loss: {}\n".format(eva.MSE(eva.true_cEmb,eva.masked_cEmb)))
            print("Initial masked rEmb MSE loss: {}\n".format(eva.MSE(eva.true_rEmb,eva.masked_rEmb)))

            model = DFALC(params, data.conceptSize, data.roleSize, eva.masked_cEmb, eva.masked_rEmb, device, name=model_name).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,  threshold=0.0001)

            for mode in range(7):
                data.mode = mode
                print(mode, len(data))
                
            nepoch =0
            best_loss = 1e9
            for nepoch in range(int(params["epochSize"])):
                stable_iter = 0
                tr_loss, nb_tr_steps = 0,0
                losses = []
                
                for mode in range(7):
                    data.mode = mode
                    if len(data) == 0: continue
                    train_dataloader = DataLoader(data, sampler = RandomSampler(data), batch_size = params["batchSize"])
                
                    for bid, batch in enumerate(train_dataloader):
                        ptriplets = [b.to(device) for b in batch]
                        loss = model(ptriplets, mode, device)
                        losses.append(loss)
                            
                loss = losses[0]
                for i in range(1,len(losses)):
                    loss += losses[i]
                    # print(i,losses[i])
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                tr_loss += loss.item()
                nb_tr_steps += 1

                optimizer.step()

                logs.append((nepoch,tr_loss/nb_tr_steps))

                if nepoch % 100==0:
                    print(nepoch, tr_loss/nb_tr_steps)
               
                    if best_loss>tr_loss/nb_tr_steps:
                        best_loss = tr_loss/nb_tr_steps
                        n_stop = 0
                    else:
                        n_stop += 1
                    if n_stop >= params["earlystopping"]:
                        break

            pickle.dump(model.cEmb.cpu().detach().numpy(),open(os.path.join(out_path,file_name+".cEmb.pkl"),"wb"))
            pickle.dump(model.rEmb.cpu().detach().numpy(),open(os.path.join(out_path,file_name+".rEmb.pkl"),"wb"))

            
            # torch.save(model.state_dict(),os.path.join(out_path,".model.pkl"))

            with open(os.path.join(out_path,file_name+'_losses.csv'), 'w') as f:
                writer= csv.writer(f)
                writer.writerows(logs)

            print("After training: ", eva.MSE(eva.true_cEmb,model.cEmb.cpu().detach().numpy()))
            print("After training: ", eva.MSE(eva.true_rEmb,model.rEmb.cpu().detach().numpy()))
            eva_log.write("loss: {}\n".format(tr_loss/nb_tr_steps))
            eva_log.write("After training: {}\n".format(eva.MSE(eva.true_cEmb,model.cEmb.cpu().detach().numpy())))
            eva_log.write("After training: {}\n".format(eva.MSE(eva.true_rEmb,model.rEmb.cpu().detach().numpy())))
            eva_log.flush()
            eva_log.close()
    
if __name__ == "__main__":
    main()