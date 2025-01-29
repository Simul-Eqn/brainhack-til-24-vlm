# code adapted from: https://github.com/openai/CLIP/issues/83 

import clip 
import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader 
from PIL import Image 

import albumentations as A 
import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt 

EPOCH = 4 
BATCH_SIZE = 16 


transform = A.Compose([
    A.HorizontalFlip(p=0.5), 
    A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1)), 
    #A.RandomRain(), 
])


# Latest Update : 18 July 2022, 09:55 GMT+7

# TO ADD :
# Gradient Checkpointing
# Filter out bias from weight decay
# Decaying learning rate with cosine schedule
# Half-precision Adam statistics
# Half-precision stochastically rounded text encoder weights were used

#BATCH_SIZE must larger than 1

device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
model, preprocess = clip.load("ViT-L/14",device=device,jit=False) #Must set jit=False for training

class image_title_dataset(Dataset):
    surrounding_pixels = 8 

    def __init__(self):
        self.vlmdata = pd.read_json('../../train_data/vlm.jsonl', lines=True) 
        self.length = 0 
        self.poss = [] 
        for i in range(len(self.vlmdata)): 
           self.poss.append( self.length ) 
           self.length += len(self.vlmdata.loc[i].annotations) 
        self.poss.append(self.length) 
        
        self.next_i = 0 

    def __len__(self):
        return self.length 

    def __getitem__(self, in_idx):
        idxset = False 
        if (self.poss[self.next_i] <= in_idx): 
            if (in_idx < self.poss[self.next_i+1]): 
                idx = self.next_i 
                idxset = True 
            elif (in_idx < self.poss[self.next_i+2]): 
               self.next_i += 1 
               idx = self.next_i 
               idxset = True 
        
        if (not idxset): 
            # binary search out idx 
            left=0 
            right=len(self.vlmdata)-1 
            
            while (right-left > 1): 
                mid = (left+right)//2 
                if (self.poss[mid] > in_idx): 
                    right = mid 
                else: 
                   left = mid 

            if (in_idx > self.poss[right]): 
               self.next_i = right 
               idx = right 
            else: 
               self.next_i = left 
               idx = left 

        

        image = Image.open("../../train_data/images/"+self.vlmdata.loc[idx].image)

        crop_bbox = self.get_crop_bbox(*self.vlmdata.loc[idx].annotations[in_idx-self.poss[self.next_i]]['bbox'], *image.size)

        image = image.crop(crop_bbox) 
        image = transform(image=np.array(image))['image']

        '''plt.figure(figsize=(20,10)) 
        plt.axis('off') 
        plt.imshow(image)
        plt.show() 
        print(image) ''' 

        image = preprocess(Image.fromarray(image)) # Image from PIL module

        caption = self.vlmdata.loc[idx].annotations[in_idx-self.poss[self.next_i]]['caption'] 
        return image, clip.tokenize(caption)[0] 
    
    def get_crop_bbox(self, x,y,w,h, maxx, maxy): 
        a = max(0, x-image_title_dataset.surrounding_pixels) 
        b = max(0, y-image_title_dataset.surrounding_pixels)
        c = min(maxx, x+w+image_title_dataset.surrounding_pixels) 
        d = min(maxy, y+h+image_title_dataset.surrounding_pixels)
        return (a, b, c, d) 

# use your own data
dataset = image_title_dataset()
train_dataloader = DataLoader(dataset,batch_size = BATCH_SIZE) #Define your own dataloader



#https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 


if device == "cpu":
  model.float()
else :
  clip.model.convert_weights(model) # Actually this line is unnecessary since clip by default already on float16

loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-7,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

print("DEVICE:", device)

# add your own code to track the training progress.
for epoch in range(EPOCH):
    print("EPOCH", epoch)
    for batch in train_dataloader :
        #print("batch")
        optimizer.zero_grad()

        images, texts = batch 

        #print("IMAGES:", images) 
        #print("TEXTS:", texts)
        
        images = images.to(device)
        texts = texts.to(device)
        
        logits_per_image, logits_per_text = model(images, texts)

        ground_truth = torch.arange(len(images),dtype=torch.long,device=device)

        total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
        total_loss.backward()
        if device == "cpu":
            optimizer.step()
        else : 
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)



    # save model 

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_loss,
        }, "model_checkpoint/model_"+str(epoch)+".pt") #just change to your preferred folder/filename

