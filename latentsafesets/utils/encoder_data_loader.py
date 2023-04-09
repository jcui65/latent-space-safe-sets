import torchvision.transforms as transforms

from PIL import Image
import numpy as np
import os


class EncoderDataLoader:
    def __init__(self, params):
        #self.data_dir = os.path.join('/home/jianning/PycharmProjects/pythonProject6/latent-space-safe-sets','data_images', params['env'])
        if params['light']=='ls3':
            if params['datasetnumber']==1:
                self.data_dir = os.path.join('','data_imagesls3', params['env'])
            elif params['datasetnumber']==2:
                print('Enters the new dataset/dataset 2!')#I remember seeing this when training the 0202 encoder!!
                self.data_dir = os.path.join('','data_images', params['env'])
        else:
            self.data_dir = os.path.join('','data_images', params['env'])
        #self.data_dir = os.path.join('', 'data_images_relative', params['env'])#for using relative image
        self.frame_stack = params['frame_stack']
        self.env = params['env']
        self.n_images = len(os.listdir(self.data_dir)) // self.frame_stack#(len(os.listdir(self.data_dir))/2) // self.frame_stack#
        if params['env'] == 'Robot':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
                transforms.RandomRotation(20),
                transforms.ToTensor
            ])
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
        ])

    def sample(self, batch_size):#plan A
        idxs = np.random.randint(self.n_images, size=batch_size)
        ims = []
        if self.frame_stack == 1:
            template = os.path.join(self.data_dir, '%d.png')#sample an image in data_images/spb#global coordinate
            #template = os.path.join(self.data_dir, 'relative_%d.png')  # sample an image in data_images/spb#ego
        else:
            template = os.path.join(self.data_dir, '%d_%d.png')
            #template = os.path.join(self.data_dir, 'relative_%d_%d.png')
        for idx in idxs:
            if self.frame_stack == 1:
                im = Image.open(template % idx)
                im = self.transform(im)
                ims.append(im)
            else:
                stack = []
                for i in range(self.frame_stack):
                    im = Image.open(template % (idx, i))
                    stack.append(im)
                ims.append(stack)
        return ims

    def sample_cbf(self, batch_size,trajectories):#plan B
        idxs = np.random.randint(self.n_images, size=batch_size)
        ims = []
        dists=[]
        if self.frame_stack == 1:
            template = os.path.join(self.data_dir, '%d.png')#sample an image in data_images/spb
            #template = os.path.join(self.data_dir, 'relative_%d.png')  # sample an image in data_images/spb
        else:
            template = os.path.join(self.data_dir, '%d_%d.png')
            #template = os.path.join(self.data_dir, 'relative_%d_%d.png')
        for idx in idxs:
            if self.frame_stack == 1:
                im = Image.open(template % idx)
                im = self.transform(im)
                ims.append(im)
                seq=idx//100
                fra=idx%100
                traj=trajectories[seq]#the seqth trajectory (length 100)
                frame=traj[fra]#the frath frame
                hvo=frame['hvo']#hvo or hvn?# I think it is hvo
                hvo13=np.cbrt(hvo)
                #dists.append(frame['hvn'])
                dists.append(hvo13)
            else:
                stack = []
                for i in range(self.frame_stack):
                    im = Image.open(template % (idx, i))
                    stack.append(im)
                ims.append(stack)
                #add the distance if needed
        return ims,dists

