import numpy as np
import torch
import torch.nn.functional as F
import imageio

import os
from skimage.draw import circle

import matplotlib.pyplot as plt
import collections

class Logger:
    def __init__(self, log_dir, batch_num, checkpoint_freq=50, visualizer_params=None, zfill_num=8, log_file_name='log.txt'):

        self.loss_list = []
        self.cpk_dir = log_dir
        self.visualizations_dir = os.path.join(log_dir, 'train-vis')
        if not os.path.exists(self.visualizations_dir):
            os.makedirs(self.visualizations_dir)
        self.log_file = open(os.path.join(log_dir, log_file_name), 'a')
        self.zfill_num = zfill_num
        self.visualizer = Visualizer(**visualizer_params)
        self.checkpoint_freq = checkpoint_freq
        self.epoch = 0
        self.best_loss = float('inf')
        self.names = None
        self.batch_num = batch_num
        self.savei = 0
        self.last_cpk_path=[]
        self.last_img_path=[]

    def log_scores(self, loss_names):
        loss_mean = np.array(self.loss_list).mean(axis=0)

        loss_string = "; ".join(["%s - %.5f" % (name, value) for name, value in zip(loss_names, loss_mean)])
        loss_string = str(self.epoch).zfill(self.zfill_num) + '/' + str(self.iter_num).zfill(self.zfill_num)+ '/' + str(self.batch_num).zfill(self.zfill_num) + ") " +str(self.data_load_time)+ '/'+str(self.batch_time)+")"+ loss_string

        print(loss_string, file=self.log_file) 
        print(loss_string)
        self.loss_list = []
        self.log_file.flush()

    def visualize_rec(self, inp, out):
        image = self.visualizer.visualize(inp['driving'], inp['source'], out)
        imageio.imsave(os.path.join(self.visualizations_dir, "%s_%s-rec.png" % (str(self.epoch).zfill(self.zfill_num),str(self.iter_num).zfill(self.zfill_num))), image)
        self.last_img_path.append(os.path.join(self.visualizations_dir, "%s_%s-rec.png" % (str(self.epoch).zfill(self.zfill_num),str(self.iter_num).zfill(self.zfill_num))))

    def save_cpk(self, emergent=False):
        
        cpk = {k: v.state_dict() for k, v in self.models.items()}
        cpk['epoch'] = self.epoch
        cpk['iter_num'] = self.iter_num
        cpk_path = os.path.join(self.cpk_dir, '%s_%s-checkpoint.pth.tar' % (str(self.epoch).zfill(self.zfill_num),str(self.iter_num).zfill(self.zfill_num))) 
        self.last_cpk_path.append(cpk_path)
        self.savei+=1
        if not (os.path.exists(cpk_path) and emergent):
            torch.save(cpk, cpk_path)

    @staticmethod
    def load_cpk(checkpoint_path, inpainting_network=None, dense_motion_network =None):
        checkpoint = torch.load(checkpoint_path)
        if inpainting_network is not None:
            inpainting_network.load_state_dict(checkpoint['inpainting_network'])
        if dense_motion_network is not None:
            dense_motion_network.load_state_dict(checkpoint['dense_motion_network'])
            

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if 'models' in self.__dict__:
            self.save_cpk()
        self.log_file.close()

    def log_iter(self, losses):
        losses = collections.OrderedDict(losses.items())
        self.names = list(losses.keys())
        self.loss_list.append(list(losses.values()))

    def log_epoch(self, data_load_time,batch_time, epoch, iter_num, models, inp, out):
        self.epoch = epoch
        self.iter_num = iter_num
        self.models = models
        self.data_load_time = data_load_time
        self.batch_time = batch_time

        self.log_scores(self.names)
        if self.iter_num % self.checkpoint_freq == 0: 
            self.save_cpk()
            self.visualize_rec(inp, out) 
        
        

class Visualizer:
    def __init__(self, kp_size=5, draw_border=False, colormap='gist_rainbow'):
        self.kp_size = kp_size
        self.draw_border = draw_border
        self.colormap = plt.get_cmap(colormap)

    def draw_image_with_kp(self, image, kp_array): 
        image = np.copy(image)
        spatial_size = np.array(image.shape[:2][::-1])[np.newaxis]
        kp_array = spatial_size * (kp_array + 1) / 2 
        num_kp = kp_array.shape[0]
        for kp_ind, kp in enumerate(kp_array):
            rr, cc = circle(kp[1], kp[0], self.kp_size, shape=image.shape[:2])
            image[rr, cc] = np.array(self.colormap(kp_ind / num_kp))[:3] 
        return image

    def create_image_column_with_kp(self, images, kp):
        image_array = np.array([self.draw_image_with_kp(v, k) for v, k in zip(images, kp)])
        return self.create_image_column(image_array)

    def create_image_column(self, images):
        if self.draw_border:  
            images = np.copy(images)
            images[:, :, [0, -1]] = (1, 1, 1)
        return np.concatenate(list(images), axis=0)

    def create_image_grid(self, *args):
        out = []
        for arg in args:
            if type(arg) == tuple:
                out.append(self.create_image_column_with_kp(arg[0], arg[1]))
            else:
                out.append(self.create_image_column(arg))
        return np.concatenate(out, axis=1)

    def visualize(self, driving, source, out):
        images = []

        source = source.data.cpu() 
        kp_source = out['kp_source']['fg_kp'].data.cpu().numpy()
        source = np.transpose(source, [0, 2, 3, 1])
        images.append((source, kp_source))
        

        # Equivariance visualization
        if 'transformed_frame' in out:
            transformed = out['transformed_frame'].data.cpu().numpy() 
            transformed = np.transpose(transformed, [0, 2, 3, 1])
            transformed_kp = out['transformed_kp']['fg_kp'].data.cpu().numpy()
            images.append((transformed, transformed_kp)) 

        kp_driving = out['kp_driving']['fg_kp'].data.cpu().numpy()
        driving = driving.data.cpu().numpy()
        driving = np.transpose(driving, [0, 2, 3, 1])
        images.append((driving, kp_driving))
        images.append(driving)

        if 'deformed' in out: 
            deformed = out['deformed'].data.cpu().numpy()
            deformed = np.transpose(deformed, [0, 2, 3, 1])
            images.append(deformed)

        prediction = out['prediction'].data.cpu().numpy() 
        prediction = np.transpose(prediction, [0, 2, 3, 1])
        if 'kp_norm' in out:  
            kp_norm = out['kp_norm']['fg_kp'].data.cpu().numpy()
            images.append((prediction, kp_norm))
        images.append(prediction)

        ## Occlusion map
        if 'occlusion_map' in out: 
            for i in range(len(out['occlusion_map'])):
                occlusion_map = out['occlusion_map'][i].data.cpu().repeat(1, 3, 1, 1)
                occlusion_map = F.interpolate(occlusion_map, size=source.shape[1:3]).numpy()
                occlusion_map = np.transpose(occlusion_map, [0, 2, 3, 1])
                images.append(occlusion_map)

        if 'deformed_source' in out: 
            full_mask = []
            for i in range(out['deformed_source'].shape[1]): 
                image = out['deformed_source'][:, i].data.cpu()
                image = F.interpolate(image, size=source.shape[1:3])  
                mask = out['contribution_maps'][:, i:(i+1)].data.cpu().repeat(1, 3, 1, 1)
                mask = F.interpolate(mask, size=source.shape[1:3])
                image = np.transpose(image.numpy(), (0, 2, 3, 1))
                mask = np.transpose(mask.numpy(), (0, 2, 3, 1))

                if i != 0:
                    color = np.array(self.colormap((i - 1) / (out['deformed_source'].shape[1] - 1)))[:3]
                else:
                    color = np.array((0, 0, 0))

                color = color.reshape((1, 1, 1, 3))

                images.append(image) 
                if i != 0:
                    images.append(mask * color) 
                else:
                    images.append(mask) 

                full_mask.append(mask * color)   

            images.append(sum(full_mask)) 

        image = self.create_image_grid(*images)
        image = (255 * image).astype(np.uint8)
        return image
