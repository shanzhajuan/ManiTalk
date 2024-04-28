import os
import torch
import numpy as np
import imageio
from skimage.transform import resize
from scipy.spatial import ConvexHull
import yaml
from argparse import ArgumentParser
from skimage import io
from functools import partial
from scipy.interpolate import interp1d

from modules.logger import Logger
from modules.inpainting_network import InpaintingNetwork
from modules.dense_motion import DenseMotionNetwork
from modules.exp import exp_generator
from utils.mediapipe_detector import get_img_pt
from utils.align import Align
from utils.syn_animation import img2video

def get_ptid_eyeball():  
    pt_idx1 = [0,2,4,6,8]
    pt_idx2 = [10,12,14,16,25]
    pt_idx3 = [27,30,31,33,35]
    pt_idx4 = [18,20,23,28,29]
    pt_idx5 = [36,38,39,41,3] 
    pt_idx6 = [42,43,46,45,13]
    pt_idx7 = [48,49,51,5,11] 
    pt_idx8 = [54,55,57,58,7] 
    pt_idx9 = [62,66,1,15,9]  
    pt_idx10 = [52,56,59,68,69] 
    pt_idx = pt_idx1 + pt_idx2 + pt_idx3 + pt_idx4 + pt_idx5 + pt_idx6 +pt_idx7 + pt_idx8 +pt_idx9 + pt_idx10
    return pt_idx

eye_brow_idx =[i for i in range(36,48)] +[(i) for i in range(17,27)]
l0 = 1/3 
l180 = 3/4
neutral_l = (l0+l180)/2
r0 = 1/4
r180 = 2/3
neutral_r = (r0+r180)/2

l90 = 0 
l270 = 1
neutral_l2 = (l90+l270)/2
eyelid_idx = [43,44,37,38]
def relative_kp(kp_source, kp_driving, kp_driving_initial,kp_driving_initial2, theta, rho):
    source_area = ConvexHull(kp_source[0].cpu().numpy()).volume 
    driving_area = ConvexHull(kp_driving_initial['fg_kp'][0].cpu().numpy()).volume 
    adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    kp_new = {}
    kp_value_diff = (kp_driving['fg_kp'] - kp_driving_initial['fg_kp']) * adapt_movement_scale # frame,70,2
    kp_value_diff[:,eye_brow_idx,:] = (kp_driving['fg_kp'][:,eye_brow_idx,:] - kp_driving_initial2['fg_kp'][:,eye_brow_idx,:]) * adapt_movement_scale
    kp_new['fg_kp'] = kp_value_diff + kp_source # frame,70,2

    ##### manipulate eyeball #####
    if rho!=0:       
        if theta==0:
            l_coeff = l0*rho + (1-rho)*neutral_l
            r_coeff = r0*rho + (1-rho)*neutral_r
            kp_new['fg_kp'][:,-2] = kp_new['fg_kp'][:,36]*l_coeff+ kp_new['fg_kp'][:,39]*(1-l_coeff)  
            kp_new['fg_kp'][:,-1] = kp_new['fg_kp'][:,42]*r_coeff+ kp_new['fg_kp'][:,45]*(1-r_coeff)
        if theta==180:
            l_coeff = l180*rho + (1-rho)*neutral_l
            r_coeff = r180*rho + (1-rho)*neutral_r
            kp_new['fg_kp'][:,-2] = kp_new['fg_kp'][:,36]*l_coeff+ kp_new['fg_kp'][:,39]*(1-l_coeff)  
            kp_new['fg_kp'][:,-1] = kp_new['fg_kp'][:,42]*r_coeff+ kp_new['fg_kp'][:,45]*(1-r_coeff)
        if theta==90:
            kp_new['fg_kp'][:,eyelid_idx,1] -= 0.005
            l_coeff = l90*rho + (1-rho)*neutral_l2
            kp_new['fg_kp'][:,-2] = kp_new['fg_kp'][:,41]*l_coeff + kp_new['fg_kp'][:,37]*(1-l_coeff)
            kp_new['fg_kp'][:,-2,0] = (kp_new['fg_kp'][:,41,0]+kp_new['fg_kp'][:,38,0])/2
        
            kp_new['fg_kp'][:,-1] = kp_new['fg_kp'][:,46]*l_coeff + kp_new['fg_kp'][:,43]*(1-l_coeff)
            kp_new['fg_kp'][:,-1,0] = (kp_new['fg_kp'][:,46,0]+kp_new['fg_kp'][:,43,0])/2
        if theta==270:
            l_coeff = l270*rho + (1-rho)*neutral_l2
            kp_new['fg_kp'][:,-2] = kp_new['fg_kp'][:,41]*l_coeff + kp_new['fg_kp'][:,37]*(1-l_coeff)
            kp_new['fg_kp'][:,-2,0] = (kp_new['fg_kp'][:,41,0]+kp_new['fg_kp'][:,38,0])/2
        
            kp_new['fg_kp'][:,-1] = kp_new['fg_kp'][:,46]*l_coeff + kp_new['fg_kp'][:,43]*(1-l_coeff)
            kp_new['fg_kp'][:,-1,0] = (kp_new['fg_kp'][:,46,0]+kp_new['fg_kp'][:,43,0])/2
            # kp_new['fg_kp'][:,eyelid_idx,1] += 0.01
    return kp_new # frame,70,2


def demo_animation(inpainting_network, dense_motion_network, checkpoint, source, source_kp, driving_kp_list, initial_frame, save_dir, device,theta,rho):

    Logger.load_cpk(checkpoint, inpainting_network=inpainting_network, dense_motion_network=dense_motion_network)    
    inpainting_network.eval()
    dense_motion_network.eval()
    pt_idx50 = get_ptid_eyeball()
     
    with torch.no_grad():        
        resize_fn = partial(resize,output_shape=(256,256,3))
        source = resize_fn(source)
        source = source.astype(np.float32).transpose((2, 0, 1))[np.newaxis,:]  # 1,3,256,256
        source = torch.tensor(source).to(device)

        source_kp = (source_kp.astype(np.float32))[np.newaxis,:,:]*2-1  #1,70,3
        kp_source = {}
        kp_source['fg_kp'] = torch.tensor(source_kp[:,pt_idx50,:2]).to(device) #1,50,2
        
        driving_kp_list[:,:,:2] = np.clip(driving_kp_list[:,:,:2].astype(np.float32),0,1)*2-1  # frame,68,3
        # set static eyeball
        l_eyeball = (driving_kp_list[:,36]/4+ driving_kp_list[:,39]/4*3)[:,np.newaxis,:]
        r_eyeball = (driving_kp_list[:,42]/4+ driving_kp_list[:,45]/4*3)[:,np.newaxis,:]
        driving_kp_list = np.concatenate((driving_kp_list,l_eyeball,r_eyeball),1)  # frame,70,2 

        kp_driving_initial = {}
        kp_driving_initial['fg_kp'] = driving_kp_list[initial_frame][np.newaxis,:]  #  1,70,2  
        kp_driving_initial['fg_kp'] = torch.tensor(kp_driving_initial['fg_kp']).to(device)
        kp_driving_initial2 = {}
        kp_driving_initial2['fg_kp'] = driving_kp_list[0][np.newaxis,:]  #  1,70,2  
        kp_driving_initial2['fg_kp'] = torch.tensor(kp_driving_initial2['fg_kp']).to(device)
        
        for i in range(np.shape(driving_kp_list)[0]):
            frame_name = str(i)
            print(frame_name)                     
            kp_driving = {} 
            kp_driving['fg_kp'] = torch.tensor(driving_kp_list[i][np.newaxis,:]).to(device) # 1,70,2            
            kp_norm = relative_kp(torch.tensor(source_kp).to(device), kp_driving, kp_driving_initial, kp_driving_initial2,theta,rho)
            kp_norm['fg_kp'] = kp_norm['fg_kp'][:,pt_idx50,:2]  # 1,50,2

            dense_motion = dense_motion_network(source_image=source, kp_driving=kp_norm,
                                                kp_source=kp_source, bg_param = None, 
                                                dropout_flag = False)
            out = inpainting_network(source, dense_motion)

            imageio.imsave(os.path.join(save_dir, frame_name+'.png'), (255 * out['prediction'][0].cpu().numpy().transpose(1,2,0)).astype(np.uint8))

                           
def main(opt):
    img_name = os.path.basename(opt.source_image_dir)[:-4]
    wav_name = os.path.basename(opt.wav_dir)[:-4]
    save_dir = os.path.join(opt.save_dir,img_name+'_'+wav_name)
    os.makedirs(save_dir,exist_ok=True)
    tmp_save_dir = os.path.join(save_dir,'kp/')
    os.makedirs(tmp_save_dir,exist_ok=True)
    tmp_img_dir = os.path.join(save_dir, 'tmp/')
    os.makedirs(tmp_img_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ############# get source landmark ###############
    pt_detector = get_img_pt()
    source = np.array(io.imread(opt.source_image_dir))
    source_kp = pt_detector.get_frame_pt(source)  # 70,3
    np.save(os.path.join(tmp_save_dir,'source_kp.npy'),source_kp)

    ############# get driving landmark from audio ###############
    exp_ge = exp_generator(opt,device)
    neutral_driving = exp_ge.test_model()
    # post-process-smooth
    neutral_driving =exp_ge.postsmooth(neutral_driving)
    np.save(os.path.join(tmp_save_dir,'neutral_driving.npy'),neutral_driving)

    ############# get head pose from audio ############### 
    # coming soon, Users can use lsp's pose generation module as an alternative
    # Live speech portraits: real-time photorealistic talking-head animation: https://github.com/YuanxunLu/LiveSpeechPortraits
    head_pose=None

    # affline transform 
    aligner = Align()
    driving_kp_list, initial_frame = aligner.align_kp(source_kp, neutral_driving, opt.v_blink, opt.v_brow, opt.a_brow, head_pose)
    driving_kp_list = np.array(driving_kp_list,dtype=np.float32)
    np.save(os.path.join(tmp_save_dir,'driving_kp_list.npy'),driving_kp_list) 

    ############# SFWNET ############### 
    # load network
    log_dir = os.path.join(*os.path.split(opt.checkpoint)[:-1])
    with open(log_dir+'/vox-256.yaml') as f:
        config = yaml.load(f)
    inpainting = InpaintingNetwork(**config['model_params']['generator_params'],
                                   **config['model_params']['common_params'])
    dense_motion_network = DenseMotionNetwork(**config['model_params']['common_params'],
                                              **config['model_params']['dense_motion_params'])      
    inpainting.to(device)
    dense_motion_network.to(device)
    demo_animation(inpainting, dense_motion_network, opt.checkpoint, source, source_kp, driving_kp_list, initial_frame, tmp_img_dir, device,opt.theta,opt.rho)

    ############# syn animation ###############
    print('syn animation...')
    img2video(save_dir,opt.wav_dir)
    print('well done!')

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default='./log/00000102_00101000-checkpoint.pth.tar')
    parser.add_argument("--source_image_dir", type=str, default='./data/image/blown.png')
    parser.add_argument("--wav_dir", type=str, default='./data/audio/chinese.wav')
    parser.add_argument("--save_dir", type=str, default='./results')
    ## control parameters
    parser.add_argument("--v_blink", type=float, default=0.5, help='value from 0 to 1')
    parser.add_argument("--v_brow", type=float, default=0.5, help='value from 0 to 1')
    parser.add_argument("--a_brow", type=float, default=0, help='value from -1 to 1')
    # Currently only 4 gaze directions are supported. 
    # To generate other directions, refer to the paper to calculate the pupil position based on the coordinates of the eyelid points.
    parser.add_argument("--theta", type=int, default=0, choices=[0,90,180,270]) 
    parser.add_argument("--rho", type=int, default=0, help='value from 0 to 1') 

    ## exp generator params
    parser.add_argument("--model_name", type=str, default='./log/74_model.pth') 
    parser.add_argument("--wav2vec_model", type=str, default='./facebook/wav2vec-base-960') 
    parser.add_argument("--train_subjects", type=str, default="FaceTalk_170728_03272_TA FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA")
    parser.add_argument("--test_subjects", type=str, default="FaceTalk_170809_00138_TA FaceTalk_170731_00024_TA")
    parser.add_argument("--result_path", type=str, default="./results/", help='path of the predictions')
    parser.add_argument("--condition", type=str, default="FaceTalk_170904_00128_TA", help='select a conditioning subject from train_subjects')
    parser.add_argument("--subject", type=str, default="FaceTalk_170904_00128_TA", help='select a subject from test_subjects or train_subjects')
    
    opt = parser.parse_args()
    main(opt)

    

