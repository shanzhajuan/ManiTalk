import numpy as np
import librosa
import os,argparse
from transformers import Wav2Vec2Processor
import torch
os.environ['PYOPENGL_PLATFORM'] = 'osmesa' # egl

from modules.faceformer import Faceformer
import matplotlib.pyplot as plt
@torch.no_grad()
class exp_generator():
    def __init__(self, config,device):
        self.config={}
        self.config['dataset'] = 'vocaset'
        self.config['feature_dim'] = 64
        self.config['period'] = 60
        self.config['vertice_dim'] = 68*3
        self.config['model_name'] = config.model_name
        self.config['condition'] = config.condition
        self.config['subject'] = config.subject
        self.config['wav2vec_model'] = config.wav2vec_model
        self.config['train_subjects'] = config.train_subjects
        self.config['device'] = device
        self.config['wav_dir'] = config.wav_dir
        

    def test_model(self):

        #build model
        model = Faceformer(self.config)

        model.load_state_dict(torch.load(self.config['model_name'], map_location='cuda'))
        model = model.to(torch.device(self.config['device']))
        model.eval()

        template_file = './data/template/'
        template_list = os.listdir(template_file)
        templates={}
        for file in template_list:
            te = np.load(template_file + file)
            templates[file[:-4]]=te 

        train_subjects_list = [i for i in self.config['train_subjects'].split(" ")]

        one_hot_labels = np.eye(len(train_subjects_list))
        iter = train_subjects_list.index(self.config['condition'])
        one_hot = one_hot_labels[iter]
        one_hot = np.reshape(one_hot,(-1,one_hot.shape[0]))
        one_hot = torch.FloatTensor(one_hot).to(self.config['device'])

        temp = templates[self.config['subject']]

        template = temp.reshape((-1))
        template = np.reshape(template,(-1,template.shape[0])) 
        template = torch.FloatTensor(template).to(self.config['device'])

        
        processor = Wav2Vec2Processor.from_pretrained(self.config['wav2vec_model'])
        wav_path = self.config['wav_dir']
        speech_array, sampling_rate = librosa.load(os.path.join(wav_path), sr=16000)   
        audio_feature = np.squeeze(processor(speech_array,sampling_rate=16000).input_values)
        audio_feature = np.reshape(audio_feature,(-1,audio_feature.shape[0]))
        audio_feature = torch.FloatTensor(audio_feature).to(self.config['device'])
        prediction = model.predict(audio_feature, template, one_hot)
        prediction = prediction.squeeze() # (seq_len, V*3)
        return np.reshape(prediction.detach().cpu().numpy(),(-1,68,3))

    def find_mutations(self,data, threshold):
        mutations = []
        changes=[]
        for i in range(1, len(data)-1):
            slope_diff = (data[i] - data[i-1]) / (i - (i-1))
            slope_diff1 = (data[i+1] - data[i]) / (i+1 - i)
            change = abs(slope_diff-slope_diff1)
            if change > threshold:
                mutations.append((i, change))
            changes.append(change)
        return mutations,changes

    def interp(self,lm,idx): 
        new_lm = np.zeros((idx[1]-idx[0]+1,68,3))
        new_lm[[0,-1]] = lm
        wrong_kp = idx[1]-idx[0]-1
        for i in range(wrong_kp):
            new_lm[i+1] = (lm[1]-lm[0])/(idx[1]-idx[0])*(i+1)+lm[0]
        return new_lm

    def postsmooth(self,kp):
        frame = np.shape(kp)[0]
        mutations,changes = self.find_mutations(kp[:,66,1]*1000,2)       
        if mutations==[]:
            return kp
        
        smo = []
        start = mutations[0][0] 
        for i in range(1,len(mutations)):
            if not start:
                start = mutations[i][0]
                continue
            tmp0 = mutations[i-1][0]
            tmp1 = mutations[i][0]
            if tmp1-tmp0>5:
                end = tmp0
                smo.append([start,end])
                start=tmp1

        if start:
            end = mutations[-1][0]
            smo.append([start,end])
        print(smo)
        for j in range(len(smo)):
            start = max(0, smo[j][0]-1)
            end = min(frame, smo[j][1]+1)
            kp[start:end+1] = self.interp(kp[[start,end]],[start,end])
        return kp

        



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='./log/74_model.pth')
    parser.add_argument("--wav2vec_model", type=str, default='./facebook/wav2vec-base-960') 
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--train_subjects", type=str, default="FaceTalk_170728_03272_TA FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA")
    parser.add_argument("--test_subjects", type=str, default="FaceTalk_170809_00138_TA FaceTalk_170731_00024_TA")

    parser.add_argument("--wav_dir", type=str, default="./data/audio/english.wav", help='path of the input audio signal')
    parser.add_argument("--result_path", type=str, default="./results/kp/", help='path of the predictions')
    parser.add_argument("--condition", type=str, default="FaceTalk_170904_00128_TA", help='select a conditioning subject from train_subjects')
    parser.add_argument("--subject", type=str, default="FaceTalk_170904_00128_TA", help='select a subject from test_subjects or train_subjects')
    args = parser.parse_args() 
    exp_ge = exp_generator(args,args.device)
    neutral_driving = exp_ge.test_model()
    neutral_driving =exp_ge.postsmooth(neutral_driving)
    # np.save(os.path.join(args.result_path, 'neutral_driving.npy'), neutral_driving)
