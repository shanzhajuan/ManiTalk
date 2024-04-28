import numpy as np
import os
import librosa
import soundfile as sf
import cv2
from tqdm import tqdm
import subprocess
import shutil 

def img2video(save_dir,audio_dir=None):
    tmp_audio_path = save_dir + '/tmp.wav'
    tmp_video_dir = save_dir + '/tmp.avi'
    final_path = save_dir +'/prediction.avi'
    
    img_dir = os.path.join(save_dir, 'tmp/')
    img_list = os.listdir(img_dir)
    nframe = np.shape(img_list)[0]
    FPS = 60
    
    if audio_dir is not None:       
        audio, sr = librosa.load(audio_dir, sr=16000)
        tmp_audio_clip = audio[ : np.int32(nframe * sr / FPS)]
        sf.write(tmp_audio_path, tmp_audio_clip, sr)


    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    if audio_dir is not None:
        out = cv2.VideoWriter(tmp_video_dir, fourcc, FPS, (256, 256))
        for j in tqdm(range(nframe), position=0, desc='writing video'):
            img = cv2.imread(os.path.join(img_dir, str(j) + '.png'))
            out.write(img)
        out.release() 
        cmd = 'ffmpeg -i "' + tmp_video_dir + '" -i "' + tmp_audio_path + '" -codec copy -shortest "' + final_path + '"'
        subprocess.call(cmd, shell=True) 
        
    else:
        out = cv2.VideoWriter(final_path, fourcc, FPS, (256, 256))
        for j in tqdm(range(nframe), position=0, desc='writing video'):
            img_list.sort()
            img = cv2.imread(os.path.join(img_dir, img_list[j]))
            out.write(img)
        out.release() 

    if os.path.exists(tmp_audio_path):
        os.remove(tmp_audio_path)
    if os.path.exists(tmp_video_dir):
        os.remove(tmp_video_dir)
    shutil.rmtree(img_dir,ignore_errors = False,onerror = None)
    

if __name__ == "__main__":
    save_dir = './results/'
    audio_dir = './data/audio/english.wav'
    img2video(save_dir,audio_dir)

