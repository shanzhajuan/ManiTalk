import cv2
import mediapipe as mp
import numpy as np 
from scipy.ndimage import gaussian_filter1d

class get_img_pt():
    def __init__(self): 
        super(get_img_pt, self).__init__()
        self._PRESENCE_THRESHOLD = 0.5
        self._VISIBILITY_THRESHOLD = 0.5
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True,  
                                        max_num_faces=1,         
                                        refine_landmarks=True,
                                        min_detection_confidence=0.5, 
                                        min_tracking_confidence=0.5) 


    def get_frame_pt(self,img):
        _img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(_img)
        landmarks = results.multi_face_landmarks 
        if landmarks:
            for face in results.multi_face_landmarks:
                array_pt = self.landmark2array(face)  # 470,3
                array_pt = self.get68pt(array_pt)  # 70,3
                array_pt[:,:2] = np.clip(array_pt[:,:2], 0, 1)
                assert np.shape(array_pt)==(68+2,3), 'landmark detect wrong!!!'
                break
        else:
            print('wrong!!!')
            exit(0)  
        return array_pt

    def get_video_pt(self,video_dir):        
        cap = cv2.VideoCapture(video_dir)        
        fps = cap.get(cv2.CAP_PROP_FPS) 
        head_kp_list = []
        while True:
            if cap.grab():
                flag, frame = cap.retrieve() 
                if not flag:
                    continue
                else:
                    array_pt = self.get_frame_pt(frame)
                    head_kp_list.append(array_pt)
            else:
                cap.release()
                break
        head_kp_list=np.array(head_kp_list)
        head_kp_list = self.landmark_smooth_3d(head_kp_list,1)
        return head_kp_list,fps
    
    def landmark_smooth_3d(self, pts, smooth_sigma=1):
        ''' 
            pts3d: [N, 70, 3]
        '''
        n, x, y = np.shape(pts)
        if not smooth_sigma == 0:
            pts3d = gaussian_filter1d(pts.reshape(-1, x*y), smooth_sigma, axis=0).reshape(-1, x, y)
        return pts3d 
    
    
    def landmark2array(self,landmark_list):
        array_lm = []
        for landmark in landmark_list.landmark:
            if ((landmark.HasField('visibility') and landmark.visibility < self._VISIBILITY_THRESHOLD) or (landmark.HasField('presence') and landmark.presence < self._PRESENCE_THRESHOLD)):
                print('something wrong!!')
            landmark_px = [landmark.x, landmark.y, landmark.z]
            array_lm.append(landmark_px) 
        array_lm=np.array(array_lm) 
        return array_lm

    def get68pt(self,pt):
        face = [[127,127],[234,234],[93,132],[132,58],[58,172],[136,136],[150,149],
                [176,148],[152,152],[377,400],[378,379],[365,365],[288,397],[361,288],[323,361],[454,454],[356,356]]
        eyebrow = [[70,46],[63,53],[105,52],[66,65],[107,55],[336,285],[295,296],[282,334],[283,293],[300,276]]
        nose = [[168,168],[6,197],[195,195],[4,5],[219,219],[239,239],[19,19],[459,459],[439,439]]
        eye = [[33,33],[160,160],[158,158],[133,133],[153,153],[144,144],[362,362],
               [385,385],[387,387],[263,263],[373,373],[380,380]]
        mouth = [[61,61],[39,40],[37,37],[0,0],[267,267],[269,270],[291,291],[321,321],[405,314],[17,17],
                 [84,181],[91,91],[62,62],[81,81],[13,13],[311,311],[292,292],[402,402],[14,14],[178,178]]
        eyeball = [[468,468],[473,473]]
        idx = face + eyebrow + nose + eye +mouth+ eyeball
        new_pt = []
        for [start,end] in idx:
            new_pt.append((pt[start]+pt[end])/2)
        new_pt = np.array(new_pt)
        return new_pt






