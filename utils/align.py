import numpy as np
import math 
from sklearn.metrics import mean_squared_error

class Align():
    def __init__(self): 
        super(Align, self).__init__()
        self.template_eye1 = np.load('./data/template/random_blink_change.npy') 
        self.template_eye2 = np.load('./data/template/random_eye_change.npy') 
        self.template_brow = np.load('./data/template/eyebrow_up_change.npy')[:,-10:,:]*(-1) 
        self.frame1 = np.shape(self.template_eye1)[0]
        self.frame2 = np.shape(self.template_eye2)[0]
        self.frame3 = np.shape(self.template_brow)[0]
        
        self.eye_ptid = [i for i in range(36,48)]
        self.mouth_ptid = [(i) for i in range(48,68)]
        self.brow_idx = [(i) for i in range(17,27)]  
        self.blink_thred = 270
        self.brow_thred = 270


    def make_eye_anim(self,frame_num,v_blink=0.5):       
        random_eye = self.template_eye1
        if v_blink!=1:
            no_blink_length=int((1-v_blink)*(self.blink_thred-self.frame1))   
            num = no_blink_length//self.frame2
            remainder = no_blink_length % self.frame2
            for j in range(num):
                random_eye = np.concatenate((random_eye, self.template_eye2),0)
            if remainder:
                tmp_length = remainder//2
                tmp_template_eye = np.concatenate((self.template_eye2[:tmp_length], self.template_eye2[:tmp_length][::1]),0)
                random_eye = np.concatenate((random_eye, tmp_template_eye),0)
            if remainder%2:
                random_eye = np.concatenate((random_eye, self.template_eye2[0:1]),0)

        eye_change = np.zeros((frame_num,12,3))
        length = np.shape(random_eye)[0]        
        for i in range(frame_num // length):
            eye_change[i*length:(i+1)*length] = random_eye

        k = frame_num % length 
        for i in range(k // self.frame2):
            eye_change[-k+i*self.frame2:(i+1)*self.frame2-k] = self.template_eye2
        if k%self.frame2:
            eye_change[-(k%self.frame2):] = self.template_eye2[:(k%self.frame2)]    
        return eye_change
    
    def make_brow_anim(self,frame_num,v_brow=0.5,a_brow=0):   
        brow_change = np.zeros((frame_num,10,3))        
        if a_brow!=0 :
            template_brow = self.template_brow*a_brow
            random_brow = template_brow
            if v_brow!=1:
                no_brow_length=int((1-v_brow)*(self.brow_thred-self.frame3))   
                static_brow = np.repeat(template_brow[0][np.newaxis,:,:],no_brow_length,0)
                if no_brow_length<=self.frame3:
                    random_brow = np.concatenate((static_brow,random_brow),0)
                else:
                    random_brow = np.concatenate((static_brow[:self.frame3],random_brow,static_brow[self.frame3:]),0)
            
            length = np.shape(random_brow)[0]        
            for i in range(frame_num // length):
                brow_change[i*length:(i+1)*length] = random_brow
            k = frame_num % length 
            if k!=0:
                brow_change[-k:] = random_brow[:k]  
        return brow_change
    
    
    def align_kp(self,source_kp, neutral_driving, v_blink=0.5,v_brow=0.5, a_brow=0, head_pose=None): 
        # manipulate blink and eyebrow
        nframe = np.shape(neutral_driving)[0]
        eye_change = self.make_eye_anim(nframe,v_blink)
        neutral_driving[:,self.eye_ptid] = neutral_driving[0,self.eye_ptid] + eye_change*0.6
        brow_change = self.make_brow_anim(nframe, v_brow, a_brow)
        neutral_driving[:,self.brow_idx] = neutral_driving[0,self.brow_idx] + brow_change

        # add head pose
        if head_pose is not None:
            while np.shape(head_pose)[0]<np.shape(neutral_driving)[0]:
                head_pose2 = head_pose[::-1]
                head_pose = np.concatenate((head_pose,head_pose2),axis=0)        
            for i in range(np.shape(neutral_driving)[0]):
                rot = self.rotation_angles_to_matrix(head_pose[i,:3]/180*math.pi)
                neutral_driving[i] = neutral_driving[i].dot(rot.T)

        affline_driving, neutral_frame = self.trans_mouth(source_kp, neutral_driving)
        return affline_driving, neutral_frame

    def pts_dis(self,ps1,ps2):
        mse = mean_squared_error(ps1,ps2)
        return mse

    def trans_mouth(self,source_kp, neutral_driving): 
        frames = np.shape(neutral_driving)[0]
        dis = {}
        for i in range(frames):
            _, trans_pt, _ = self.procrustes(source_kp[self.mouth_ptid], neutral_driving[i,self.mouth_ptid])  
            dis[i] = self.pts_dis(trans_pt[:,:2],source_kp[self.mouth_ptid,:2])
        min_dis_mouth = min(zip(dis.values(), dis.keys()))

        _, _, tform_lm = self.procrustes(source_kp[self.mouth_ptid], neutral_driving[min_dis_mouth[1],self.mouth_ptid])
        rot=tform_lm['rotation']
        trans=tform_lm['translation']
        scale=tform_lm['scale']
        affline_driving= np.matmul(neutral_driving,rot)*scale + trans 
        return affline_driving, min_dis_mouth[1]
    
    def procrustes(self,X, Y, scaling=True, reflection='best'): # Y TO X
        n,m = X.shape
        ny,my = Y.shape
        muX = X.mean(0)
        muY = Y.mean(0)
        X0 = X - muX
        Y0 = Y - muY
        ssX = (X0**2.).sum()
        ssY = (Y0**2.).sum()
        # centred Frobenius norm
        normX = np.sqrt(ssX)
        normY = np.sqrt(ssY)
        # scale to equal (unit) norm
        X0 /= normX
        Y0 /= normY
        if my < m:
            Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)
        # optimum rotation matrix of Y
        A = np.dot(X0.T, Y0)
        U,s,Vt = np.linalg.svd(A,full_matrices=False)
        V = Vt.T
        T = np.dot(V, U.T)
        if reflection is not 'best':
            # does the current solution use a reflection?
            have_reflection = np.linalg.det(T) < 0
            # if that's not what was specified, force another reflection
            if reflection != have_reflection:
                V[:,-1] *= -1
                s[-1] *= -1
                T = np.dot(V, U.T)
        traceTA = s.sum()
        if scaling:
            # optimum scaling of Y
            b = traceTA * normX / normY
            # standarised distance between X and b*Y*T + c
            d = 1 - traceTA**2
            # transformed coords
            Z = normX*traceTA*np.dot(Y0, T) + muX
        else:
            b = 1
            d = 1 + ssY/ssX - 2 * traceTA * normY / normX
            Z = normY*np.dot(Y0, T) + muX
        # transformation matrix
        if my < m:
            T = T[:my,:]
        c = muX - b*np.dot(muY, T)
        #transformation values 
        tform = {'rotation':T, 'scale':b, 'translation':c}
        return d, Z, tform
    
    def rotation_angles_to_matrix(self,theta) :  # rad 
        R_x = np.array([[1,                  0,                   0],
                        [0, math.cos(theta[0]), -math.sin(theta[0])],
                        [0,  math.sin(theta[0]), math.cos(theta[0]) ]])
                            
        R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                        [0,                  1,                  0],
                        [-math.sin(theta[1]), 0, math.cos(theta[1])]])
                    
        R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                        [math.sin(theta[2]), math.cos(theta[2]),  0],
                        [0,                  0,                   1]])                                            
        R = np.dot(R_z, np.dot( R_y, R_x ))
        return R

if __name__ == "__main__":

    identity_name = 'blown'
    lm_name = 'english'
    source_kp = np.array(np.load('./results/blown_english/kp/source_kp.npy'),dtype='float32')  # 68--50个点, 0-1
    driving_kp_list = np.load('./results/blown_english/kp/neutral_driving.npy')
    head_pose = None
    aligner = Align()
    affline_driving, initial_frame = aligner.align_kp(source_kp,driving_kp_list,head_pose=head_pose)
    print(np.shape(affline_driving))
    




