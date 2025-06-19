import os, glob, pickle, cv2, numpy as np, torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import mediapipe as mp
from ..constants import NUM_FRAMES, SEG_LEN, STRIDE, IMG_SIZE


class FIV2HemifaceDS(Dataset):
    def __init__(self, root_videos, anno_pkl,
                 n_frames=NUM_FRAMES, seg_len=SEG_LEN, stride=STRIDE):
        super().__init__()
        self.root, self.T, self.L, self.S = root_videos, n_frames, seg_len, stride
        with open(anno_pkl, "rb") as f:
            ann = pickle.load(f, encoding="latin1")
        self.ids   = list(ann["extraversion"].keys())
        self.label = lambda vid: torch.tensor(
            [ann[t][vid] for t in ("extraversion","neuroticism",
                                   "agreeableness","conscientiousness","openness")],
            dtype=torch.float32)

        self.tr  = T.Compose([T.ToTensor(),
                              T.Resize((IMG_SIZE, IMG_SIZE)),
                              T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
        self.det = mp.solutions.face_detection.FaceDetection(0, 0.5)

    def _face_crop(self, rgb, det):
        h,w,_ = rgb.shape
        bb = det.location_data.relative_bounding_box
        x1,y1 = int(bb.xmin*w), int(bb.ymin*h)
        x2,y2 = int((bb.xmin+bb.width)*w), int((bb.ymin+bb.height)*h)
        return rgb[max(0,y1):y2, max(0,x1):x2]

    def __getitem__(self, idx):
        vid = self.ids[idx];   y = self.label(vid)
        path = glob.glob(os.path.join(self.root,"**",vid), recursive=True)[0]
        cap  = cv2.VideoCapture(path)
        tot  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = np.linspace(0, tot-1, self.T).astype(int)
        L, R = [], []
        for i in range(tot):
            ret, frm = cap.read()
            if not ret or i not in idxs:  continue
            rgb  = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
            dets = self.det.process(rgb).detections
            if not dets:  continue
            face = self._face_crop(rgb, dets[0])
            if face.size==0: continue
            h,w,_ = face.shape
            L.append(self.tr(face[:, :w//2]))
            R.append(self.tr(face[:,  w//2:]))
        cap.release()
        while len(L) < self.T:
            L.append(L[-1]); R.append(R[-1])
        vidL, vidR = torch.stack(L,0), torch.stack(R,0)   
        segL, segR = [], []
        for s in range(0, self.T-self.L+1, self.S):
            segL.append(vidL[s:s+self.L]); segR.append(vidR[s:s+self.L])
        return torch.stack(segL), torch.stack(segR), y   

    def __len__(self): return len(self.ids)