import numpy as np

def distance_rectangle(r1,r2):
    d1 = np.sqrt((r1[0]-r2[0])**2+(r1[1]-r2[1])**2)
    d2 = np.sqrt((r1[2]-r2[2])**2+(r1[3]-r2[3])**2)        
    return d1+d2
                
class BlobTracker(object):
    def __init__(self):
        self.blobs = []
        self.uid = 0
        
    def match(self, rects):
        min_threshold = 500.0
        max_life = 60
        uids = [0 for r in rects]
        kills = []
        for blob_a in self.blobs:            
            ds = sorted([(distance_rectangle(blob_b, blob_a['rect']),blob_b) for blob_b in rects])                        
            if len(ds)>0:
                if ds[0][0]<min_threshold:
                    blob_a['rect'] = list(ds[0][1])
                    blob_a['life'] = max_life
                    ds[0][1][0] = 1e10 # mark this as unusable for any other assignment
                    ds[0][1][1] = blob_a['uid']
                blob_a['life'] -= 1
                if blob_a['life']<0:
                    kills.append(blob_a)
                    
        # kill old blobs
        for blob in kills:
            self.blobs.remove(blob)
        # birth new blobs
        for j,blob_b in enumerate(rects):
            if blob_b[0]<1e8:                
                self.blobs.append({'rect':blob_b, 'life':max_life, 'uid':self.uid})
                uids[j] = self.uid
                self.uid = self.uid+1
            else:
                uids[j] = blob_b[1]
        return uids