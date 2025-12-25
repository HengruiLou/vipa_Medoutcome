import cv2
import openslide
import numpy as np
import matplotlib.pyplot as plt
import glob

root = '/home/lhm/mnt/medical/'
slides = glob.glob('/home/lhm/mnt/medical/yxt/liverWSI/Hepatoma_2_5_years_800/*.svs')
import tqdm

for i in tqdm.tqdm(slides):
    name = i.split('/')[-1].split('.')[0]
    s = openslide.open_slide(i)
    samples = np.array(s.level_downsamples, dtype=np.int32)
    w, h = s.dimensions
    img = None
    if samples[1] == 4:
        img = np.array(s.read_region((w // 2, h // 2), 1, (1024, 1024)).convert('RGB'))
        img = cv2.resize(img, (img.shape[0] // 2, img.shape[1] // 2))
    elif samples[1] == 8:
        img = np.array(s.read_region((w // 2, h // 2), 1, (1024, 1024)).convert('RGB'))
    if img is not None:
        cv2.imwrite('/home/lhm/mnt/medical/lhm/liverWSI/level2/' + name + '.png', img)
        print('saved to'+'/home/lhm/mnt/medical/lhm/liverWSI/level2' + name + '.png')

# cnt1, cnt2 = 0, 0
# for s_name in slides:
#     s = openslide.open_slide(s_name)
#     size = s.level_dimensions[-1]
#     if (s.level_count < 3):
#         print(s.level_downsamples)
#         continue
#     samples = np.array(s.level_downsamples, dtype=np.int)
#     if samples[2] == 16:
#         cnt1 += 1
#     elif samples[2] == 32:
#         cnt2 += 1
#     else:
#         print(samples[2])
# print(cnt1, cnt2)

# img = s.read_region((0,0),cnt-1,size).convert('RGB')
# name = s_name.split('/')[-1].split('.')[0]
# img.save('/medical-data/lhm/CAMEL/lv0/tumor/'+name+'.png')
# gs = glob.glob('/nfs3/lhm/HCC/gnn_1/*.npy')
# for g in gs:
#     gh = np.load(g,allow_pickle=True).item()
#     print(np.unique(gh['y']))
