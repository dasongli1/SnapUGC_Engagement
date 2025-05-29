import os
import json
video_path = "/home/jupyter/snapdataset/0928/videos/videos_7/"
wav_path = "/home/jupyter/snapdataset/0928/wavs/wav_7/"
videos = []
for ele in os.listdir(video_path):
    if ele[-4:] == ".mp4" and os.path.exists(os.path.join(wav_path, ele[0:-4]+".wav")):
        # print(ele)
        videos.append(ele[0:-4])
        # exit(0)
        # if len(videos) > 50:
        #     break
print(len(videos))
with open('test_out2_ours_7.json', 'a') as fp:
    for kk in range(len(videos)):
        temp = {}
        temp['video_id'] = videos[kk]
        json.dump(temp, fp)
        fp.write("\n")
        
        
