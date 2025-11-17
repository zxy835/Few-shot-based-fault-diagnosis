from DTW import find_best_match
import numpy as np

def sample_every_other_block(data, block_size):
    segments = []
    total_length = data.shape[0]
    for start in range(0, total_length, block_size * 2):
        end = start + block_size
        segment = data[start:end]
        segments.append(segment)
    return np.vstack(segments) if segments else np.empty((0, data.shape[1]))
jiankang_path = r"Health.npy"
baochijia_path = r"Cage.npy"
neiquan_path = r"Inner.npy"
waiquan_path = r"Outer.npy"
path = r"train_informer.npy"
train_data = np.load(path)
num = len(train_data) // 4096
jiankang = np.load(jiankang_path)
baochijia = np.load(baochijia_path)
neiquan = np.load(neiquan_path)
waiquan = np.load(waiquan_path)

jiankang1 = jiankang[0:65536,:-1]
baochijia1 = baochijia[0:65536,:-1]
neiquan1 = neiquan[0:65536,:-1]
waiquan1 = waiquan[0:65536,:-1]

jiankang = sample_every_other_block(jiankang1, 2048)
baochijia = sample_every_other_block(baochijia1, 2048)
neiquan = sample_every_other_block(neiquan1, 2048)
waiquan = sample_every_other_block(waiquan1, 2048)

traindata = []
severedata = []
for i in range (num):
    index1 = i*4096
    index2 = index1+2048
    index3 = (i+1)*4096
    train = train_data[index1:index2,:-1]
    severe = train_data[index2:index3,:-1]
    label = train_data[index1:index2, -1:]
    df = train
    zuiyou_jiankang, start1,c1 = find_best_match(jiankang, df)
    zuiyou_baochijia, start2,c2 = find_best_match(baochijia, df)
    zuiyou_neiquan, start3,c3 = find_best_match(neiquan, df)
    zuiyou_waiquan, start4,c4 = find_best_match(waiquan, df)
    start1 = start1 // 2048
    start2 = start2 // 2048
    start3 = start3 // 2048
    start4 = start4 // 2048
    combined = np.hstack((df, zuiyou_jiankang, zuiyou_baochijia, zuiyou_neiquan, zuiyou_waiquan, label))
    traindata.append(combined)
    combined1 = np.hstack((severe, jiankang1[start1*4096+2048:(start1+1)*4096], baochijia1[start1*4096+2048:(start1+1)*4096], neiquan1[start1*4096+2048:(start1+1)*4096], waiquan1[start1*4096+2048:(start1+1)*4096]))
    severedata.append(combined1)
    print(i+1, '/', num)
traindata = np.array(traindata)
severedata = np.array(severedata)
np.save(r'sim_traindata.npy', traindata)
np.save(r'severe_traindata.npy', severedata)


