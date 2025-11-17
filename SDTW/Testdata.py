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

path = r"test_informer.npy"

train_data = np.load(path)
train_data = train_data.reshape(-1,7)
num = len(train_data) // 2048

jiankang = np.load(jiankang_path)
baochijia = np.load(baochijia_path)
neiquan = np.load(neiquan_path)
waiquan = np.load(waiquan_path)

jiankang = jiankang [0:65536,:-1]
baochijia = baochijia [0:65536,:-1]
neiquan = neiquan [0:65536,:-1]
waiquan = waiquan [0:65536,:-1]

jiankang = sample_every_other_block(jiankang, 2048)
baochijia = sample_every_other_block(baochijia, 2048)
neiquan = sample_every_other_block(neiquan, 2048)
waiquan = sample_every_other_block(waiquan, 2048)

traindata = []
for i in range (num):
    index1 = i*2048
    index2 = index1+2048
    train = train_data[index1:index2,:-1]
    label = train_data[index1:index2, -1:]
    df = train
    zuiyou_jiankang, _ = find_best_match(jiankang, df)
    zuiyou_baochijia, _ = find_best_match(baochijia, df)
    zuiyou_neiquan, _ = find_best_match(neiquan, df)
    zuiyou_waiquan, _ = find_best_match(waiquan, df)
    combined = np.hstack((df, zuiyou_jiankang, zuiyou_baochijia, zuiyou_neiquan, zuiyou_waiquan, label))
    traindata.append(combined)
    print(i+1, '/', num)
traindata = np.array(traindata)
np.save(r'sim_testdata.npy', traindata)


