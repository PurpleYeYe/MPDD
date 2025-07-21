import numpy as np

stationary = {8: 0, 12: 1, 16: 0, 17: 1, 22: 0, 29: 0, 39: 0}
if __name__ == "__main__":
    a = np.load('./MPDD-Young/Training/1s/Audio/mfccs/001_001.npy')
    v = np.load('./MPDD-Young/Training/1s/Visual/resnet/001_001.npy')
    print(a.shape)
    print(v.shape)
