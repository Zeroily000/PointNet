import numpy as np
import matplotlib.pyplot as plt

acc = '93.40/'
dir = '../results/segmentation/'
loss = np.load(dir + acc + 'loss.npz')
loss_train = loss['train']
loss_valid = loss['valid']

acc = np.load(dir + acc + 'accuracy.npz')
acc_train = acc['train']
acc_valid = acc['valid']
print(acc_valid[-1])
plt.figure()
plt.plot(range(len(loss_train)), loss_train, label='train')
plt.plot(range(len(loss_valid)), loss_valid, label='validation')
plt.legend()
plt.title('Loss')

plt.figure()
plt.plot(range(len(acc_train)), acc_train, label='train')
plt.plot(range(len(acc_valid)), acc_valid, label='validation')
plt.legend()
plt.title('Accuracy')

plt.show()
