import numpy as np
import matplotlib.pyplot as plt


dir = '../results/86.73/'
loss = np.load(dir + 'loss.npz')
loss_train = loss['train']
loss_valid = loss['valid']

acc = np.load(dir + 'accuracy.npz')
acc_train = acc['train']
acc_valid = acc['valid']
print acc_valid[-1]
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
