import numpy as np




train_size = 100
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
print(batch_mask)
# x_batch = x_train[batch_mask]
# t_batch = t_train[batch_mask]