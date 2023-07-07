# 用pytorch写一个三层的全连接神经网络，用来测试copilot的性能
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time

# 超参数
input_size = 10
hidden_size = 10
output_size = 10
learning_rate = 1e-4
num_epoches = 1000

# 构建数据集
x_train = np.random.randn(100, input_size)
y_train = np.random.randn(100, output_size)

# 构建模型
model = nn.Sequential(
    nn.Linear(input_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, output_size)
)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 开始训练
start = time.time()
for epoch in range(num_epoches):
    # 转换为torch的tensor
    inputs = torch.from_numpy(x_train).float()
    targets = torch.from_numpy(y_train).float()

    # 前向传播
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print('Epoch[{}/{}], loss: {:.6f}'.format(epoch + 1, num_epoches, loss.item()))

end = time.time()
print('Total time: {:.6f}'.format(end - start))

# 保存模型
torch.save(model.state_dict(), './model.ckpt')

# 测试模型
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for data in x_train:
        inputs = torch.from_numpy(data).float()
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 0)
        total += 1
        correct += (predicted == targets).sum()
    print('Accuracy of the network on the test images: {} %'.format(100 * correct / total))
