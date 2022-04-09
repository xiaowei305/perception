#_*_coding:utf8_*_
import numpy as np

class nn():
    def __init__(self, in_size=10, hidden_size=10, out_size=10):
        self.weight1 = np.random.randn(in_size, hidden_size) / (in_size * hidden_size) #第一层权重随机初始化
        self.bias1 = np.zeros(hidden_size)                                             #第一层偏置初始化为0
        self.weight2 = np.random.randn(hidden_size, out_size) / (hidden_size * out_size) #第二层权重随机初始化
        self.bias2 = np.zeros(out_size)                                                  #第二层偏置初始化为0
        self.lr = 0.0005  # 学习速率设置0.0005, 你也可以试试其他数值
        self.reg = 0.001  # 正则项系数

    def forward(self, x):
        self.input = x / 255 - 0.5
        self.l1_out = self.input.dot(self.weight1) + self.bias1       # 第一层输出
        self.l1_sigmoid =  1. / (1 + np.exp(-self.l1_out))            # 第一层激活
        self.l2_out = self.l1_sigmoid.dot(self.weight2) + self.bias2  # 第二层输出
        self.l2_sigmoid =  1. / (1 + np.exp(-self.l2_out))            # 第二层激活
        return np.argmax(self.l2_sigmoid, axis=1)

    def backward(self, label):
        l2_diff = self.l2_sigmoid - label           # loss 相对于第二层输出的梯度
        l2_weight_diff = self.l1_out.T.dot(l2_diff) # loss 相对于第二层权重的梯度
        l2_bias_diff = l2_diff.sum(axis=0)          # loss 相对于第二层偏置的梯度
        l1_sigmoid_diff = l2_diff.dot(self.weight2.T)    # loss 相对于第一层激活的梯度
        l1_diff = l1_sigmoid_diff * self.l1_sigmoid * (1 - self.l1_sigmoid) # loss 相对于第一层输出的梯度
        l1_weight_diff = self.input.T.dot(l1_diff)  # loss 相对于第一层权重的梯度
        l1_bias_diff = l1_diff.sum(axis=0)          # loss 相对于第一层偏置的梯度

        self.weight1 -= l1_weight_diff * self.lr + self.weight1 * self.reg # 更新第一层权重， 带L2正则
        self.bias1 -= l1_bias_diff * self.lr                               # 更新第一层偏置
        self.weight2 -= l2_weight_diff * self.lr + self.weight2 * self.reg # 更新第二层权重， 带L2正则
        self.bias2 -= l2_bias_diff * self.lr                               # 更新第一层偏置
        self.lr *= 0.99  # 学习速率减小一下
    
    def train(self, x, y):
        for i in range(100):
            self.forward(x)         # 前向传播
            self.backward(y)        # 反向传播

            # 以下都仅为了显示训练过程
            if i % 10 == 0:
                hx = self.l2_sigmoid
                loss = -((1 - y) * np.log(1 - hx) + y * np.log(hx)).mean() # 交叉熵损失函数
                pred = np.argmax(hx, axis=1)
                label = np.argmax(y, axis=1)
                accuracy = np.float32((pred == label)).sum() / label.size  #  计算在训练集上的精度
                print("loss = %f, accuracy = %f, lr = %f" % (loss, accuracy, self.lr))

    def save(self, path):
        model = {"w1":self.weight1, "b1": self.bias1, "w2": self.weight2, "b2": self.bias2}
        np.save(path, model)       # 保存权重

    def load(self, path):
        model = np.load(path, allow_pickle=True).item()  # 加载已经保存的权重
        self.weight1 = model["w1"]
        self.bias1 = model["b1"]
        self.weight2 = model["w2"]
        self.bias2 = model["b2"]

