#通过感知机分类半月数据
import numpy as np
import matplotlib.pyplot as plt

def sgn(y):
    y[y > 0] = 1
    y[y < 0] = -1
    return y

class Perceptron(object):
    '''单层感知机
    '''
    def __init__(self, shape):
        super(Perceptron, self).__init__()

        self.w = np.ones(shape)      #weigth
        self.b = 1.5                                 #the bias
        self.activate_func = sgn

    def update(self,x,y,out,learning_rate):
        self.w += learning_rate * x.T * (y - out)

    def calclate(self, x):
        return self.activate_func(np.dot(self.w, x.T) + self.b)

    def loss_func(self, pre_y, gt_y):
        return (pre_y - gt_y) ** 2

    def train(self, x, y, epochs, learning_rate):
        losses = []
        for epoch in range(epochs):
            loss_tmp = []
            for i in range(x.shape[0]):
                out = self.calclate(x[i])
                loss_tmp.append(self.loss_func(out, y[i]))
                self.update(x[i], y[i], out, learning_rate)

            losses.append(sum(loss_tmp)/len(loss_tmp))
        return losses

    def predict(self, x):
        out = self.calclate(x)
        return out
    
    def test(self, x,y):
        label = self.predict(x)
        gt_count = np.sum(label==y)
        wrong_count = np.sum(label!=y)
        return wrong_count/(wrong_count+gt_count),gt_count/(wrong_count+gt_count)


    def get_params(self):
        return {'weight':self.w, 'bias':self.b}

    def draw(self):
        axis = [i for i in range(1000)]
        out = [self.w * i + self.b for i in axis]
        
        plt.plot(axis, out)
        plt.show()

def load_data(file):
    x = []
    y = []
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(',')
            
            x_item = [float(line[0]), float(line[1])]
            y_item = float(line[2])
            
            x.append(x_item)
            y.append(y_item)
    
    return np.array(x), np.array(y)


def split_data(x, y):
    train_x, test_x = x[:int(x.shape[0]*0.7)], x[int(x.shape[0]*0.7):]
    train_y, test_y = y[:int(y.shape[0]*0.7)], y[int(y.shape[0]*0.7):]
    
    return train_x, train_y, test_x, test_y

if __name__ == '__main__':
    #进行非线性数据的分类实验时，只需要将数据的间隔缩小保证二者重合即可
    desc = 'nonlinear'
    file = './halfmoon.txt'
    x,y = load_data(file)

    train_x, train_y, test_x, test_y = split_data(x, y)

    neur = Perceptron((1,2))
    losses = neur.train(train_x,train_y,100, 0.0001)
    
    err,acc = neur.test(test_x, test_y)
    print('rate of error:', err)
    print('rate of accuracy:', acc)


    #画损失曲线
    axis = [i for i in range(len(losses))]
    plt.figure()
    plt.plot(axis, losses)
    plt.savefig('../imgs/%s_mse_loss.png' % desc)
    #plt.show()

    #画决策面   
    x_aixs = x[:,0]
    y_aixs = x[:,1]

    neg_x_axis = x_aixs[y==-1]
    neg_y_axis = y_aixs[y==-1]
    
    pos_x_axis = x_aixs[y==1]
    pos_y_axis = y_aixs[y==1]

    #感知机的参数
    params = neur.get_params()
    w = params['weight']
    b = params['bias']

    k = -1 * w[0][0] / w[0][1]
    b = -1 * b / w[0][1]

    divid_x = [i for i in range(-15,25)]
    divid_y = [k * i + b for i in divid_x]

    plt.figure()
    plt.plot(divid_x, divid_y, c='r')
    plt.scatter(neg_x_axis,neg_y_axis,c="b",s=10)
    plt.scatter(pos_x_axis,pos_y_axis,c="g",s=10)
    plt.savefig('../imgs/%s_divide.png' % desc)   #保存决策面