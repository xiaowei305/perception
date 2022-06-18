import numpy as np

class Kalman():
    def __init__(self, A, H, Q, R, P):
        self.A = A
        self.H = H
        self.Q = Q
        self.R = R
        self.P = P
        self.I = np.eye(P.shape[0])

    def init(self, x):
        self.x = x

    def update(self, z):
        '''
        x: (n, 1)  x = [[v], [d]]
        z: (m, 1)  z = [[d]]
        A: (n, n)  v = v, d = v + d
        H: (m, n)  v = 0 * a + v
        P: (n, n) 
        Q: (n, n) 
        R: (m, m) 
        K: (n, m) 
        '''
        A = self.A
        H = self.H

        x = A.dot(self.x)                                                 # (1)
        P = A.dot(self.P).dot(A.T) + self.Q                               # (2)
        K = P.dot(H.T).dot(np.linalg.inv(H.dot(P).dot(H.T) + self.R))     # (3)
        self.x = x + K.dot(z - H.dot(x))                                  # (4)
        self.P = (self.I - K.dot(H)).dot(P)                               # (5)
        return self.x

if __name__ == "__main__":
    from matplotlib import pyplot as plt

    x = np.arange(100)
    w = np.random.normal(loc=0.0, scale=0.2, size=x.shape)
    v = np.random.normal(loc=0.0, scale=1.2, size=x.shape)

    y_truth = 10 * np.ones(x.shape) + 0.2 * x + w    # d = d0 + 0.2 * t
    y_measure = 10 * np.ones(x.shape) + 0.2 * x + v

    A = np.array([[1, 0], [1, 1]]) #v = v, d = v + d
    H = np.array([[0, 1]])
    Q = np.array([[0.001, 0], [0, 0.001]])
    R = np.array([[1.2]])
    P = np.array([[0, 0], [0, 0.03]])
    kalman = Kalman(A, H, Q, R, P)
               
    kalman.init(np.array([[0, y_measure[0]]]).T)  # init with (v, d)

    output = np.zeros(x.shape)
    output[0] = y_measure[0]
    for i, z in enumerate(y_measure[1:]):
        v, d = kalman.update(np.array([[z]]))     # update with d
        output[i+1] = d

    plt.plot(x, y_truth, label='truth')
    plt.plot(x, y_measure, label='measure')
    plt.plot(x, output, label='kalman')
    plt.legend()
    plt.show()

