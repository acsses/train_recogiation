#①　NNabla関連モジュールのインポート
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
from nnabla.utils.data_iterator import data_iterator_simple

#②　NNabla関連以外のモジュールのインポート
import numpy as np
from sklearn.datasets import load_digits

#③　tiny_digits.pyから転載したデータ整形function
def data_iterator_tiny_digits(digits, batch_size=64, shuffle=False, rng=None):
    def load_func(index):
         """Loading an image and its label"""
         img = digits.images[index]
         label = digits.target[index]
         return img[None], np.array([label]).astype(np.int32)
    return data_iterator_simple(load_func, digits.target.shape[0], batch_size, shuffle, rng, with_file_cache=False)

#④　損失グラフを構築する関数を定義する
def logreg_loss(y, t):
     loss_f = F.mean(F.softmax_cross_entropy(y, t))
     return loss_f

#⑤ トレーニング関数を定義する
def training(xt,tt,data_t,loss_t,steps,learning_rate):
     solver = S.Sgd(learning_rate)
     solver.set_parameters(nn.get_parameters()) # Set parameter variables to be updatd.
     for i in range(steps):
         xt.d, tt.d = data_t.next()
         loss_t.forward()
         solver.zero_grad() # Initialize gradients of all parameters to zero.
         loss_t.backward()
         solver.weight_decay(1e-5) # Applying weight decay as an    regularization
         solver.update()
         if i % 100 == 0: # Print for each 10 iterations
              print(str(i) +":" + str(loss.d))

#⑥ ニューラルネットを構築する関数を定義する
def network(x):
    with nn.parameter_scope("cnn"): 
         with nn.parameter_scope("conv1"):
             h = F.tanh(PF.batch_normalization(
             PF.convolution(x, 4, (3, 3), pad=(1, 1), stride=(2, 2))))
         with nn.parameter_scope("conv2"):
             h = F.tanh(PF.batch_normalization(
             PF.convolution(h, 8, (3, 3), pad=(1, 1))))
             h = F.average_pooling(h, (2, 2))
         with nn.parameter_scope("fc3"):
             h = F.tanh(PF.affine(h, 32))
         with nn.parameter_scope("classifier"):
             h = PF.affine(h,10)
    return h

#⑦　実行開始：scikit_learnでdigits（8✕8サイズ）データを取得し、NNablaで処理可能に整形する
np.random.seed(0)
digits = load_digits(n_class=10)
data = data_iterator_tiny_digits(digits, batch_size=64, shuffle=True)

#⑧　ニューラルネットワークを構築する
nn.clear_parameters()
img, label = data.next()
x = nn.Variable(img.shape)
y = network(x)
t = nn.Variable(label.shape)
loss = logreg_loss(y, t)

#⑨　学習する
learning_rate = 1e-1
training(x,t,data,loss,1000,learning_rate)

#⑩　推論し、最後に正確さを求めて表示する
x.d, t.d = data.next()
y.forward()
mch = 0
for p in range(len(t.d)):
    if t.d[p] == y.d.argmax(axis=1)[p] :
         mch += 1

print("Accuracy:{}".format(mch / len(t.d)))

 