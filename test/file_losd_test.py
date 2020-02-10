from __future__ import absolute_import
from six.moves import range

import os

import nnabla as nn
# ①　NNabla関連モジュールのインポート
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
from nnabla.utils.data_iterator import data_iterator_simple
import nnabla.initializer as I
import nnabla.utils.save as save

# ②　NNabla関連以外のモジュールのインポート
import numpy as np
from PIL import Image
from _checkpoint_nnp_util import save_nnp
from args import get_args
from base import load_digits

args = get_args()
path_w = "./log.txt"


# ③　tiny_digits.pyから転載したデータ整形function


def data_iterator_tiny_digits(digits, batch_size=16, shuffle=False, rng=None):
    def load_func(index):
        data = digits.images[index]
        label = digits.target[index]
        return data[None],np.array([label]).astype(np.int32)
    return data_iterator_simple(load_func, digits.target.shape[0], batch_size, shuffle, rng, with_file_cache=False)

# ④　損失グラフを構築する関数を定義する


def logreg_loss(y, t):
    loss_f = F.mean(F.softmax_cross_entropy(y, t))
    return loss_f

# ⑤ トレーニング関数を定義する


def training(xt, tt, data_t, loss_t, steps, learning_rate):
    solver = S.Sgd(learning_rate)
    # Set parameter variables to be updatd.
    solver.set_parameters(nn.get_parameters())
    for i in range(steps):
        xt.d, tt.d = data_t.next()
        loss_t.forward()
        solver.zero_grad()  # Initialize gradients of all parameters to zero.
        loss_t.backward()
        # Applying weight decay as an    regularization
        solver.weight_decay(1e-5)
        solver.update()
        if i % 100 == 0:  # Print for each 10 iterations
            print(str(i) + ":" + str(loss.d))

# ⑥ ニューラルネットを構築する関数を定義する


def network(x):
    with nn.parameter_scope("cnn"):
        with nn.parameter_scope("conv1"):
            h = F.tanh(PF.batch_normalization(
            PF.convolution(x, 4, (3,3), pad=(1, 1), stride=(2, 2))))
        with nn.parameter_scope("conv2"):
            h = F.tanh(PF.batch_normalization(
            PF.convolution(h, 8, (3, 3), pad=(1, 1))))
            h = F.average_pooling(h, (2, 2))
        with nn.parameter_scope("fc3"):
            h = F.tanh(PF.affine(h, 6))
        with nn.parameter_scope("classifier"):
            h = PF.affine(h,2)
    return h


# ⑦　実行開始：scikit_learnでdigits（8✕8サイズ）データを取得し、NNablaで処理可能に整形する
np.random.seed(0)
digits = load_digits(n_class=2)
print(digits)
data = data_iterator_tiny_digits(digits,batch_size=32, shuffle=True)

# ⑧　ニューラルネットワークを構築する
nn.clear_parameters()
img, label = data.next()
x = nn.Variable(img.shape)
y = network(x)
t = nn.Variable(label.shape)
loss = logreg_loss(y, t)

# ⑨　学習する
learning_rate = 1e-1
training(x, t, data, loss, 1000, learning_rate)

contents = save_nnp({'x': x}, {'y': y}, 26)
save.save(os.path.join(args.model_save_path,'{}_result.nnp'.format(args.net)), contents)

# ⑩　推論し、最後に正確さを求めて表示する
x.d, t.d = data.next()
y.forward()
mch = 0
for p in range(len(t.d)):
    with open(path_w, mode='a') as f:
        f.write(str(y.d.argmax(axis=1))+"\n")
    print(y.d)
    print(t.d)
    if t.d[p] == y.d.argmax(axis=1)[p]:
        mch = mch + 1
        print("hit")

print("Accuracy:{}".format(mch / len(t.d)))

