import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
from nnabla.utils.nnp_graph import NnpLoader
from nnabla.utils.data_iterator import data_iterator_simple


import numpy as np

def data_iterator_tiny_digits(digits, batch_size=1, shuffle=False, rng=None):
    def load_func(index):
        data_all = np.load("./test_data.npy")
        data_all = data_all[None]
        target_all = np.load("./test_target.npy")
        data = data_all[index]
        print(data)
        target = target_all[index]
        return data[None],np.array([target]).astype(np.int32)
    return data_iterator_simple(load_func, 1, batch_size, shuffle, rng, with_file_cache=False)

# Read a .nnp file.
nnp = NnpLoader("./test/tmp.monitor/lenet_result.nnp")
# Assume a graph `graph_a` is in the nnp file.
net = nnp.get_network("Validation", 1)
# import nnabla.solvers as S
print(net)
digits = 0
data = data_iterator_tiny_digits(digits,batch_size=1, shuffle=True)

t = nn.Variable((1,1))
x = nn.Variable((64,64))
y = net.outputs['y']
x.d, t.d = data.next()
y.forward()
mch = 0
for p in range(len(t.d)):
    if t.d[p] == y.d.argmax(axis=0)[p]:
        acc = y.d.argmax
        mch += 1
        print(y.d.argmax)


