import mxnet as mx
from time import time
import numpy as np

n = 10000
batch_size = 1
beam_size = 5
max_output_length = 50
encoded_source_length = 40
context = mx.cpu()


def init_data():
    sequences = mx.nd.full((batch_size * beam_size, max_output_length), val=0, ctx=context, dtype='int32')
    finished = mx.nd.zeros((batch_size * beam_size,), ctx=context, dtype='int32')
    lengths = mx.nd.ones((batch_size * beam_size, 1), ctx=context, dtype='int32')
    attentions = mx.nd.zeros((batch_size * beam_size, max_output_length, encoded_source_length), ctx=context)
    return sequences, finished, lengths, attentions


#symbolic
def take_module(batch_size, beam_size, max_output_length, encoded_source_length, context):
    indices = mx.sym.Variable('best_hyp_indices', dtype='int32')
    inputs = [mx.sym.Variable('sequences', dtype='int32'),
              mx.sym.Variable('finished', dtype='int32'),
              mx.sym.Variable('lengths', dtype='int32'),
              mx.sym.Variable('attentions')]
    sym = mx.sym.Group([mx.sym.take(inp, indices, name='%s_taken' % inp.name) for inp in inputs])
    shapes = [mx.io.DataDesc(name='best_hyp_indices', shape=(batch_size*beam_size,), dtype='int32'),
              mx.io.DataDesc(name='sequences', shape=(batch_size * beam_size, max_output_length), dtype='int32'),
              mx.io.DataDesc(name='finished', shape=(batch_size * beam_size,), dtype='int32'),
              mx.io.DataDesc(name='lengths', shape=(batch_size * beam_size, 1), dtype='int32'),
              mx.io.DataDesc(name='attentions', shape=(batch_size * beam_size, max_output_length, encoded_source_length))]
    mod = mx.mod.Module(symbol=sym, data_names=sym.list_arguments(), label_names=[], context=context)

    mod.bind(data_shapes=shapes, for_training=False, grad_req="null")
    mod.init_params()
    return mod


#Gluon
class Taker(mx.gluon.HybridBlock):
    def __init__(self):
        super().__init__()

    def hybrid_forward(self, F, indices, sequences, finished, lengths, attentions):
        return F.take(sequences, indices), F.take(finished, indices), F.take(lengths, indices), F.take(attentions, indices)


#imperative
def take(indices, sequences, finished, lengths, attentions):
    return sequences.take(indices), finished.take(indices), lengths.take(indices), attentions.take(indices)


def get_indices(n, batch_size, beam_size, context):
    return [mx.nd.array(np.random.randint(0, batch_size * beam_size, size=(batch_size * beam_size,)), ctx=context) for _ in range(n)]


def benchmark_imperative():
    sequences, finished, lengths, attentions = init_data()
    indeces = get_indices(n, batch_size, beam_size, context)
    start = time()
    for idx in indeces:
        sequences, finished, lengths, attentions = take(idx, sequences, finished, lengths, attentions)
        mx.nd.waitall()
    end = time()
    runtime = end - start
    print("Imperative: init=0.0 runtime=%.4f" % runtime)


def benchmark_symbolic():
    sequences, finished, lengths, attentions = init_data()
    indeces = get_indices(n, batch_size, beam_size, context)
    start = time()
    mod = take_module(batch_size, beam_size, max_output_length, encoded_source_length, context)
    end = time()
    init_time = end - start
    start = time()
    for idx in indeces:
        mod.forward(mx.io.DataBatch([idx, sequences, finished, lengths, attentions]), is_train=False)
        sequences, finished, lengths, attentions = mod.get_outputs()
        mx.nd.waitall()
    end = time()
    runtime = end - start
    print("Symbolic: init=%.4f %.4f" % (init_time, runtime))


def benchmark_hybridblock(hybridize):
    sequences, finished, lengths, attentions = init_data()
    indeces = get_indices(n, batch_size, beam_size, context)
    start = time()
    take = Taker()
    if hybridize:
        take.hybridize()
    end = time()
    init_time = end - start
    start = time()
    for idx in indeces:
        sequences, finished, lengths, attentions = take(idx, sequences, finished, lengths, attentions)
        mx.nd.waitall()
    end = time()
    runtime = end - start
    if hybridize:
        print("HybridBlock: init=%.4f %.4f" % (init_time, runtime))
    else:
        print("Block: init=%.4f %.4f" % (init_time, runtime))


benchmark_imperative()
benchmark_symbolic()
benchmark_hybridblock(hybridize=True)
benchmark_hybridblock(hybridize=False)