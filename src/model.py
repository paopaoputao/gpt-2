import numpy as np
import tensorflow as tf
from tensorflow.contrib.training import HParams

def default_hparams():
    return HParams(
        # 字典容量
        n_vocab=0,
        # 上下文长度
        n_ctx=1024,
        # 向量尺寸
        n_embd=768,
        # 注意力流数
        n_head=12,
        # 自注意力+前馈网络组成的模块的层数
        n_layer=12,
    )

def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def softmax(x, axis=-1):
    x = x - tf.reduce_max(x, axis=axis, keepdims=True)
    ex = tf.exp(x)
    return ex / tf.reduce_sum(ex, axis=axis, keepdims=True)

# 一种新的激活函数
def gelu(x):
    return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))

# 层归一化
def norm(x, scope, *, axis=-1, epsilon=1e-5):
    """Normalize to mean = 0, std = 1, then do a diagonal affine transform."""
    with tf.variable_scope(scope):
        n_state = x.shape[-1].value
        g = tf.get_variable('g', [n_state], initializer=tf.constant_initializer(1))
        b = tf.get_variable('b', [n_state], initializer=tf.constant_initializer(0))
        u = tf.reduce_mean(x, axis=axis, keepdims=True)
        s = tf.reduce_mean(tf.square(x-u), axis=axis, keepdims=True)
        x = (x - u) * tf.rsqrt(s + epsilon)
        x = x*g + b
        return x

# 切分最后一个维度
def split_states(x, n):
    """Reshape the last dimension of x into [n, x.shape[-1]/n]."""
    *start, m = shape_list(x)
    return tf.reshape(x, start + [n, m//n])

# 合并最后两个维度
def merge_states(x):
    """Smash the last two dimensions of x into a single dimension."""
    *start, a, b = shape_list(x)
    return tf.reshape(x, start + [a*b])

# 前馈网络 f(x) = wx + b
def conv1d(x, scope, nf, *, w_init_stdev=0.02):
    with tf.variable_scope(scope):
        *start, nx = shape_list(x)
        w = tf.get_variable('w', [1, nx, nf], initializer=tf.random_normal_initializer(stddev=w_init_stdev))
        b = tf.get_variable('b', [nf], initializer=tf.constant_initializer(0))
        c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf]))+b, start+[nf])
        return c

# 返回三角矩阵的注意力掩码矩阵
def attention_mask(nd, ns, *, dtype):
    """1's in the lower triangle, counting from the lower right corner.

    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
    """
    i = tf.range(nd)[:,None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    return tf.cast(m, dtype)


def attn(x, scope, n_state, *, past, hparams):
    # 三个维度分别是【批次batch，序列长度sequence，向量尺寸features】
    assert x.shape.ndims == 3  # Should be [batch, sequence, features]
    # n_state 会平均拆分成 n_head 个注意力流，所以要求此处可以整除
    assert n_state % hparams.n_head == 0
    # 如果past不为空，则其维度必须是【批次batch，2，注意力流数heads，序列长度sequence，向量尺寸features】
    # 第二个维度`2`分别是(`K`, `V`)
    if past is not None:
        assert past.shape.ndims == 5  # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]

    # 输入向量【批次batch，序列长度sequence，向量尺寸features】
    # 最后一个维度进行切分为【批次batch，序列长度sequence，注意力流数heads，向量分量尺寸features_per_head】
    # 然后再转置为【批次batch，注意力流数heads，序列长度sequence，向量分量尺寸features_per_head】
    def split_heads(x):
        # From [batch, sequence, features] to [batch, heads, sequence, features]
        return tf.transpose(split_states(x, hparams.n_head), [0, 2, 1, 3])

    # split_heads的逆操作，先转置回去再合并
    def merge_heads(x):
        # Reverse of split_heads
        return merge_states(tf.transpose(x, [0, 2, 1, 3]))

    def mask_attn_weights(w):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        _, _, nd, ns = shape_list(w)
        # 掩码矩阵是个三角矩阵，维度为【源序列长度，目标序列长度】
        b = attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w*b - tf.cast(1e10, w.dtype)*(1-b)
        return w

    # `q`和`k`的维度都是【批次，注意力流数，序列长度，向量份量尺寸】
    def multihead_attn(q, k, v):
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)
        # 除以`v`的尺寸的平方根
        w = w * tf.rsqrt(tf.cast(v.shape[-1].value, w.dtype))

        w = mask_attn_weights(w)
        w = softmax(w)
        a = tf.matmul(w, v)
        return a

    with tf.variable_scope(scope):
        c = conv1d(x, 'c_attn', n_state*3)
        q, k, v = map(split_heads, tf.split(c, 3, axis=2))
        present = tf.stack([k, v], axis=1)
        if past is not None:
            pk, pv = tf.unstack(past, axis=1)
            k = tf.concat([pk, k], axis=-2)
            v = tf.concat([pv, v], axis=-2)
        a = multihead_attn(q, k, v)
        a = merge_heads(a)
        a = conv1d(a, 'c_proj', n_state)
        return a, present

# MLP(x) = F(gelu(F(x))) FFN网络
def mlp(x, scope, n_state, *, hparams):
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        h = gelu(conv1d(x, 'c_fc', n_state))
        h2 = conv1d(h, 'c_proj', nx)
        return h2

# 注意力模块+前馈网络组成的模块block
def block(x, scope, *, past, hparams):
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        a, present = attn(norm(x, 'ln_1'), 'attn', nx, past=past, hparams=hparams)
        x = x + a
        m = mlp(norm(x, 'ln_2'), 'mlp', nx*4, hparams=hparams)
        x = x + m
        return x, present

# 历史序列的形状
def past_shape(*, hparams, batch_size=None, sequence=None):
    return [batch_size, hparams.n_layer, 2, hparams.n_head, sequence, hparams.n_embd // hparams.n_head]

# 扩展最后一个维度
def expand_tile(value, size):
    """Add a new axis of given size."""
    value = tf.convert_to_tensor(value, name='value')
    ndims = value.shape.ndims
    return tf.tile(tf.expand_dims(value, axis=0), [size] + [1]*ndims)

def positions_for(tokens, past_length):
    batch_size = tf.shape(tokens)[0]
    nsteps = tf.shape(tokens)[1]
    return expand_tile(past_length + tf.range(nsteps), batch_size)


def model(hparams, X, past=None, scope='model', reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        results = {}
        batch, sequence = shape_list(X)

        # 位置编码 + 符号编码 = 得到输入向量
        wpe = tf.get_variable('wpe', [hparams.n_ctx, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.01))
        wte = tf.get_variable('wte', [hparams.n_vocab, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.02))
        past_length = 0 if past is None else tf.shape(past)[-2]
        h = tf.gather(wte, X) + tf.gather(wpe, positions_for(X, past_length))

        # 这里是一个阉割版的Transformer，仅剩带自掩码的自注意力模块和前馈网络
        # Transformer
        presents = []
        pasts = tf.unstack(past, axis=1) if past is not None else [None] * hparams.n_layer
        assert len(pasts) == hparams.n_layer
        for layer, past in enumerate(pasts):
            h, present = block(h, 'h%d' % layer, past=past, hparams=hparams)
            presents.append(present)
        # [batch_size, n_layer, 2[out of attn + mlp, K & V in attn], n_head, seq_len, n_embd_per_head]
        # 返回的present是【批次，层数，2，注意力流数，序列长度，向量分量尺寸】
        # 其中的`2`是由K和V组成
        results['present'] = tf.stack(presents, axis=1)
        # 最后再做一次层归一化
        h = norm(h, 'ln_f')

        # 用输出向量和符号编码的内积/点乘来生成logits
        # Language model loss.  Do tokens <n predict token n?
        h_flat = tf.reshape(h, [batch*sequence, hparams.n_embd])
        logits = tf.matmul(h_flat, wte, transpose_b=True)
        logits = tf.reshape(logits, [batch, sequence, hparams.n_vocab])
        results['logits'] = logits
        return results


if __name__ == '__main__':

    hparams = HParams(
        n_vocab=4,
        n_ctx=5,
        n_embd=4,
        n_head=2,
        n_layer=3,
    )

    # 创建常量变量，构建GPT-2模型
    input_ids_constant = tf.constant([[1, 2, 3, ]], dtype=tf.int32)

    # 占位符方式
    input_value = [[1, 2, 3, ]]
    input_ids_placeholder = tf.placeholder(shape=[len(input_value), None, ], dtype=tf.int32)

    gpt2_constant = model(hparams=hparams, X=input_ids_constant, reuse=tf.AUTO_REUSE)
    gpt2_placeholder = model(hparams=hparams, X=input_ids_placeholder, reuse=tf.AUTO_REUSE)

    with tf.Session() as sess:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        input_ids_constant_, gpt2_constant_ = sess.run([input_ids_constant, gpt2_constant])

        input_ids_placeholder, gpt2_placeholder_ = sess.run([input_ids_placeholder, gpt2_placeholder], feed_dict={input_ids_placeholder: input_value})

        gpt2_constant_present, gpt2_constant_logits = gpt2_constant_['present'], gpt2_constant_['logits']
        gpt2_placeholder_present, gpt2_placeholder_logits = gpt2_placeholder_['present'], gpt2_placeholder_['logits']

        print('input_ids_constant_:', input_ids_constant_.shape, input_ids_constant_.tolist())
        print('present[batch_size, n_layer, 2[out of attn + mlp, K & V in attn], n_head, seq_len, n_embd_per_head]:', gpt2_constant_present.shape)
        print('logits[batch_size, seq_len, vocab_size]:', gpt2_constant_logits, softmax(gpt2_constant_logits).eval())

        # print('input_ids_placeholder:', input_ids_placeholder.shape, input_ids_placeholder.tolist())
        # print('present[batch_size, n_layer, 2[out of attn + mlp, K & V in attn], n_head, seq_len, n_embd_per_head]:', gpt2_placeholder_present.shape)
        # print('logits[batch_size, seq_len, vocab_size]:', gpt2_placeholder_logits.shape)







