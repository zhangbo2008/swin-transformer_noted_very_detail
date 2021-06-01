import torch
from torch import nn, einsum
import numpy as np
from einops import rearrange, repeat


class CyclicShift(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):  # https://www.jianshu.com/p/6f68ad61f39a torch.roll也是很简单的函数.
        return torch.roll(x, shifts=(self.displacement, self.displacement), dims=(1, 2))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


def create_mask(window_size, displacement, upper_lower, left_right):
    mask = torch.zeros(window_size ** 2, window_size ** 2) # 注意这里面是平方啊.

    if upper_lower: #  -21:,    :-21 , 也就是矩阵的右上角.   和左下角都干成负无穷.
        mask[-displacement * window_size:, :-displacement * window_size] = float('-inf') # 28个0, 21个-inf
        mask[:-displacement * window_size, -displacement * window_size:] = float('-inf')

    if left_right:
        mask = rearrange(mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2', h1=window_size, h2=window_size)
        mask[:, -displacement:, :, :-displacement] = float('-inf')
        mask[:, :-displacement, :, -displacement:] = float('-inf')
        mask = rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)') # 这个mask 是一块4*4的 拼起来.

    return mask


def get_relative_distances(window_size):
    indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))  # 0-6之间22配对即可.
    distances = indices[None, :, :] - indices[:, None, :]
    return distances # 表示 49,49,2    7*7个window,   我们传入的时候图片已经变成了56*56. 所以每7个我们定义成一个windows. 所以 每一个window内部有49个patch. 那么取2个patch ,然后记录他们的横坐标差和列坐标差, 第一个减第二个.我们就得到了一个49,49,2的矩阵. 就是这个distances  ===================# 下面解释, torch里面自动补充矩阵shape在这个问题上的应用.首先每一个张量都可以看做一个向量. 只不过这个向量里面的东西也是一个高维向量而已.   indices 是 00 01--------->66 这个49个元素组成的. 然后我们拓展,第一个部分是在前面补一个None 轴, 也就是现在indices 里面现在的元素都是作为一个 列来补进去的. 所以每一个00,需要 复制拓展为一个列向量.所以我们知道第一部分补完是这个样子


    """
    00     01-------------------66
    00     .                     .
    .     .                     .
    .     .                     .

    00     01-------------------66
    
    同理第二个部分补完是
    00     00-------------------00
    01    .                     .
    .     .                     .
    .     .                     .

    66     66-------------------66
    不懂的童鞋,自己debug, 就知道我这里面讲的是对的了!!!!!!!!
    
    """


class WindowAttention(nn.Module):
    def __init__(self, dim, heads, head_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        inner_dim = head_dim * heads   # 下面这些初始化读了.

        self.heads = heads
        self.scale = head_dim ** -0.5  # 跟transformer学的. 一个temperature系数
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted

        if self.shifted:
            displacement = window_size // 2  # 保证每次翻滚window的一半. 总共7*7的,
            self.cyclic_shift = CyclicShift(-displacement) # 行列像素都往左滚3个
            self.cyclic_back_shift = CyclicShift(displacement) # 都往右滚3个.  # 注意戏码2个mask 的区别一个是上下, 一个是左右的...........
            self.upper_lower_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                             upper_lower=True, left_right=False), requires_grad=False)
            self.left_right_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                            upper_lower=False, left_right=True), requires_grad=False)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if self.relative_pos_embedding:  # 这个都是启用的.
            self.relative_indices = get_relative_distances(window_size) + window_size - 1  # 虽然全部的索引是49*49 但是里面的数是从-6 到6 的一共13个数, 所以我们下一行就是13*13就足够了.
            self.pos_embedding = nn.Parameter(  torch.randn(2 * window_size - 1, 2 * window_size - 1)  ) # 因为窗口是7, 所以最大和最小差距是12, 我们每次处理的是窗口和窗口之间的交叉部分. 所以最大是-6 到6.
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size ** 2, window_size ** 2))

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):       # 下面这个函数就是整个的最核心的布冯了!!!!!!!!!!!!!!!也很短!!!!!!!!!
        if self.shifted:
            x = self.cyclic_shift(x)
        # 总感觉这种只算window内部的注意力是不是无法计算比较远的特征.s
        b, n_h, n_w, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1) # 最后一个维度直接chunk,切割成3块就可以了.
        nw_h = n_h // self.window_size       # 每一个窗口里面有patch
        nw_w = n_w // self.window_size         # h, d 表示 3个头和 dimension 32 的乘机也就是96.
# 输出的shape 是 q: batch_size,
        q, k, v = map(
            lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
                                h=h, w_h=self.window_size, w_w=self.window_size), qkv)

        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale  # 我们算的是  最后的dimension 进行乘机消去.  得到的就是 bhwij.    1,3, 64, 49 ,49   .因为一个图片最后是切成了  从 224*224--------->56*56 ------>然后再拼成window-------> 7*7----------->dots:   1,3, 64, 49 ,49          3是3个head, 64是8*8个, 每一个里面是一个7*7的window, 每一个window 里面的特征是 patch 代表的.======================================整体再分析一下. 一个图片里面有64个window. 然后一个windows 里面有 49个patch, 相对注意力,就是windows内部计算的.
        self.relative_indices=self.relative_indices.type(torch.long)
        if self.relative_pos_embedding: # 跟nlp一样加入位置编码. 位置编码的计算也是7*7 里面的       # 我们代码里面一直用的就是这个部分. 来走的.
            dots += self.pos_embedding[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
        else:
            dots += self.pos_embedding

        if self.shifted: # 加上偏移的mask
            dots[:, :, -nw_w:] += self.upper_lower_mask  # 最后8个,就是整个图片是8*8个windows的, 我们只对最后一行的window进行操作mask , 我们对这个最后一行的window进行mask, 因为我们的滚动方向是向上滚动,所以这里面的数据 7行patch里面有3行是从旧的图片的上部roll过来的.他们之间无关,不要算注意力. 所以总共就是

            # 还是画一下图. 不然太复杂了. 整个注意力图就是这个.
            #--------------------------------------
            #----------------------------------
            #.               .                .
            #.               .                .
            #.               .                .
            #.               .                .
            #.               .                .
            #49个   49个 49 49  49 49 49    49          # 最后一行有8个window, 每一个window里面有49个块.就是49个patch
            #      每一个49都长这样
            #       0        -inf
            #       -inf       0   是28  21分块.       0表示不进行mask
            #        因为做滚3,上滚3, 所以
            #       #  只看上滚, 对于7*7.他的最下面 21个是不应该算注意力的. 所以就是 49里面最后21 个不要跟其他算.但是他自身都需要算.所以就是上面-inf的由来了.




            dots[:, :, nw_w - 1::nw_w] += self.left_right_mask # 7 跳着走,步长为8.


            # 下面看left_right_mask的计算方法:
            # self.left_right_mask   是 4个0,3个inf. 的矩阵拼起来的.


            # 等式左边的也跟上面上下的道理一样.只有屏幕最右边的8个windows是我们要做mask的部分
            # 然后里面每一个东西.都是一个7*7的.因为是左边roll过来的.所以 这个7*7 最后3列是要mask的部分.
            # 对应到坐标就是.0到49里面的 每7个里面4个不管,3个mask掉. 所拼成的扣去分块对角阵.~~~~~证毕
            # 2021-06-01,20点11       ========到此我们完美的扣完了swin的全部细节~!散花.














        attn = dots.softmax(dim=-1)

        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)
        out = rearrange(out, 'b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)',
                        h=h, w_h=self.window_size, w_w=self.window_size, nw_h=nw_h, nw_w=nw_w)
        out = self.to_out(out)

        if self.shifted: # 往右滚就不用管mask了.
            out = self.cyclic_back_shift(out)
        return out


class SwinBlock(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        self.attention_block = Residual(PreNorm(dim, WindowAttention(dim=dim,
                                                                     heads=heads,
                                                                     head_dim=head_dim,
                                                                     shifted=shifted,
                                                                     window_size=window_size,
                                                                     relative_pos_embedding=relative_pos_embedding)))
        self.mlp_block = Residual(PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim)))

    def forward(self, x):
        x = self.attention_block(x)
        x = self.mlp_block(x)
        return x

#分割图片为path 然后merge
class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, downscaling_factor):
        super().__init__()
        self.downscaling_factor = downscaling_factor   # uofold 看说明即可.https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html   就是  把后2维度的矩阵 进行 kernal_size,就是 跟卷积一样抽取 kernal_size*kernal_size大小的框,然后抽取里面的数据然后拉直成一条. stride就是每次先向右平移一个stride, 到头了,再从来时向下平移一个stride. 让窗口都平移完全即可.
        self.patch_merge = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor, padding=0)
        self.linear = nn.Linear(in_channels * downscaling_factor ** 2, out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor  # 每4*4我们当做一个条来处理.
        x = self.patch_merge(x).view(b, -1, new_h, new_w).permute(0, 2, 3, 1) # 下面我们根据unfold函数来计算 shape. 输入shape 是 1,3, 224 ,224   然后我们unfold (4,4) 也就是kernal 是4, stride 也是4.  所以我们 在224*224 的里面 做 4*4 的窗口然后右, 下每次平移4个单位.所以一共有 224/4 然后平方整多个方框, 也就是3136. 第2个维度是48, 也简单就是 4*4 然后乘以之前的channel 3 即可.  这个方案就是经典的vit方案.
        x = self.linear(x) # 然后我们进入embedding一下. 跟nlp学的.
        return x


class StageModule(nn.Module):
    def __init__(self, in_channels, hidden_dimension, layers, downscaling_factor, num_heads, head_dim, window_size,
                 relative_pos_embedding):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        self.patch_partition = PatchMerging(in_channels=in_channels, out_channels=hidden_dimension,
                                            downscaling_factor=downscaling_factor)

        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):         # 注意这里面的区别一个是shifted ,一个不是shifted!!!!!!!!!!!!
            self.layers.append(nn.ModuleList([
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
            ]))

    def forward(self, x):
        x = self.patch_partition(x)  # 变到 56 * 56 了. 把图片4个像素变成一个patch
        for regular_block, shifted_block in self.layers:
            x = regular_block(x)
            x = shifted_block(x)
        return x.permute(0, 3, 1, 2)


class SwinTransformer(nn.Module):
    def __init__(self, *, hidden_dim, layers, heads, channels=3, num_classes=1000, head_dim=32, window_size=7,
                 downscaling_factors=(4, 2, 2, 2), relative_pos_embedding=True):
        super().__init__()

        self.stage1 = StageModule(in_channels=channels, hidden_dimension=hidden_dim, layers=layers[0],
                                  downscaling_factor=downscaling_factors[0], num_heads=heads[0], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage2 = StageModule(in_channels=hidden_dim, hidden_dimension=hidden_dim * 2, layers=layers[1],
                                  downscaling_factor=downscaling_factors[1], num_heads=heads[1], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage3 = StageModule(in_channels=hidden_dim * 2, hidden_dimension=hidden_dim * 4, layers=layers[2],
                                  downscaling_factor=downscaling_factors[2], num_heads=heads[2], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage4 = StageModule(in_channels=hidden_dim * 4, hidden_dimension=hidden_dim * 8, layers=layers[3],
                                  downscaling_factor=downscaling_factors[3], num_heads=heads[3], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 8),
            nn.Linear(hidden_dim * 8, num_classes)
        )

    def forward(self, img):
        x = self.stage1(img)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = x.mean(dim=[2, 3])
        return self.mlp_head(x)


def swin_t(hidden_dim=96, layers=(2, 2, 6, 2), heads=(3, 6, 12, 24), **kwargs):
    return SwinTransformer(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)


def swin_s(hidden_dim=96, layers=(2, 2, 18, 2), heads=(3, 6, 12, 24), **kwargs):
    return SwinTransformer(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)


def swin_b(hidden_dim=128, layers=(2, 2, 18, 2), heads=(4, 8, 16, 32), **kwargs):
    return SwinTransformer(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)


def swin_l(hidden_dim=192, layers=(2, 2, 18, 2), heads=(6, 12, 24, 48), **kwargs):
    return SwinTransformer(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)
