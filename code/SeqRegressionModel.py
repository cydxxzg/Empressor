import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

def Seq2Scalar(mode='denselstm',seqL=100):
    """
    Parameters:
        mode (str)          -- the model you wanna use: denselstm | cnn_k15 | DenseLSTM2
        seqL (int)          -- the length of the input sequence
    """
    print('The model type is : ')
    print(mode)
    if mode == 'denselstm':
        return DenseLSTM(input_nc=4, growth_rate=32, block_config=(2, 2, 4, 2),
                 num_init_features=64, bn_size=4, drop_rate=0.2, input_length=seqL)
    elif mode == 'cnn_k15':
        return CNN_K15(input_nc = 4, channel_num = 200, drop_rate = 0.2, kernel_size = 5,
                      use_dropout = False, input_length = seqL)
    elif mode == 'DenseLSTM2':
        return DenseLSTM2(input_nc=4, growth_rate=32, block_config=(2, 2, 4, 2),
                 num_init_features=128, bn_size=4, drop_rate=0.2,input_length=seqL)

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm1d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv1d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm1d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv1d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)

class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm1d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv1d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool1d(kernel_size=2, stride=2))

class DenseLSTM(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """
    def __init__(self, input_nc=4, growth_rate=32, block_config=(2, 2, 4, 2),
                 num_init_features=64, bn_size=4, drop_rate=0, input_length=100):

        super(DenseLSTM, self).__init__()

        # First convolution
        self.features0 = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv1d(input_nc, num_init_features, kernel_size=7, stride=1, padding=3, bias=False)),
            ('norm0', nn.BatchNorm1d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool1d(kernel_size=3, stride=2, padding=1)),
        ]))
        self.features = nn.Sequential(OrderedDict([]))
        length = np.floor((input_length + 2 * 1 - 1 - 2)/2 + 1)
        # Each denseblock
        self.lstm = nn.Sequential(OrderedDict([]))
        self.lstm.add_module('lstm_layer', torch.nn.LSTM(input_size=num_init_features, hidden_size=num_init_features,
                                                       num_layers=3, bias=True, batch_first=True, bidirectional=True))
        num_features = 2*num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
                length = np.floor((length - 1 - 1) / 2 + 1)

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm1d(num_features))

        # Linear layer
        self.ratio = nn.Linear(int(length) * num_features, 1)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        features0 = self.features0(x)
        features0 = features0.permute(0, 2, 1)
        features1, (h_n, c_n) = self.lstm(features0)
        features1 = features1.permute(0, 2, 1)
        features1 = self.features(features1)
        out = F.relu(features1, inplace=True)
        out = F.avg_pool1d(out, kernel_size=7, stride=1, padding=3).view(out.size(0), -1)
        out = self.ratio(out)
        out = out.squeeze(-1)
        return out

class CNN_K15(nn.Module):
    r"""basic CNN model class, 4 convolutional layers.

    Args:
            input_nc (int)      -- the number of channels in input seq
            drop_rate (float)   -- the dropout rate of the model
            use_dropout (bool)  -- if use dropout layers
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
    """

    def __init__(self, input_nc = 4, channel_num = 200, drop_rate = 0.2, kernel_size = 5,
                      use_dropout = False, input_length = 100):
        super(CNN_K15, self).__init__()

        padding_num = int(kernel_size/2)
        model = [nn.Conv1d(input_nc, 100, kernel_size=kernel_size, stride=1, padding=padding_num, bias=False),
                 nn.BatchNorm1d(100),
                 nn.ReLU(),
                 nn.Conv1d(100, channel_num, kernel_size=kernel_size, stride=1, padding=padding_num, bias=False),
                 nn.BatchNorm1d(channel_num),
                 nn.ReLU(),
                 nn.MaxPool1d(kernel_size=2, stride=2)]
        model += [nn.Conv1d(channel_num, channel_num, kernel_size=kernel_size, stride=1, padding=padding_num, bias=False),
                 nn.BatchNorm1d(channel_num),
                 nn.ReLU(),
                 nn.Conv1d(channel_num, 10, kernel_size=kernel_size, stride=1, padding=padding_num, bias=False),
                 nn.BatchNorm1d(10),
                 nn.ReLU(),
                 nn.MaxPool1d(kernel_size=2, stride=2)]
        self.model = nn.Sequential(*model)
        self.Linear = nn.Sequential(nn.Linear(int(int(input_length/2)/2) * 10, 1))

    def forward(self, inputSeq):
        x = self.model(inputSeq)
        x = x.view(x.size(0), -1)
        # import pdb; pdb.set_trace()
        output = self.Linear(x)
        output = output.squeeze(-1)
        return output


class AttentionPool(nn.Module):
    def __init__(self, spacial_dim, embed_dim, num_heads, output_dim=None,q='mean',use_pos=True,hot_num=1):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim + hot_num, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.q = q
        self.use_pos = use_pos
        if q == 'cls':
            self.cls_token = nn.Parameter(torch.zeros(hot_num, 1, embed_dim))
        self.num_heads = num_heads

    def forward(self, x):
        x = x.permute(2, 0, 1)  # NCL -> LNC
        if self.q == 'mean':
            x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (L+1)NC
            hot_num = 1
        elif self.q == 'cls':
            B_size = x.size(1)
            hot_num = self.cls_token.size(0)
            cls_tokens = self.cls_token.expand(-1, B_size, -1)
            x = torch.cat([cls_tokens, x], dim=0)
        if x.size(0) > self.positional_embedding.size(0):
            x[:self.positional_embedding.size(0)] += self.positional_embedding[:, None, :]
        else:
            x = x + self.positional_embedding[:x.size(0), None, :].to(x.dtype)  # (L+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:hot_num], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        # return x.permute(1, 2, 0)  # LNC -> NCL
        return x.squeeze(0)
    
class DenseLSTM2(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    "Flexible" means that the model can process different length of input seqs to scalar,
    because it only uses the last output of LSTM to predict the scalar.

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """
    def __init__(self, input_nc=4, growth_rate=64,block_config=(2, 2, 4, 2),
                 num_init_features=64, bn_size=4, drop_rate=0, input_length=100):

        super(DenseLSTM2, self).__init__()

        # First convolution
        self.features0 = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv1d(input_nc, num_init_features, kernel_size=7, stride=1, padding=3, bias=False)),
            ('norm0', nn.BatchNorm1d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool1d(kernel_size=3, stride=2, padding=1)),
        ]))
        self.features = nn.Sequential(OrderedDict([]))
        # LSTM
        self.lstm = nn.Sequential(OrderedDict([]))
        self.lstm.add_module('lstm_layer', nn.LSTM(input_size=num_init_features, hidden_size=num_init_features,
                                                       num_layers=3, bias=True, batch_first=True, bidirectional=True))
        length = np.floor((input_length + 2 * 1 - 1 - 2)/2 + 1)
        # Each denseblock
        num_features = 2*num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
                length = np.floor((length - 1 - 1) / 2 + 1)

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm1d(num_features))

        # Seq_len pool layer
        # B*seq_len*feature -> B*feature
        self.pool = AttentionPool(embed_dim=num_features,num_heads=4,spacial_dim=int(length),q='cls')

        # Linear layer
        self.ratio = nn.Linear(num_features, 1)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        features0 = self.features0(x)
        features0 = features0.permute(0, 2, 1)
        features1, (h_n, c_n) = self.lstm(features0)
        features1 = features1.permute(0, 2, 1)
        features1 = self.features(features1)
        out = F.relu(features1, inplace=True)
        # out = self.pool(out).reshape(out.size(0), -1)
        out = self.pool(out)
        out = self.ratio(out)
        out = out.squeeze(-1)
        return out


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028
    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss
    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None

class ResBlock(nn.Module):

    def __init__(self, input_nc, output_nc, kernel_size=13, padding=6, bias=True):
        super(ResBlock, self).__init__()
        model = [nn.ReLU(inplace=False),
                 nn.Conv1d(input_nc, output_nc, kernel_size=kernel_size, padding=padding, bias=bias),
                 nn.ReLU(inplace=False),
                 nn.Conv1d(input_nc, output_nc, kernel_size=kernel_size, padding=padding, bias=bias),
                 ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return x + 0.3*self.model(x)


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class EncoderLayer(nn.Module):

    def __init__(self, emb_num, num_heads):
        super(EncoderLayer, self).__init__()
        self.attn = nn.MultiheadAttention(emb_num, num_heads)

    def forward(self, x):
        x1, self.weights = self.attn(x, x, x)
        return x1

class Generator(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=512, seqL=50, bias=True, layer_num=1, num_heads=16):
        super(Generator, self).__init__()
        self.ngf, self.seqL = ngf, seqL
        self.first_linear = nn.Linear(seqL*input_nc, seqL*ngf, bias=bias)
        self.layer_num = layer_num
        for i in range(self.layer_num):
            exec("self.layer_{} = EncoderLayer(ngf, {})".format(i, num_heads))
        model = [ResBlock(ngf, ngf),
                 ResBlock(ngf, ngf),
                 ResBlock(ngf, ngf),
                 ResBlock(ngf, ngf),
                 ResBlock(ngf, ngf),
                 nn.Conv1d(ngf, output_nc, 1),
                 nn.Softmax(dim=1), ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x1 = self.first_linear(x.view(x.size(0), -1)).reshape([-1, self.ngf, self.seqL])
        x_t = x1.permute(2, 0, 1)
        for i in range(self.layer_num):
            exec("x_t = self.layer_{}(x_t)".format(i))
        features0 = x_t.permute(1, 2, 0)
        return self.model(features0)

class Discriminator(nn.Module):

    def __init__(self, output_nc, ndf=512, seqL=50, bias=True, layer_num=1, num_heads=16):
        super(Discriminator, self).__init__()
        self.Conv1 = nn.Conv1d(output_nc, ndf, 1)
        self.layer_num = layer_num
        for i in range(self.layer_num):
            exec("self.layer_{} = EncoderLayer(ndf, {})".format(i, num_heads))
        model = [ResBlock(ndf, ndf),
                 ResBlock(ndf, ndf),
                 ResBlock(ndf, ndf),
                 ResBlock(ndf, ndf),
                 ResBlock(ndf, ndf), ]
        self.model = nn.Sequential(*model)
        self.last_linear = nn.Linear(seqL*ndf, 1, bias=bias)

    def forward(self, x):
        x1 = self.Conv1(x)
        x_t = x1.permute(2, 0, 1)
        for i in range(self.layer_num):
            exec("x_t = self.layer_{}(x_t)".format(i))
        x_t = x_t.permute(1, 2, 0)
        x_t = self.model(x_t)
        return self.last_linear(x_t.contiguous().view(x_t.size(0), -1))

class WGAN():

    def __init__(self, input_nc, output_nc, seqL=100, lr=1e-4, gpu_ids='0', l1_w=10):
        super(WGAN, self).__init__()
        self.gpu_ids = gpu_ids
        self.l1_w = l1_w
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.generator = Generator(input_nc, output_nc, seqL=seqL)
        self.discriminator = Discriminator(input_nc + output_nc, seqL=seqL)
        if len(gpu_ids) > 0:
            self.generator = self.generator.cuda()
            self.discriminator = self.discriminator.cuda()
        self.l1_loss = torch.nn.L1Loss()
        self.optim_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.9))
        self.optim_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))

    def backward_g(self, fake_x):
        for p in self.discriminator.parameters():
            p.requires_grad = False
        self.generator.zero_grad()
        self.fake_inputs = self.generator(fake_x)
        pred_fake = self.discriminator(torch.cat((fake_x, self.fake_inputs), 1))
        self.g_loss = -pred_fake.mean()
        self.g_l1 = self.l1_w*self.l1_loss(fake_x, self.fake_inputs)
        self.g_total_loss = self.g_loss + self.g_l1
        self.g_total_loss.backward()
        self.optim_g.step()

    def backward_d(self, real_x, fake_x):
        for p in self.discriminator.parameters():
            p.requires_grad = True
        self.fake_inputs = self.generator(fake_x)
        fakeAB = torch.cat((fake_x, self.fake_inputs), 1)
        realAB = torch.cat((fake_x, real_x), 1)
        self.discriminator.zero_grad()
        pred_fake, pred_real = self.discriminator(fakeAB), self.discriminator(realAB)
        self.d_loss = pred_fake.mean() - pred_real.mean()
        self.gp, gradients = cal_gradient_penalty(self.discriminator, realAB, fakeAB, device=self.device)
        self.d_total_loss = self.d_loss + self.gp
        self.d_total_loss.backward()
        self.optim_d.step()


