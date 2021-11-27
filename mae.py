import paddle
from paddle import nn
import paddle.nn.functional as F
from einops import repeat

from transformer import PatchEmbedding,get_sinusoid_encoding_table,Encoder,Identity

class MAEEncoder(nn.Layer):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_learnable_pos_emb=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        num_patches = (img_size // patch_size) * (img_size // patch_size)

        # create patch embedding with positional embedding
        self.patch_embedding = PatchEmbedding(img_size,
                                              patch_size,
                                              in_chans,
                                              embed_dim,
                                              drop_rate)
        # create multi head self-attention layers
        self.encoder = Encoder(embed_dim,
                               num_heads,
                               depth,
                               qkv_bias,
                               mlp_ratio,
                               drop_rate,
                               attn_drop_rate,
                               drop_path_rate)

        self.head = nn.Linear(embed_dim, embed_dim,
            weight_attr=paddle.framework.ParamAttr(initializer=nn.initializer.TruncatedNormal(std=.02)),
            bias_attr=paddle.framework.ParamAttr(initializer=nn.initializer.Constant(0))) if num_classes > 0 else Identity()


    def get_num_layers(self):
        return len(self.blocks)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, mask=None):
        x = self.patch_embedding(x)
        B, _, C = x.shape
        if mask is not None:
            x_vis = x[paddle.bitwise_not(mask)].reshape([B, -1, C]) # ~mask means visible
        else:
            x_vis = x.reshape([B, -1, C])
        x_vis, attn = self.encoder(x_vis)
        return x_vis

    def forward(self, x, mask=None):
        x = self.forward_features(x, mask)
        x = self.head(x)
        return x

class MAEDecoder(nn.Layer):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, patch_size=16, num_classes=768, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, num_patches=196,
                 ):
        super().__init__()
        self.num_classes = num_classes
        #assert num_classes == 3 * patch_size ** 2
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_size = patch_size

        self.decoder = Encoder(embed_dim,
                               num_heads,
                               depth,
                               qkv_bias,
                               mlp_ratio,
                               drop_rate,
                               attn_drop_rate,
                               drop_path_rate)

        self.head = nn.Linear(embed_dim, num_classes,
            weight_attr=paddle.framework.ParamAttr(initializer=nn.initializer.TruncatedNormal(std=.02)),
            bias_attr=paddle.framework.ParamAttr(initializer=nn.initializer.Constant(0))) if num_classes > 0 else Identity()



    def get_num_layers(self):
        return len(self.blocks)


    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, return_token_num=0):
        x, attn = self.decoder(x)
        if return_token_num >0:
            x = self.head(x[:, -return_token_num:]) # only return the mask tokens predict pixels
        else:
            x = self.head(x) # [B, N, 3*16^2]
        return x

class MAE(nn.Layer):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 encoder_in_chans=3, 
                 encoder_num_classes=0, 
                 encoder_embed_dim=768, 
                 encoder_depth=12,
                 encoder_num_heads=12, 
                 decoder_num_classes=768, 
                 decoder_embed_dim=512, 
                 decoder_depth=8,
                 decoder_num_heads=8, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, 
                 init_values=0.,
                 use_learnable_pos_emb=False):
        super().__init__()
        self.encoder = MAEEncoder(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=encoder_in_chans, 
            num_classes=encoder_num_classes, 
            embed_dim=encoder_embed_dim, 
            depth=encoder_depth,
            num_heads=encoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values,
            use_learnable_pos_emb=use_learnable_pos_emb)

        self.decoder = MAEDecoder(
            patch_size=patch_size, 
            num_classes=decoder_num_classes, 
            embed_dim=decoder_embed_dim, 
            depth=decoder_depth,
            num_heads=decoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values)

        num_patches =  (img_size // patch_size) * (img_size // patch_size)
        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias_attr=False)
        self.mask_token = paddle.create_parameter(
            shape=[1, 1, decoder_embed_dim],
            dtype='float32',
            default_initializer=paddle.nn.initializer.TruncatedNormal(std=.02))
           
        self.pos_embed =get_sinusoid_encoding_table(num_patches, decoder_embed_dim)

        self.norm =  nn.LayerNorm(decoder_embed_dim,epsilon=1e-6,
            weight_attr=paddle.framework.ParamAttr(initializer=nn.initializer.Constant(1)),
            bias_attr=paddle.framework.ParamAttr(initializer=nn.initializer.Constant(0)))

        # classifier head (for finetuning)
        w_attr_1 = paddle.ParamAttr(
            initializer=paddle.nn.initializer.KaimingUniform())
        b_attr_1 = paddle.ParamAttr(
            initializer=paddle.nn.initializer.KaimingUniform())
        self.classifier = nn.Linear(decoder_embed_dim,decoder_num_classes,weight_attr=w_attr_1,bias_attr=b_attr_1)

    def get_num_layers(self):
        return len(self.blocks)

    def forward(self, x, mask=None):
        if mask is not None:
            '''
            maskg = RandomMaskingGenerator((14,14), 0.75)
            mask = maskg()
            mask = paddle.to_tensor(mask).astype('bool')
            '''
            x_vis = self.encoder(x, mask) # [B, N_vis, C_e]
            x_vis = self.encoder_to_decoder(x_vis) # [B, N_vis, C_d]

            B, N, C = x_vis.shape
        
        # we don't unshuffle the correct visible token order, 
        # but shuffle the pos embedding accorddingly.
            expand_pos_embed = self.pos_embed.expand([B, -1, -1]).astype(x.dtype).clone().detach()
            pos_emd_vis = expand_pos_embed[paddle.bitwise_not(mask)].reshape([B, -1, C])
            pos_emd_mask = expand_pos_embed[mask].reshape([B, -1, C])
            x_full = paddle.concat([x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], axis=1)

            x = self.decoder(x_full, pos_emd_mask.shape[1]) # [B, N_mask, 3 * 16 * 16]
            return x
        else:
            x_vis = self.encoder(x) # [B, N_vis, C_e]
            #print(x_vis.shape)
            x_vis = self.encoder_to_decoder(x_vis) # [B, N_vis, C_d]

            B, N, C = x_vis.shape
            x = self.norm(x_vis)
            x = x.mean(1).reshape([B,-1])
            #print(x.shape)
            #print(x.shape)
            x = self.classifier(x)
            return x

def build_mae(config):
    model = MAE(img_size=config.DATA.IMAGE_SIZE,
                              patch_size=config.MODEL.TRANS.PATCH_SIZE,
                              encoder_in_chans=3,
                              encoder_embed_dim=config.MODEL.TRANS.ENCODER_EMBED_DIM,
                              encoder_depth=config.MODEL.TRANS.ENCODER_DEPTH,
                              encoder_num_heads=config.MODEL.TRANS.ENCODER_NUM_HEADS,
                              encoder_num_classes= config.MODEL.TRANS.ENCODER_NUM_CLASSES,
                              decoder_num_classes=config.MODEL.TRANS.DECODER_NUM_CLASSES, 
                              decoder_embed_dim=config.MODEL.TRANS.DECODER_EMBED_DIM, 
                              decoder_depth=config.MODEL.TRANS.DECODER_DEPTH,
                              decoder_num_heads=config.MODEL.TRANS.DECODER_NUM_HEADS, 
                              mlp_ratio=config.MODEL.TRANS.MLP_RATIO,
                              qkv_bias=config.MODEL.TRANS.QKV_BIAS)
    return model