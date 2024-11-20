#ifndef __VITMAT__H__
#define __VITMAT__H__
#include "ggml_engine.h"
#include "ggml_nn.h"

#pragma GCC diagnostic ignored "-Wformat-truncation"

#include <vector>

ggml_tensor_t *get_abs_pos(struct ggml_context* ctx, ggml_tensor_t* abs_pos, int H, int W);
ggml_tensor_t *attn_add_rel_pos(struct ggml_context* ctx, ggml_tensor_t* attn, ggml_tensor_t* q,
    ggml_tensor_t* rel_pos_h, ggml_tensor_t* rel_pos_w, int H, int W);
struct ggml_tensor* get_rel_pos(struct ggml_context* ctx, struct ggml_tensor* a, int qh, int kh);

/*
 BasicConv3x3(
  (conv): Conv2d(68, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU()
) */

struct BasicConv3x3 {
    int in_channels;
    int out_channels;
    int stride;

    struct Conv2d conv;
    struct BatchNorm2d bn;

    void create_weight_tensors(struct ggml_context* ctx) {
        conv.in_channels = in_channels;
        conv.out_channels = out_channels;
        conv.kernel_size = {3, 3};
        conv.stride = { stride, stride };
        conv.padding = { 1, 1 };
        // conv.dilation = { 1, 1 };
        // conv.is_depthwise = false;
        conv.has_bias = false;
        conv.create_weight_tensors(ctx);
 
        bn.num_features = out_channels;
        bn.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "conv.");
        conv.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "bn.");
        bn.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        x = conv.forward(ctx, x);
        x = bn.forward(ctx, x);
        x = ggml_relu(ctx, x);

        return x;
    }
};

/*
 Mlp(
  (fc1): Linear(in_features=384, out_features=1536, bias=True)
  (act): GELU(approximate='none')
  (fc2): Linear(in_features=1536, out_features=384, bias=True)
) */
struct Mlp {
    int in_features = 384;
    int hidden_features = 1536;

    // network params
    struct Linear fc1;
    struct Linear fc2;

    void create_weight_tensors(struct ggml_context* ctx) {
        fc1.in_features = in_features;
        fc1.out_features = hidden_features;
        fc1.has_bias = true;
        fc1.create_weight_tensors(ctx);

        fc2.in_features = hidden_features;
        fc2.out_features = in_features;
        fc2.has_bias = true;
        fc2.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "fc1.");
        fc1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "fc2.");
        fc2.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        x = fc1.forward(ctx, x);
        x = ggml_relu(ctx, x);
        x = fc2.forward(ctx, x);

        return x;
    }
};


struct MattingHead {
    // network params
    struct Conv2d conv_0;
    struct BatchNorm2d bn_1;
    struct Conv2d conv_3;

    /*
     MattingHead(
      (matting_convs): Sequential(
        (0): Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Conv2d(16, 1, kernel_size=(1, 1), stride=(1, 1))
      )
    ) */

    void create_weight_tensors(struct ggml_context* ctx) {
        conv_0.in_channels = 32;
        conv_0.out_channels = 16;
        conv_0.kernel_size = {3, 3};
        conv_0.stride = { 1, 1 };
        conv_0.padding = { 1, 1 };
        // conv_0.dilation = { 1, 1 };
        // conv_0.is_depthwise = false;
        // conv_0.has_bias = true;
        conv_0.create_weight_tensors(ctx);
 
        bn_1.num_features = 16;
        bn_1.create_weight_tensors(ctx);

        conv_3.in_channels = 16;
        conv_3.out_channels = 1;
        conv_3.kernel_size = {1, 1};
        conv_3.stride = { 1, 1 };
        conv_3.padding = { 0, 0 };
        // conv_3.dilation = { 1, 1 };
        // conv_3.is_depthwise = false;
        // conv_3.has_bias = true;
        conv_3.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
        snprintf(s, sizeof(s), "%s%s", prefix, "matting_convs.0.");
        conv_0.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "matting_convs.1.");
        bn_1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "matting_convs.3.");
        conv_3.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        CheckPoint("h1");
        ggml_tensor_dump("hx1", x);
        x = conv_0.forward(ctx, x);
        CheckPoint("h1");
        ggml_tensor_dump("hx2", x);

        x = bn_1.forward(ctx, x);
        CheckPoint("h1");
        ggml_tensor_dump("hx3", x);

        x = ggml_relu(ctx, x);
        ggml_tensor_dump("hx4", x);

        x = conv_3.forward(ctx, x);
        CheckPoint("h3");
        ggml_tensor_dump("hx5", x);

    	return x;
    }
};

/*
 FusionBlock(
  (conv): BasicConv3x3(
    (conv): Conv2d(68, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): ReLU()
  )
) */

struct FusionBlock {
    int in_channels;
    int out_channels;

    // network params
    struct BasicConv3x3 conv;

    void create_weight_tensors(struct ggml_context* ctx) {
        conv.in_channels = in_channels;
        conv.out_channels = out_channels;
        conv.stride = 1;
        conv.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
        snprintf(s, sizeof(s), "%s%s", prefix, "conv.");
        conv.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x, ggml_tensor_t* dn) {
        // up = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
        // out = torch.cat([D, up], dim=1)
        // out = self.conv(out)

        // return out   

        int W = (int)x->ne[0];
        int H = (int)x->ne[1];
        int C = (int)x->ne[2];
        int B = (int)x->ne[3];

        ggml_tensor_t *up = ggml_upscale_ext(ctx, x, 2*W, 2*H, C, B);
        CheckPoint("FusionBlock 1");
        ggml_tensor_dump("x", x);
        ggml_tensor_dump("dn", dn);
        ggml_tensor_dump("up", up);
        ggml_tensor_t *out = ggml_concat(ctx, dn, up, 2 /*dim on channel*/);
        CheckPoint("FusionBlock 2");

        out = conv.forward(ctx, out);

    	return out;
    }
};


/*
 ConvStream(
  (convs): ModuleList(
    (0): BasicConv3x3(
      (conv): Conv2d(4, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU()
    )
    (1): BasicConv3x3(
      (conv): Conv2d(48, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU()
    )
    (2): BasicConv3x3(
      (conv): Conv2d(96, 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU()
    )
  )
) */

struct ConvStream {
    // network params
    struct BasicConv3x3 convs_0;
    struct BasicConv3x3 convs_1;
    struct BasicConv3x3 convs_2;

    void create_weight_tensors(struct ggml_context* ctx) {
        convs_0.in_channels = 4;
        convs_0.out_channels = 48;
        convs_0.stride = 2;
        convs_0.create_weight_tensors(ctx);

        convs_1.in_channels = 48;
        convs_1.out_channels = 96;
        convs_1.stride = 2;
        convs_1.create_weight_tensors(ctx);

        convs_2.in_channels = 96;
        convs_2.out_channels = 192;
        convs_2.stride = 2;
        convs_2.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
        snprintf(s, sizeof(s), "%s%s", prefix, "convs.0.");
        convs_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "convs.1.");
        convs_1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "convs.2.");
        convs_2.setup_weight_names(s);
    }

    std::vector<ggml_tensor_t *> forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        std::vector<ggml_tensor_t *> xlist;
        xlist.push_back(x);

        x = convs_0.forward(ctx, x);
        xlist.push_back(x);

        x = convs_1.forward(ctx, x);
        xlist.push_back(x);

        x = convs_2.forward(ctx, x);
        xlist.push_back(x);

    	return xlist;
    }
};

/*
 DetailCapture(
  (convstream): ConvStream(
    (convs): ModuleList(
      (0): BasicConv3x3(
        (conv): Conv2d(4, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (1): BasicConv3x3(
        (conv): Conv2d(48, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (2): BasicConv3x3(
        (conv): Conv2d(96, 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
    )
  )
  (fusion_blks): ModuleList(
    (0): FusionBlock(
      (conv): BasicConv3x3(
        (conv): Conv2d(576, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
    )
    (1): FusionBlock(
      (conv): BasicConv3x3(
        (conv): Conv2d(352, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
    )
    (2): FusionBlock(
      (conv): BasicConv3x3(
        (conv): Conv2d(176, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
    )
    (3): FusionBlock(
      (conv): BasicConv3x3(
        (conv): Conv2d(68, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
    )
  )
  (matting_head): MattingHead(
    (matting_convs): Sequential(
      (0): Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
      (3): Conv2d(16, 1, kernel_size=(1, 1), stride=(1, 1))
    )
  )
) */

struct DetailCapture {
    // network params
    struct ConvStream convstream;
    struct FusionBlock fusion_blks_0;
    struct FusionBlock fusion_blks_1;
    struct FusionBlock fusion_blks_2;
    struct FusionBlock fusion_blks_3;
    struct MattingHead matting_head;


    void create_weight_tensors(struct ggml_context* ctx) {
        convstream.create_weight_tensors(ctx);

        // [576, 256], [352, 128], [176, 64], [68, 32]
        fusion_blks_0.in_channels = 576;
        fusion_blks_0.out_channels = 256;
        fusion_blks_0.create_weight_tensors(ctx);

        fusion_blks_1.in_channels = 352;
        fusion_blks_1.out_channels = 128;
        fusion_blks_1.create_weight_tensors(ctx);

        fusion_blks_2.in_channels = 176;
        fusion_blks_2.out_channels = 64;
        fusion_blks_2.create_weight_tensors(ctx);

        fusion_blks_3.in_channels = 68;
        fusion_blks_3.out_channels = 32;
        fusion_blks_3.create_weight_tensors(ctx);

        matting_head.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
        snprintf(s, sizeof(s), "%s%s", prefix, "convstream.");
        convstream.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "fusion_blks.0.");
        fusion_blks_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "fusion_blks.1.");
        fusion_blks_1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "fusion_blks.2.");
        fusion_blks_2.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "fusion_blks.3.");
        fusion_blks_3.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "matting_head.");
        matting_head.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* features, ggml_tensor_t* images) {
        // def forward(self, features, images):
        //     detail_features = self.convstream(images)
        //     for i, m in enumerate(self.fusion_blks): # len(self.fusion_blks) -- 4 -- 4
        //         features = m(features, detail_features[len(self.fusion_blks)-i-1]) # D3, D2, D1, D0
            
        //     return torch.sigmoid(self.matting_head(features))

        // tensor [features] size: [1, 384, 64, 43], min: -9.721466, max: 27.064917, mean: 0.036542
        // tensor [images] size: [1, 4, 1024, 688], min: -2.117904, max: 2.64, mean: -0.247312
        // detail_features is list: len = 4
        //     tensor [item] size: [1, 4, 1024, 688], min: -2.117904, max: 2.64, mean: -0.247312
        //     tensor [item] size: [1, 48, 512, 344], min: 0.0, max: 18.381886, mean: 0.468337
        //     tensor [item] size: [1, 96, 256, 172], min: 0.0, max: 19.386992, mean: 0.449869
        //     tensor [item] size: [1, 192, 128, 86], min: 0.0, max: 19.881739, mean: 0.502548
        // tensor [features2] size: [1, 32, 1024, 688], min: 0.0, max: 5.583046, mean: 0.26782

        ggml_tensor_dump("features", features);
        ggml_tensor_dump("images", images);

        std::vector<ggml_tensor_t *> detail_features = convstream.forward(ctx, images);
        ggml_tensor_dump("detail_features[3]", detail_features[3]);
        ggml_tensor_dump("detail_features[2]", detail_features[2]);
        ggml_tensor_dump("detail_features[1]", detail_features[1]);
        ggml_tensor_dump("detail_features[0]", detail_features[0]);

        // features    f32 [64, 384, 43, 1],  (permuted) (cont) (permuted) (cont)
        // images    f32 [688, 1024, 4, 1], 
        // detail_features[3]    f32 [86, 128, 192, 1], 
        // detail_features[2]    f32 [172, 256, 96, 1], 
        // detail_features[1]    f32 [344, 512, 48, 1], 
        // detail_features[0]    f32 [688, 1024, 4, 1], 



        features = fusion_blks_0.forward(ctx, features, detail_features[3]);
        ggml_tensor_dump("features0", features);
        features = fusion_blks_1.forward(ctx, features, detail_features[2]);
        ggml_tensor_dump("features1", features);

        features = fusion_blks_2.forward(ctx, features, detail_features[1]);
        ggml_tensor_dump("features2", features);
        features = fusion_blks_3.forward(ctx, features, detail_features[0]);
        ggml_tensor_dump("features3", features);

        features = matting_head.forward(ctx, features);
        ggml_tensor_dump("features4", features);

    	return features;
    }
};

/*
 LayerNorm() */

struct CustomLayerNorm {
    int64_t normalized_shape;
    float eps = 1e-6;

    ggml_tensor_t* w;
    ggml_tensor_t* b;

    void create_weight_tensors(ggml_context_t* ctx)
    {
        GGML_ASSERT(normalized_shape > 0);

        w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, normalized_shape);
        b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, normalized_shape);
    }

    void setup_weight_names(const char* prefix)
    {
        ggml_format_name(w, "%s%s", prefix, "weight");
        ggml_format_name(b, "%s%s", prefix, "bias");
    }

    ggml_tensor_t* forward(ggml_context_t* ctx, ggml_tensor_t* x)
    {
        int C = x->ne[2];

        // CheckPoint("norm1");
        ggml_tensor_t *u = ggml_nn_mean(ctx, x, 2/*dim on channel*/);
        ggml_tensor_t *d = ggml_sub(ctx, x, u);
        ggml_tensor_t *s = ggml_mul(ctx, d, d);

        // CheckPoint("norm2");
        s = ggml_nn_mean(ctx, s, 2 /*dim on channel*/);
        s = ggml_nn_add(ctx, s, eps);
        s = ggml_sqrt(ctx, s);
        x = ggml_div(ctx, d, s);

        // CheckPoint("norm3");
        // ggml_tensor_dump("norm3 x", x);
        // ggml_tensor_dump("norm3 w", w);
        // ggml_tensor_dump("norm3 b", b);

        // ------------------------------------------------
        w = ggml_reshape_4d(ctx, w, 1, 1, C, 1);
        b = ggml_reshape_4d(ctx, b, 1, 1, C, 1);

        x = ggml_mul(ctx, x, w);
        x = ggml_add(ctx, x, b);

        // CheckPoint("norm4");

        return x;
    }
};


struct ResBottleneckBlock {
    // network hparams
    int in_channels = 384;
    int out_channels = 384;
    int bottleneck_channels = 192;

    // network params
    struct Conv2d conv1;
    struct CustomLayerNorm norm1;
    struct Conv2d conv2;
    struct CustomLayerNorm norm2;
    struct Conv2d conv3;
    struct CustomLayerNorm norm3;

    /*
     ResBottleneckBlock(
      (conv1): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (norm1): LayerNorm()
      (act1): GELU(approximate='none')
      (conv2): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (norm2): LayerNorm()
      (act2): GELU(approximate='none')
      (conv3): Conv2d(192, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (norm3): LayerNorm()
    ) */
    void create_weight_tensors(struct ggml_context* ctx) {
        conv1.in_channels = in_channels;
        conv1.out_channels = bottleneck_channels;
        conv1.kernel_size = {1, 1};
        conv1.has_bias = false;
        conv1.create_weight_tensors(ctx);

        norm1.normalized_shape = bottleneck_channels;
        norm1.create_weight_tensors(ctx);

        conv2.in_channels = bottleneck_channels;
        conv2.out_channels = bottleneck_channels;
        conv2.padding = { 1, 1 };
        conv2.has_bias = false;
        conv2.create_weight_tensors(ctx);

        norm2.normalized_shape = bottleneck_channels;
        norm2.create_weight_tensors(ctx);

        conv3.in_channels = bottleneck_channels;
        conv3.out_channels = out_channels;
        conv3.kernel_size = {1, 1};
        conv3.has_bias = false;
        conv3.create_weight_tensors(ctx);

        norm3.normalized_shape = out_channels;
        norm3.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "conv1.");
        conv1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "norm1.");
        norm1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "conv2.");
        conv2.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "norm2.");
        norm2.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "conv3.");
        conv3.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "norm3.");
        norm3.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        // out = x
        // for layer in self.children():
        //     out = layer(out)

        // out = x + out
        // return out
        ggml_tensor_t *out = x;
        out = conv1.forward(ctx, out);
        CheckPoint("res1, bottleneck_channels = %d", bottleneck_channels);
        ggml_tensor_dump("==> res1", out);

        out = norm1.forward(ctx, out);
        out = ggml_relu(ctx, out);
        CheckPoint("res2");

        out = conv2.forward(ctx, out);
        out = norm2.forward(ctx, out);
        out = ggml_relu(ctx, out);
        CheckPoint("res3");

        out = conv3.forward(ctx, out);
        out = norm3.forward(ctx, out);
        CheckPoint("res4");

    	x = ggml_add(ctx, x, out);
        CheckPoint("res5");

        return x;        
    }
};

/*
 Attention(
  (qkv): Linear(in_features=384, out_features=1152, bias=True)
  (proj): Linear(in_features=384, out_features=384, bias=True)
) */
// xxxx_debug
struct Attention {
    // network hparams
    int dim = 384;
    int num_heads = 6;
    int input_height = 14;
    int input_width = 14;
    int head_dim = 64;
    float scale = 0.125;

    struct Linear qkv;
    struct Linear proj;

    // self.qkv = nn.Linear(dim, dim * 3, bias=True)
    // self.proj = nn.Linear(dim, dim)

    ggml_tensor_t *rel_pos_h;
    ggml_tensor_t *rel_pos_w;

    void create_weight_tensors(struct ggml_context* ctx) {
        qkv.in_features = dim;
        qkv.out_features = dim * 3;
        qkv.has_bias = true;
        qkv.create_weight_tensors(ctx);

        proj.in_features = dim;
        proj.out_features = dim;
        proj.has_bias = false;
        proj.create_weight_tensors(ctx);

        rel_pos_h = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64 /*head_dim*/, 2*input_height - 1);
        rel_pos_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64 /*head_dim*/, 2*input_width - 1);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "qkv.");
        qkv.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "proj.");
        proj.setup_weight_names(s);

        ggml_format_name(rel_pos_h, "%s%s", prefix, "rel_pos_h");
        ggml_format_name(rel_pos_w, "%s%s", prefix, "rel_pos_w");
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        // B, H, W, C = x.size(), [20, 14, 14, 384] --> [20, 196, 3*6, 64]
        int C = (int)x->ne[0];
        int W = (int)x->ne[1];
        int H = (int)x->ne[2];
        int B = (int)x->ne[3];

        ggml_tensor_t *y = qkv.forward(ctx, x);
        y = ggml_reshape_4d(ctx, y, head_dim, 3*num_heads, W*H, B); // torch -- [B, H*W, 18, 64]
        ggml_tensor_t *q = ggml_nn_slice(ctx, y, 1/*dim*/, 0*num_heads, 1*num_heads, 1 /*step*/);
        ggml_tensor_t *k = ggml_nn_slice(ctx, y, 1/*dim*/, 1*num_heads, 2*num_heads, 1 /*step*/);
        ggml_tensor_t *v = ggml_nn_slice(ctx, y, 1/*dim*/, 2*num_heads, 3*num_heads, 1 /*step*/);

        // q, k, v f32 [64, 6, 196, 20] --> [64, 196, 120]
        q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3)); // [64, 6, 196, 20] -> [64, 196, 6, 20]
        q = ggml_reshape_3d(ctx, q, head_dim, H*W, num_heads * B);

        k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3)); // [64, 6, 196, 20] -> [64, 196, 6, 20]
        k = ggml_reshape_3d(ctx, k, head_dim, H*W, num_heads * B);

        v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3)); // [64, 6, 196, 20] -> [64, 196, 6, 20]
        v = ggml_reshape_3d(ctx, v, head_dim, H*W, num_heads * B);

        ggml_tensor_t* q_scale = ggml_scale(ctx, q, scale);
        ggml_tensor_t* k_transpose = ggml_transpose(ctx, k); 
        ggml_tensor_t *attn = ggml_nn_mul_mat(ctx, q_scale, k_transpose);

        // ----------------------------------------------------
        attn = attn_add_rel_pos(ctx, attn, q, rel_pos_h, rel_pos_w, H, W);


        attn = ggml_soft_max(ctx, attn);

        ggml_tensor_dump("attn-1", attn);
        ggml_tensor_dump("v", v);
        attn = ggml_nn_mul_mat(ctx, attn, v); // [120, 196, 64]
        ggml_tensor_dump("attn-2", attn);

        // test_reshape_case
        // x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        // # [120, 196, 64] -> [20, 6, 14, 14, 64] -> [20, 14, 14, 6, 64] -> [20, 14, 14, 384]
        // # ggml: [120, 196, 64] -> [20, 6, 196, 64] -> [20, 196, 6, 64] -> [20, 196, 384] -> [20, 14, 14, 384]

        // x = self.proj(x)
        // # tensor [x] size: [20, 14, 14, 384], min: -477.986969, max: 1045.842041, mean: -1.808324
        x = ggml_reshape_4d(ctx, attn, head_dim, W*H, num_heads, B);
        ggml_tensor_dump("attn-3", x); // [64, 2752, 6, 1]

        x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3)); // [64, 196, 6, 20] -> [64, 6, 196, 20]
        ggml_tensor_dump("attn-4", x); // [64, 6, 2752, 1]
        x = ggml_reshape_3d(ctx, x, num_heads * head_dim, W*H, B); // [384, 196, 20]
        ggml_tensor_dump("attn-5", x); // [384, 2752, 1, 1]
        x = ggml_reshape_4d(ctx, x, num_heads * head_dim, W, H, B); // [384, 14, 14, 20]
        ggml_tensor_dump("attn-6", x); // [384, 43, 64, 1]

        x = proj.forward(ctx, x); // f32 [384, 14, 14, 20], 
        ggml_tensor_dump("attn-7", x); // [384, 43, 64, 1]

        // tensor [---- Attention output] size: [20, 14, 14, 384], min: -0.34561, max: 0.4639, mean: 0.000979
        // tensor [---- Attention output] size: [1, 64, 43, 384], min: -0.486019, max: 0.83199, mean: 0.019024

        return x;
    }
};

/*
 BlockForZeroWindow(
  (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
  (attn): Attention(
    (qkv): Linear(in_features=384, out_features=1152, bias=True)
    (proj): Linear(in_features=384, out_features=384, bias=True)
  )
  (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
  (mlp): Mlp(
    (fc1): Linear(in_features=384, out_features=1536, bias=True)
    (act): GELU(approximate='none')
    (fc2): Linear(in_features=1536, out_features=384, bias=True)
  )
  (residual): ResBottleneckBlock(
    (conv1): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (norm1): LayerNorm()
    (act1): GELU(approximate='none')
    (conv2): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (norm2): LayerNorm()
    (act2): GELU(approximate='none')
    (conv3): Conv2d(192, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (norm3): LayerNorm()
  )
) */

struct BlockForZeroWindow {
    // network hparams
    int dim = 384;
    int num_heads = 6;
    int input_size = 32;

    struct LayerNorm norm1;
    struct Attention attn;
    struct LayerNorm norm2;
    struct Mlp mlp;
    struct ResBottleneckBlock residual; 

    void create_weight_tensors(struct ggml_context* ctx) {
        norm1.normalized_shape = dim;
        norm1.create_weight_tensors(ctx);

        attn.dim = dim;
        attn.num_heads = num_heads;
        attn.input_height = input_size;
        attn.input_width = input_size;
        attn.create_weight_tensors(ctx);

        norm2.normalized_shape = dim;
        norm2.create_weight_tensors(ctx);

        mlp.in_features = dim;
        mlp.hidden_features = dim * 4;
        mlp.create_weight_tensors(ctx);

        residual.in_channels = dim;
        residual.out_channels = dim;
        residual.bottleneck_channels = dim/2;
        residual.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "norm1.");
        norm1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "attn.");
        attn.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "norm2.");
        norm2.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "mlp.");
        mlp.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "residual.");
        residual.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        // shortcut = x
        // x = self.norm1(x)

        // # Window partition
        // x = self.attn(x)

        // # Reverse window partition
        // x = shortcut + x
        // x = x + self.mlp(self.norm2(x))
        // x = self.residual(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        // return x

        ggml_tensor_t *shortcut = x;
        x = norm1.forward(ctx, x);
        x = attn.forward(ctx, x);

        x = ggml_add(ctx, x, shortcut);

        ggml_tensor_t *y = norm2.forward(ctx, x);

        y = mlp.forward(ctx, y);
        x = ggml_add(ctx, x, y);

        // # x.size() -- [1, 64, 43, 384]
        // # x.permute(0, 3, 1, 2).size() -- [1, 384, 64, 43]

        // x = self.residual(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        // # [1, 384, 64, 43] -> [1, 64, 43, 384]
        x = ggml_cont(ctx, ggml_permute(ctx, x, 2, 0, 1, 3)); // [384, 43, 64, 1] -> [43, 64, 384, 1]/[C, W, H, B] -> [W, H, C, B]
        // --xxx1    f32 [384, 43, 64, 1], 
        // --xxx2    f32 [43, 64, 384, 1],  (permuted)
        // # x.size() -- [1, 64, 43, 384]
        // # x.permute(0, 3, 1, 2).size() -- [1, 384, 64, 43]

        y = residual.forward(ctx, x);
        y = ggml_cont(ctx, ggml_permute(ctx, y, 1, 2, 0, 3)); // [43, 64, 384, 1]->[384, 43, 64, 1]/[W, H, C, B] -> [C, W, H, B]

        CheckPoint("BlockForZeroWindow 6");

    	return y;
    }
};

/*
 BlockForNormalWindow(
  (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
  (attn): Attention(
    (qkv): Linear(in_features=384, out_features=1152, bias=True)
    (proj): Linear(in_features=384, out_features=384, bias=True)
  )
  (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
  (mlp): Mlp(
    (fc1): Linear(in_features=384, out_features=1536, bias=True)
    (act): GELU(approximate='none')
    (fc2): Linear(in_features=1536, out_features=384, bias=True)
  )
) */

struct BlockForNormalWindow {
    // network hparams
    int dim = 384;
    int num_heads = 6;
    int window_size = 14;

    // network params
    struct LayerNorm norm1;
    struct Attention attn;
    struct LayerNorm norm2;
    struct Mlp mlp;

    void create_weight_tensors(struct ggml_context* ctx) {
        norm1.normalized_shape = dim;
        norm1.create_weight_tensors(ctx);

        attn.dim = dim;
        attn.num_heads = num_heads;
        attn.input_height = window_size;
        attn.input_width = window_size;
        attn.create_weight_tensors(ctx);

        norm2.normalized_shape = dim;
        norm2.create_weight_tensors(ctx);

        mlp.in_features == dim;
        mlp.hidden_features == 4*dim;
        mlp.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "norm1.");
        norm1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "attn.");
        attn.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "norm2.");
        norm2.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "mlp.");
        mlp.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        // shortcut = x
        // x = self.norm1(x)

        // # Window partition
        // # Support torch.jit.script
        // pad_h:int = 0
        // pad_w:int = 0
        // H:int = 0
        // W:int = 0
        // H, W = x.shape[1], x.shape[2]
        // x, pad_h, pad_w = window_partition(x, self.window_size)

        // x = self.attn(x)

        // # Reverse window partition
        // x = window_unpartition(x, self.window_size, (pad_h, pad_w), (H, W))

        // x = shortcut + x
        // x = x + self.mlp(self.norm2(x))
        // return x

        int C = (int)x->ne[0];
        int W = (int)x->ne[1];
        int H = (int)x->ne[2];
        int B = (int)x->ne[3];
        // self-attn: x1    f32 [384, 43, 64, 1], 

        ggml_tensor_t *shortcut = x;
        x = norm1.forward(ctx, x);
        x = ggml_win_part(ctx, x, window_size); // window_size -- 14
        x = attn.forward(ctx, x);
        // [384, 14, 14, 20]
        // tensor [self-attn: x] size: [20, 14, 14, 384], min: -7.174149, max: 1.417144, mean: -0.005509
        CheckPoint("BlockForNormalWindow 1");
        x = ggml_win_unpart(ctx, x, W, H, window_size); // window_size -- 14
        CheckPoint("BlockForNormalWindow 2");

        x = ggml_add(ctx, x, shortcut);

        ggml_tensor_t *y = norm2.forward(ctx, x);
        y = mlp.forward(ctx, y);

        return ggml_add(ctx, x, y);
    }
};

/*
 PatchEmbed(
  (proj): Conv2d(4, 384, kernel_size=(16, 16), stride=(16, 16))
) */

struct PatchEmbed {
    // network hparams
    int in_chans = 4;
    int embed_dim = 384;
    int patch_size = 16;

    struct Conv2d proj;

    void create_weight_tensors(struct ggml_context* ctx) {
        proj.in_channels = in_chans;
        proj.out_channels = embed_dim;

        proj.kernel_size = { patch_size, patch_size };
        proj.stride = { patch_size, patch_size };
        // proj.padding = { 0, 0 };
        // proj.dilation = { 1, 1 };
        // proj.is_depthwise = false;
        // proj.has_bias = true;
        proj.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "proj.");
        proj.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        // x = self.proj(x)
        // x = x.permute(0, 2, 3, 1) # [B, C, H, W] -> [B, H, W, C]
        // return x
        x = proj.forward(ctx, x);
        x = ggml_permute(ctx, x, 1, 2, 0, 3); // [W, H, C, B] -> [C, W, H, B] ---- [384, 43, 64, 1]

        // tensor [x0] size: [1, 4, 1024, 688], min: -2.117904, max: 2.64, mean: -0.247312
        // tensor [x1] size: [1, 384, 64, 43], min: -1.791317, max: 6.925103, mean: 0.017269
        // tensor [x2] size: [1, 64, 43, 384], min: -1.791317, max: 6.925103, mean: 0.017269

    	return ggml_cont(ctx, x);
    }
};

/*
 ViT(
  (patch_embed): PatchEmbed(
    (proj): Conv2d(4, 384, kernel_size=(16, 16), stride=(16, 16))
  )
  (blocks): ModuleList(
    (2): BlockForZeroWindow(
      (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
      (attn): Attention(
        (qkv): Linear(in_features=384, out_features=1152, bias=True)
        (proj): Linear(in_features=384, out_features=384, bias=True)
      )
      (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
      (mlp): Mlp(
        (fc1): Linear(in_features=384, out_features=1536, bias=True)
        (act): GELU(approximate='none')
        (fc2): Linear(in_features=1536, out_features=384, bias=True)
      )
      (residual): ResBottleneckBlock(
        (conv1): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (norm1): LayerNorm()
        (act1): GELU(approximate='none')
        (conv2): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (norm2): LayerNorm()
        (act2): GELU(approximate='none')
        (conv3): Conv2d(192, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (norm3): LayerNorm()
      )
    )
    (3-4): 2 x BlockForNormalWindow(
      (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
      (attn): Attention(
        (qkv): Linear(in_features=384, out_features=1152, bias=True)
        (proj): Linear(in_features=384, out_features=384, bias=True)
      )
      (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
      (mlp): Mlp(
        (fc1): Linear(in_features=384, out_features=1536, bias=True)
        (act): GELU(approximate='none')
        (fc2): Linear(in_features=1536, out_features=384, bias=True)
      )
    )
    (5): BlockForZeroWindow(
      (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
      (attn): Attention(
        (qkv): Linear(in_features=384, out_features=1152, bias=True)
        (proj): Linear(in_features=384, out_features=384, bias=True)
      )
      (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
      (mlp): Mlp(
        (fc1): Linear(in_features=384, out_features=1536, bias=True)
        (act): GELU(approximate='none')
        (fc2): Linear(in_features=1536, out_features=384, bias=True)
      )
      (residual): ResBottleneckBlock(
        (conv1): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (norm1): LayerNorm()
        (act1): GELU(approximate='none')
        (conv2): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (norm2): LayerNorm()
        (act2): GELU(approximate='none')
        (conv3): Conv2d(192, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (norm3): LayerNorm()
      )
    )
    (6-7): 2 x BlockForNormalWindow(
      (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
      (attn): Attention(
        (qkv): Linear(in_features=384, out_features=1152, bias=True)
        (proj): Linear(in_features=384, out_features=384, bias=True)
      )
      (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
      (mlp): Mlp(
        (fc1): Linear(in_features=384, out_features=1536, bias=True)
        (act): GELU(approximate='none')
        (fc2): Linear(in_features=1536, out_features=384, bias=True)
      )
    )
    (8): BlockForZeroWindow(
      (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
      (attn): Attention(
        (qkv): Linear(in_features=384, out_features=1152, bias=True)
        (proj): Linear(in_features=384, out_features=384, bias=True)
      )
      (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
      (mlp): Mlp(
        (fc1): Linear(in_features=384, out_features=1536, bias=True)
        (act): GELU(approximate='none')
        (fc2): Linear(in_features=1536, out_features=384, bias=True)
      )
      (residual): ResBottleneckBlock(
        (conv1): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (norm1): LayerNorm()
        (act1): GELU(approximate='none')
        (conv2): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (norm2): LayerNorm()
        (act2): GELU(approximate='none')
        (conv3): Conv2d(192, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (norm3): LayerNorm()
      )
    )
    (9-10): 2 x BlockForNormalWindow(
      (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
      (attn): Attention(
        (qkv): Linear(in_features=384, out_features=1152, bias=True)
        (proj): Linear(in_features=384, out_features=384, bias=True)
      )
      (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
      (mlp): Mlp(
        (fc1): Linear(in_features=384, out_features=1536, bias=True)
        (act): GELU(approximate='none')
        (fc2): Linear(in_features=1536, out_features=384, bias=True)
      )
    )
    (11): BlockForZeroWindow(
      (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
      (attn): Attention(
        (qkv): Linear(in_features=384, out_features=1152, bias=True)
        (proj): Linear(in_features=384, out_features=384, bias=True)
      )
      (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
      (mlp): Mlp(
        (fc1): Linear(in_features=384, out_features=1536, bias=True)
        (act): GELU(approximate='none')
        (fc2): Linear(in_features=1536, out_features=384, bias=True)
      )
      (residual): ResBottleneckBlock(
        (conv1): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (norm1): LayerNorm()
        (act1): GELU(approximate='none')
        (conv2): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (norm2): LayerNorm()
        (act2): GELU(approximate='none')
        (conv3): Conv2d(192, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (norm3): LayerNorm()
      )
    )
  )
) */

struct ViT {
    // network hparams

    // num_patches = (224 // patch_size) * (224 // patch_size) # 14 * 14 ==> 196
    // num_positions = (num_patches + 1) # 197

    // self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim)) # size() -- [1, 197, 384]

    ggml_tensor_t* pos_embed;

    // network params
    struct PatchEmbed patch_embed;
    struct BlockForNormalWindow blocks_0;
    struct BlockForNormalWindow blocks_1;

    struct BlockForZeroWindow blocks_2;
    struct BlockForNormalWindow blocks_3;
    struct BlockForNormalWindow blocks_4;
    struct BlockForZeroWindow blocks_5;
    struct BlockForNormalWindow blocks_6;
    struct BlockForNormalWindow blocks_7;
    struct BlockForZeroWindow blocks_8;
    struct BlockForNormalWindow blocks_9;
    struct BlockForNormalWindow blocks_10;
    struct BlockForZeroWindow blocks_11;


    void create_weight_tensors(struct ggml_context* ctx) {
        pos_embed = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 384, 197, 1);

        patch_embed.create_weight_tensors(ctx);
        // (0-1): 2 x BlockForNormalWindow(
        //   (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        //   (attn): Attention(
        //     (qkv): Linear(in_features=384, out_features=1152, bias=True)
        //     (proj): Linear(in_features=384, out_features=384, bias=True)
        //   )
        //   (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        //   (mlp): Mlp(
        //     (fc1): Linear(in_features=384, out_features=1536, bias=True)
        //     (act): GELU(approximate='none')
        //     (fc2): Linear(in_features=1536, out_features=384, bias=True)
        //   )
        // )
        blocks_0.create_weight_tensors(ctx);
        blocks_1.create_weight_tensors(ctx);

        blocks_2.create_weight_tensors(ctx);
        blocks_3.create_weight_tensors(ctx);
        blocks_4.create_weight_tensors(ctx);
        blocks_5.create_weight_tensors(ctx);
        blocks_6.create_weight_tensors(ctx);
        blocks_7.create_weight_tensors(ctx);
        blocks_8.create_weight_tensors(ctx);
        blocks_9.create_weight_tensors(ctx);
        blocks_10.create_weight_tensors(ctx);
        blocks_11.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        ggml_format_name(pos_embed, "%s%s", prefix, "pos_embed");

        snprintf(s, sizeof(s), "%s%s", prefix, "patch_embed.");
        patch_embed.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.0.");
        blocks_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.1.");
        blocks_1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.2.");
        blocks_2.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.3.");
        blocks_3.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.4.");
        blocks_4.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.5.");
        blocks_5.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.6.");
        blocks_6.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.7.");
        blocks_7.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.8.");
        blocks_8.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.9.");
        blocks_9.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.10.");
        blocks_10.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.11.");
        blocks_11.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        // # tensor [x] size: [1, 4, 1024, 688], min: -2.117904, max: 2.610589, mean: 0.374034
        // x = self.patch_embed(x) # B, H, W, C
        // x = x + get_abs_pos(self.pos_embed, (x.shape[1], x.shape[2])) # H, W ?

        // for blk in self.blocks:
        //     x = blk(x)

        // # tensor [x] size: [1, 64, 43, 384], min: -10.419563, max: 32.650574, mean: 0.029471
        // return x.permute(0, 3, 1, 2) # [1, 64, 43, 384] -> [1, 384, 64, 43] -- (B, C, H, W)

        CheckPoint("vit0");
        x = patch_embed.forward(ctx, x);

        int C = (int)x->ne[0];
        int W = (int)x->ne[1];
        int H = (int)x->ne[2];
        int B = (int)x->ne[3];
        ggml_tensor_t *rel_pos = get_abs_pos(ctx, pos_embed, H, W);
        x = ggml_add(ctx, x, rel_pos);
        // tensor [get_abs_pos] size: [1, 64, 43, 384], min: -1.792006, max: 6.911754, mean: 0.017271

        CheckPoint("vit1");
        x = blocks_0.forward(ctx, x);
        CheckPoint("vit2");
        x = blocks_1.forward(ctx, x);
        CheckPoint("vit3");
        x = blocks_2.forward(ctx, x);
        CheckPoint("vit3");
        x = blocks_3.forward(ctx, x);
        CheckPoint("vit4");
        x = blocks_4.forward(ctx, x);
        CheckPoint("vit5");
        x = blocks_5.forward(ctx, x);
        CheckPoint("vit6");
        x = blocks_6.forward(ctx, x);
        CheckPoint("vit7");
        x = blocks_7.forward(ctx, x);
        CheckPoint("vit8");
        x = blocks_8.forward(ctx, x);
        CheckPoint("vit9");
        x = blocks_9.forward(ctx, x);
        CheckPoint("vit10");
        x = blocks_10.forward(ctx, x);
        CheckPoint("vit11");
        x = blocks_11.forward(ctx, x);
        CheckPoint("vit12");

        // [1, 64, 43, 384] -> [1, 384, 64, 43]
        // # [384, 43, 64, 1] -> [43, 64, 384, 1]

        x = ggml_cont(ctx, ggml_permute(ctx, x, 2, 0, 1, 3)); // [C, W, H, B] -> [W, H, C, B]

        CheckPoint("vit13");

        return x;
    }
};

/*
 ViTMatte(
  (backbone): ViT(
    (patch_embed): PatchEmbed(
      (proj): Conv2d(4, 384, kernel_size=(16, 16), stride=(16, 16))
    )
    (blocks): ModuleList(
      (0-1): 2 x BlockForNormalWindow(
        (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        (attn): Attention(
          (qkv): Linear(in_features=384, out_features=1152, bias=True)
          (proj): Linear(in_features=384, out_features=384, bias=True)
        )
        (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=384, out_features=1536, bias=True)
          (act): GELU(approximate='none')
          (fc2): Linear(in_features=1536, out_features=384, bias=True)
        )
      )
      (2): BlockForZeroWindow(
        (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        (attn): Attention(
          (qkv): Linear(in_features=384, out_features=1152, bias=True)
          (proj): Linear(in_features=384, out_features=384, bias=True)
        )
        (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=384, out_features=1536, bias=True)
          (act): GELU(approximate='none')
          (fc2): Linear(in_features=1536, out_features=384, bias=True)
        )
        (residual): ResBottleneckBlock(
          (conv1): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (norm1): LayerNorm()
          (act1): GELU(approximate='none')
          (conv2): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (norm2): LayerNorm()
          (act2): GELU(approximate='none')
          (conv3): Conv2d(192, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (norm3): LayerNorm()
        )
      )
      (3-4): 2 x BlockForNormalWindow(
        (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        (attn): Attention(
          (qkv): Linear(in_features=384, out_features=1152, bias=True)
          (proj): Linear(in_features=384, out_features=384, bias=True)
        )
        (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=384, out_features=1536, bias=True)
          (act): GELU(approximate='none')
          (fc2): Linear(in_features=1536, out_features=384, bias=True)
        )
      )
      (5): BlockForZeroWindow(
        (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        (attn): Attention(
          (qkv): Linear(in_features=384, out_features=1152, bias=True)
          (proj): Linear(in_features=384, out_features=384, bias=True)
        )
        (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=384, out_features=1536, bias=True)
          (act): GELU(approximate='none')
          (fc2): Linear(in_features=1536, out_features=384, bias=True)
        )
        (residual): ResBottleneckBlock(
          (conv1): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (norm1): LayerNorm()
          (act1): GELU(approximate='none')
          (conv2): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (norm2): LayerNorm()
          (act2): GELU(approximate='none')
          (conv3): Conv2d(192, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (norm3): LayerNorm()
        )
      )
      (6-7): 2 x BlockForNormalWindow(
        (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        (attn): Attention(
          (qkv): Linear(in_features=384, out_features=1152, bias=True)
          (proj): Linear(in_features=384, out_features=384, bias=True)
        )
        (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=384, out_features=1536, bias=True)
          (act): GELU(approximate='none')
          (fc2): Linear(in_features=1536, out_features=384, bias=True)
        )
      )
      (8): BlockForZeroWindow(
        (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        (attn): Attention(
          (qkv): Linear(in_features=384, out_features=1152, bias=True)
          (proj): Linear(in_features=384, out_features=384, bias=True)
        )
        (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=384, out_features=1536, bias=True)
          (act): GELU(approximate='none')
          (fc2): Linear(in_features=1536, out_features=384, bias=True)
        )
        (residual): ResBottleneckBlock(
          (conv1): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (norm1): LayerNorm()
          (act1): GELU(approximate='none')
          (conv2): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (norm2): LayerNorm()
          (act2): GELU(approximate='none')
          (conv3): Conv2d(192, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (norm3): LayerNorm()
        )
      )
      (9-10): 2 x BlockForNormalWindow(
        (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        (attn): Attention(
          (qkv): Linear(in_features=384, out_features=1152, bias=True)
          (proj): Linear(in_features=384, out_features=384, bias=True)
        )
        (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=384, out_features=1536, bias=True)
          (act): GELU(approximate='none')
          (fc2): Linear(in_features=1536, out_features=384, bias=True)
        )
      )
      (11): BlockForZeroWindow(
        (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        (attn): Attention(
          (qkv): Linear(in_features=384, out_features=1152, bias=True)
          (proj): Linear(in_features=384, out_features=384, bias=True)
        )
        (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=384, out_features=1536, bias=True)
          (act): GELU(approximate='none')
          (fc2): Linear(in_features=1536, out_features=384, bias=True)
        )
        (residual): ResBottleneckBlock(
          (conv1): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (norm1): LayerNorm()
          (act1): GELU(approximate='none')
          (conv2): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (norm2): LayerNorm()
          (act2): GELU(approximate='none')
          (conv3): Conv2d(192, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (norm3): LayerNorm()
        )
      )
    )
  )
  (decoder): DetailCapture(
    (convstream): ConvStream(
      (convs): ModuleList(
        (0): BasicConv3x3(
          (conv): Conv2d(4, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU()
        )
        (1): BasicConv3x3(
          (conv): Conv2d(48, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU()
        )
        (2): BasicConv3x3(
          (conv): Conv2d(96, 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU()
        )
      )
    )
    (fusion_blks): ModuleList(
      (0): FusionBlock(
        (conv): BasicConv3x3(
          (conv): Conv2d(576, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU()
        )
      )
      (1): FusionBlock(
        (conv): BasicConv3x3(
          (conv): Conv2d(352, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU()
        )
      )
      (2): FusionBlock(
        (conv): BasicConv3x3(
          (conv): Conv2d(176, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU()
        )
      )
      (3): FusionBlock(
        (conv): BasicConv3x3(
          (conv): Conv2d(68, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU()
        )
      )
    )
    (matting_head): MattingHead(
      (matting_convs): Sequential(
        (0): Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Conv2d(16, 1, kernel_size=(1, 1), stride=(1, 1))
      )
    )
  )
  (normal): Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
) */

struct ViTMatte : GGMLNetwork {
    // network hparams
    int MAX_H = 1024;
    int MAX_W = 2048;
    int MAX_TIMES = 16;

    // network params
    struct ViT backbone;
    struct DetailCapture decoder;
    struct Normalize normal;

    size_t get_graph_size()
    {
        return GGML_DEFAULT_GRAPH_SIZE * 4; // 2048 * 4
    }

    void create_weight_tensors(struct ggml_context* ctx) {
        backbone.create_weight_tensors(ctx);
        decoder.create_weight_tensors(ctx);
        normal.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
        snprintf(s, sizeof(s), "%s%s", prefix, "backbone.");
        backbone.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "decoder.");
        decoder.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "normal.");
        normal.setup_weight_names(s);
    }

    // ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
    ggml_tensor_t* forward(ggml_context_t* ctx, int argc, ggml_tensor_t* argv[]) {
        GGML_UNUSED(argc);
        ggml_tensor_t* x = argv[0];
        // '''
        //     x is Bx4xHxW tensor, alpha channel of x is trimap
        // '''
        // B, C, H, W = x.size() # [1, 4, 672, 992]

        // pad_h = (self.MAX_TIMES - (H % self.MAX_TIMES)) % self.MAX_TIMES
        // pad_w = (self.MAX_TIMES - (W % self.MAX_TIMES)) % self.MAX_TIMES
        // x = F.pad(x, (0, pad_w, 0, pad_h), 'reflect')

        // # normalize
        // images = x[:, 0:3, :, :]
        // trimap = x[:, 3:4, :, :]
        // images = self.normal(images)

        // images = torch.cat((images, trimap), dim=1)
        // features = self.backbone(images)            # size() -- [1, 384, 42, 62]
        // mask = self.decoder(features, images)       # size() -- [1, 1, 672, 992]

        // output = torch.cat((x[:, 0:3, :, :], mask), dim=1)

        // return output[:, :, 0:H, 0:W]
        int W = (int)x->ne[0];
        int H = (int)x->ne[1];
        int C = (int)x->ne[2];
        int B = (int)x->ne[3];
        int pad_h = (MAX_TIMES - (H % MAX_TIMES)) % MAX_TIMES;
        int pad_w = (MAX_TIMES - (W % MAX_TIMES)) % MAX_TIMES;
        if (pad_h > 0 || pad_w > 0) {
            x = ggml_replication_pad2d(ctx, x, 0, pad_w, 0, pad_h);
        }
        ggml_tensor_t *images = ggml_nn_slice(ctx, x, 2/*dim on channels*/, 0, 3, 1/*step*/);
        ggml_tensor_t *trimap = ggml_nn_slice(ctx, x, 2/*dim on channels*/, 3, 4, 1/*step*/);
        ggml_tensor_t *x3 = ggml_dup(ctx, images);

        images = normal.forward(ctx, images);
        CheckPoint("concat 1");
        images = ggml_concat(ctx, images, trimap, 2 /*dim on channel*/);
        CheckPoint("concat 2");

        // ----------------------------------------------

        ggml_tensor_t* features = backbone.forward(ctx, images);
        // return features; // xxxx_debug

        ggml_tensor_t* mask = decoder.forward(ctx, features, images);
        mask = ggml_constant(ctx, mask, 1.0);

        CheckPoint("concat 3");
        ggml_tensor_t* output = ggml_concat(ctx, x3, mask, 2/*dim on channels*/);
        CheckPoint("concat 4");

        if (pad_h > 0 || pad_w > 0) {
            output = ggml_nn_slice(ctx, output, 0/*W*/, 0, W, 1/*step*/);
            output = ggml_nn_slice(ctx, output, 1/*H*/, 0, H, 1/*step*/);
        }
        output = ggml_cont(ctx, output);

        return output;
    }
};

#endif // __VITMAT__H__
