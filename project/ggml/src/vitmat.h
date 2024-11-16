#ifndef __VITMAT__H__
#define __VITMAT__H__
#include "ggml_engine.h"
#include "ggml_nn.h"

#pragma GCC diagnostic ignored "-Wformat-truncation"

/*
 Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) */

struct Normalize {
    // network hparams
    int inplace = False;

    // network params
    


    void create_weight_tensors(struct ggml_context* ctx) {
        
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
        
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    	// please implement forward by your self, please !!!

    	return x;
    }
};

/*
 MattingHead(
  (matting_convs): Sequential(
    (0): Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Conv2d(16, 1, kernel_size=(1, 1), stride=(1, 1))
  )
) */

struct MattingHead {
    // network hparams
    

    // network params
    struct ggml_tensor* matting_convs_0_weight;  // torch.float32, [16, 32, 3, 3] 
    struct ggml_tensor* matting_convs_0_bias;  // torch.float32, [16] 
    struct ggml_tensor* matting_convs_1_weight;  // torch.float32, [16] 
    struct ggml_tensor* matting_convs_1_bias;  // torch.float32, [16] 
    struct ggml_tensor* matting_convs_1_running_mean;  // torch.float32, [16] 
    struct ggml_tensor* matting_convs_1_running_var;  // torch.float32, [16] 
    struct ggml_tensor* matting_convs_3_weight;  // torch.float32, [1, 16, 1, 1] 
    struct ggml_tensor* matting_convs_3_bias;  // torch.float32, [1]


    void create_weight_tensors(struct ggml_context* ctx) {
        matting_convs_0_weight = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 3, 32, 16);
        matting_convs_0_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 16);
        matting_convs_1_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 16);
        matting_convs_1_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 16);
        matting_convs_1_running_mean = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 16);
        matting_convs_1_running_var = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 16);
        matting_convs_3_weight = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, 16, 1);
        matting_convs_3_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
        ggml_format_name(matting_convs_0_weight, "%s%s", prefix, "matting_convs.0.weight");
        ggml_format_name(matting_convs_0_bias, "%s%s", prefix, "matting_convs.0.bias");
        ggml_format_name(matting_convs_1_weight, "%s%s", prefix, "matting_convs.1.weight");
        ggml_format_name(matting_convs_1_bias, "%s%s", prefix, "matting_convs.1.bias");
        ggml_format_name(matting_convs_1_running_mean, "%s%s", prefix, "matting_convs.1.running_mean");
        ggml_format_name(matting_convs_1_running_var, "%s%s", prefix, "matting_convs.1.running_var");
        ggml_format_name(matting_convs_3_weight, "%s%s", prefix, "matting_convs.3.weight");
        ggml_format_name(matting_convs_3_bias, "%s%s", prefix, "matting_convs.3.bias");
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    	// please implement forward by your self, please !!!

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
    // network hparams
    

    // network params
    struct BasicConv3x3 conv;


    void create_weight_tensors(struct ggml_context* ctx) {
        conv.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
        snprintf(s, sizeof(s), "%s%s", prefix, "conv.");
        conv.setup_weight_names(s);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    	// please implement forward by your self, please !!!

    	return x;
    }
};

/*
 BasicConv3x3(
  (conv): Conv2d(68, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU()
) */

struct BasicConv3x3 {
    // network hparams
    

    // network params
    struct ggml_tensor* conv_weight;  // torch.float32, [32, 68, 3, 3] 
    struct ggml_tensor* bn_weight;  // torch.float32, [32] 
    struct ggml_tensor* bn_bias;  // torch.float32, [32] 
    struct ggml_tensor* bn_running_mean;  // torch.float32, [32] 
    struct ggml_tensor* bn_running_var;  // torch.float32, [32]


    void create_weight_tensors(struct ggml_context* ctx) {
        conv_weight = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 3, 68, 32);
        bn_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32);
        bn_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32);
        bn_running_mean = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32);
        bn_running_var = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
        ggml_format_name(conv_weight, "%s%s", prefix, "conv.weight");
        ggml_format_name(bn_weight, "%s%s", prefix, "bn.weight");
        ggml_format_name(bn_bias, "%s%s", prefix, "bn.bias");
        ggml_format_name(bn_running_mean, "%s%s", prefix, "bn.running_mean");
        ggml_format_name(bn_running_var, "%s%s", prefix, "bn.running_var");
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    	// please implement forward by your self, please !!!

    	return x;
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
    // network hparams
    

    // network params
    struct BasicConv3x3 convs_0;
    struct BasicConv3x3 convs_1;
    struct BasicConv3x3 convs_2;


    void create_weight_tensors(struct ggml_context* ctx) {
        convs_0.create_weight_tensors(ctx);
        convs_1.create_weight_tensors(ctx);
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

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    	// please implement forward by your self, please !!!

    	return x;
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
    // network hparams
    

    // network params
    struct ConvStream convstream;
    struct FusionBlock fusion_blks_0;
    struct FusionBlock fusion_blks_1;
    struct FusionBlock fusion_blks_2;
    struct FusionBlock fusion_blks_3;
    struct MattingHead matting_head;


    void create_weight_tensors(struct ggml_context* ctx) {
        convstream.create_weight_tensors(ctx);
        fusion_blks_0.create_weight_tensors(ctx);
        fusion_blks_1.create_weight_tensors(ctx);
        fusion_blks_2.create_weight_tensors(ctx);
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

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    	// please implement forward by your self, please !!!

    	return x;
    }
};

/*
 LayerNorm() */

struct LayerNorm {
    // network hparams
    float eps = 1e-06;

    // network params
    


    void create_weight_tensors(struct ggml_context* ctx) {
        
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
        
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    	// please implement forward by your self, please !!!

    	return x;
    }
};

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

struct ResBottleneckBlock {
    // network hparams
    int in_channels = 384;
    int out_channels = 384;

    // network params
    struct ggml_tensor* conv1_weight;  // torch.float32, [192, 384, 1, 1] 
    struct LayerNorm norm1;
    struct ggml_tensor* conv2_weight;  // torch.float32, [192, 192, 3, 3] 
    struct LayerNorm norm2;
    struct ggml_tensor* conv3_weight;  // torch.float32, [384, 192, 1, 1] 
    struct LayerNorm norm3;


    void create_weight_tensors(struct ggml_context* ctx) {
        conv1_weight = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, 384, 192);
        norm1.create_weight_tensors(ctx);
        conv2_weight = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 3, 192, 192);
        norm2.create_weight_tensors(ctx);
        conv3_weight = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, 192, 384);
        norm3.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
        ggml_format_name(conv1_weight, "%s%s", prefix, "conv1.weight");
        snprintf(s, sizeof(s), "%s%s", prefix, "norm1.");
        norm1.setup_weight_names(s);
        ggml_format_name(conv2_weight, "%s%s", prefix, "conv2.weight");
        snprintf(s, sizeof(s), "%s%s", prefix, "norm2.");
        norm2.setup_weight_names(s);
        ggml_format_name(conv3_weight, "%s%s", prefix, "conv3.weight");
        snprintf(s, sizeof(s), "%s%s", prefix, "norm3.");
        norm3.setup_weight_names(s);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    	// please implement forward by your self, please !!!

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
    

    // network params
    struct ggml_tensor* norm1_weight;  // torch.float32, [384] 
    struct ggml_tensor* norm1_bias;  // torch.float32, [384] 
    struct Attention attn;
    struct ggml_tensor* norm2_weight;  // torch.float32, [384] 
    struct ggml_tensor* norm2_bias;  // torch.float32, [384] 
    struct Mlp mlp;
    struct ResBottleneckBlock residual;


    void create_weight_tensors(struct ggml_context* ctx) {
        norm1_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 384);
        norm1_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 384);
        attn.create_weight_tensors(ctx);
        norm2_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 384);
        norm2_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 384);
        mlp.create_weight_tensors(ctx);
        residual.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
        ggml_format_name(norm1_weight, "%s%s", prefix, "norm1.weight");
        ggml_format_name(norm1_bias, "%s%s", prefix, "norm1.bias");
        snprintf(s, sizeof(s), "%s%s", prefix, "attn.");
        attn.setup_weight_names(s);
        ggml_format_name(norm2_weight, "%s%s", prefix, "norm2.weight");
        ggml_format_name(norm2_bias, "%s%s", prefix, "norm2.bias");
        snprintf(s, sizeof(s), "%s%s", prefix, "mlp.");
        mlp.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "residual.");
        residual.setup_weight_names(s);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    	// please implement forward by your self, please !!!

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
    // network hparams
    

    // network params
    struct ggml_tensor* fc1_weight;  // torch.float32, [1536, 384] 
    struct ggml_tensor* fc1_bias;  // torch.float32, [1536] 
    struct ggml_tensor* fc2_weight;  // torch.float32, [384, 1536] 
    struct ggml_tensor* fc2_bias;  // torch.float32, [384]


    void create_weight_tensors(struct ggml_context* ctx) {
        fc1_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 384, 1536);
        fc1_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1536);
        fc2_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1536, 384);
        fc2_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 384);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
        ggml_format_name(fc1_weight, "%s%s", prefix, "fc1.weight");
        ggml_format_name(fc1_bias, "%s%s", prefix, "fc1.bias");
        ggml_format_name(fc2_weight, "%s%s", prefix, "fc2.weight");
        ggml_format_name(fc2_bias, "%s%s", prefix, "fc2.bias");
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    	// please implement forward by your self, please !!!

    	return x;
    }
};

/*
 Attention(
  (qkv): Linear(in_features=384, out_features=1152, bias=True)
  (proj): Linear(in_features=384, out_features=384, bias=True)
) */

struct Attention {
    // network hparams
    int num_heads = 6;
    float scale = 0.125;

    // network params
    struct ggml_tensor* qkv_weight;  // torch.float32, [1152, 384] 
    struct ggml_tensor* qkv_bias;  // torch.float32, [1152] 
    struct ggml_tensor* proj_weight;  // torch.float32, [384, 384] 
    struct ggml_tensor* proj_bias;  // torch.float32, [384]


    void create_weight_tensors(struct ggml_context* ctx) {
        qkv_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 384, 1152);
        qkv_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1152);
        proj_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 384, 384);
        proj_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 384);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
        ggml_format_name(qkv_weight, "%s%s", prefix, "qkv.weight");
        ggml_format_name(qkv_bias, "%s%s", prefix, "qkv.bias");
        ggml_format_name(proj_weight, "%s%s", prefix, "proj.weight");
        ggml_format_name(proj_bias, "%s%s", prefix, "proj.bias");
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    	// please implement forward by your self, please !!!

    	return x;
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
    int window_size = 14;

    // network params
    struct ggml_tensor* norm1_weight;  // torch.float32, [384] 
    struct ggml_tensor* norm1_bias;  // torch.float32, [384] 
    struct Attention attn;
    struct ggml_tensor* norm2_weight;  // torch.float32, [384] 
    struct ggml_tensor* norm2_bias;  // torch.float32, [384] 
    struct Mlp mlp;


    void create_weight_tensors(struct ggml_context* ctx) {
        norm1_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 384);
        norm1_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 384);
        attn.create_weight_tensors(ctx);
        norm2_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 384);
        norm2_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 384);
        mlp.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
        ggml_format_name(norm1_weight, "%s%s", prefix, "norm1.weight");
        ggml_format_name(norm1_bias, "%s%s", prefix, "norm1.bias");
        snprintf(s, sizeof(s), "%s%s", prefix, "attn.");
        attn.setup_weight_names(s);
        ggml_format_name(norm2_weight, "%s%s", prefix, "norm2.weight");
        ggml_format_name(norm2_bias, "%s%s", prefix, "norm2.bias");
        snprintf(s, sizeof(s), "%s%s", prefix, "mlp.");
        mlp.setup_weight_names(s);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    	// please implement forward by your self, please !!!

    	return x;
    }
};

/*
 PatchEmbed(
  (proj): Conv2d(4, 384, kernel_size=(16, 16), stride=(16, 16))
) */

struct PatchEmbed {
    // network hparams
    

    // network params
    struct ggml_tensor* proj_weight;  // torch.float32, [384, 4, 16, 16] 
    struct ggml_tensor* proj_bias;  // torch.float32, [384]


    void create_weight_tensors(struct ggml_context* ctx) {
        proj_weight = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 16, 4, 384);
        proj_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 384);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
        ggml_format_name(proj_weight, "%s%s", prefix, "proj.weight");
        ggml_format_name(proj_bias, "%s%s", prefix, "proj.bias");
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    	// please implement forward by your self, please !!!

    	return x;
    }
};

/*
 ViT(
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
) */

struct ViT {
    // network hparams
    

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
        patch_embed.create_weight_tensors(ctx);
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

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    	// please implement forward by your self, please !!!

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

struct ViTMatte {
    // network hparams
    int MAX_H = 1024;
    int MAX_W = 2048;
    int MAX_TIMES = 16;

    // network params
    struct ViT backbone;
    struct DetailCapture decoder;
    struct Normalize normal;


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

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    	// please implement forward by your self, please !!!

    	return x;
    }
};

#endif // __VITMAT__H__
Using device CPU ...
Running model on cpu ...
