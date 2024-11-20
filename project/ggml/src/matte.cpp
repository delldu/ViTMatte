#include "matte.h"

#define GGML_ENGINE_IMPLEMENTATION
#include <ggml_engine.h>
#define GGML_NN_IMPLEMENTATION
#include <ggml_nn.h>

ggml_tensor_t *attn_add_rel_pos(struct ggml_context* ctx, ggml_tensor_t* attn, ggml_tensor_t* q,
	ggml_tensor_t* rel_pos_h, ggml_tensor_t* rel_pos_w, int H, int W)
{
    ggml_tensor_t* r_h = get_rel_pos(ctx, rel_pos_h, H/*qh*/, H/*kh*/);
    ggml_tensor_t* r_w = get_rel_pos(ctx, rel_pos_w, W/*qw*/, W/*kw*/);

    int B = (int)q->ne[2];
    int C = (int)q->ne[0]; // head_dim

    ggml_tensor_t* r_q = ggml_cont(ctx, ggml_reshape_4d(ctx, q, C, W, H, B));

    // rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, r_h) # [120, 14, 14, 14]
    r_h = ggml_cont(ctx, ggml_transpose(ctx, r_h));
    ggml_tensor_t* rel_h = ggml_nn_mul_mat(ctx, r_q, r_h);

    // rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, r_w) # [120, 14, 14, 14]
    int K = r_w->ne[1]; // CKH -- ggml dimensions ...
    r_w = ggml_cont(ctx, ggml_transpose(ctx, r_w));
    r_q = ggml_reshape_4d(ctx, r_q, C, 1, W, B*H);
    ggml_tensor_t* rel_w = ggml_nn_mul_mat(ctx, r_q, r_w);
    rel_w = ggml_cont(ctx, ggml_reshape_4d(ctx, rel_w, K, W, H, B));

    // attn = (
    //     attn.view(B, H*W, H, W) + rel_h.reshape(B, H*W, H, 1) + rel_w.reshape(B, H*W, 1, W)
    // ).view(B, H * W, H * W)
    attn = ggml_reshape_4d(ctx, attn, W, H, W*H, B);
    rel_h = ggml_reshape_4d(ctx, rel_h, 1, H, W*H, B);
    rel_w = ggml_reshape_4d(ctx, rel_w, W, 1, W*H, B);

    attn = ggml_add(ctx, attn, rel_h);
    attn = ggml_add(ctx, attn, rel_w);
    attn = ggml_cont(ctx, ggml_reshape_3d(ctx, attn, W*H, W*H, B));

    return attn;
}

// def upsample_like(src, tar):
//     return F.interpolate(src, size=tar.shape[2:], mode="bilinear", align_corners=True)
ggml_tensor_t *upsample_like(struct ggml_context* ctx, ggml_tensor_t* src, ggml_tensor_t* dst)
{
    int W = (int)dst->ne[0];
    int H = (int)dst->ne[1];
    // int C = (int)src->ne[2];
    // int B = (int)src->ne[3];
    // return ggml_upscale_ext(ctx, src, W, H, C, B);

    src = ggml_interpolate(ctx, src, 0, W);
    src = ggml_interpolate(ctx, src, 1, H);
    return src;
}

ggml_tensor_t *get_abs_pos(struct ggml_context* ctx, ggml_tensor_t* abs_pos, int H, int W)
{
    int C = (int)abs_pos->ne[0];
    int HW = (int)abs_pos->ne[1] - 1;
    int B = (int)abs_pos->ne[2];

    abs_pos = ggml_nn_slice(ctx, abs_pos, 1/*dim*/, 0, HW, 1/*step*/);
    int size = (int)sqrtf((float)HW);
    GGML_ASSERT( HW == size * size);

    abs_pos = ggml_reshape_4d(ctx, abs_pos, C, size, size, B);
    // abs_pos = ggml_upscale_ext(ctx, abs_pos, C, W, H, B);
    abs_pos = ggml_interpolate(ctx, abs_pos, 1, W);
    abs_pos = ggml_interpolate(ctx, abs_pos, 2, H);

    return abs_pos; // (size, size) --> (H, W)
}


struct ggml_tensor* get_rel_pos(struct ggml_context* ctx, struct ggml_tensor* a, int qh, int kh)
{
    // M = 2 * Max(qh, kh) - 1;
    int M = (qh > kh)? 2*qh - 1 : 2*kh - 1;
    if (a->ne[1] != M) {
        a = ggml_interpolate(ctx, a, 1 /*dim*/, M);
    }
    a = ggml_cast(ctx, a, GGML_TYPE_F16);
    a = ggml_get_rel_pos(ctx, a, qh, kh);
    a = ggml_cast(ctx, a, GGML_TYPE_F32);

    return a;
}
