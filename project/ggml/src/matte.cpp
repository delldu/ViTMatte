#include "matte.h"

#define GGML_ENGINE_IMPLEMENTATION
#include <ggml_engine.h>
#define GGML_NN_IMPLEMENTATION
#include <ggml_nn.h>


ggml_tensor_t *attn_add_rel_pos(struct ggml_context* ctx, ggml_tensor_t* attn, ggml_tensor_t* q,
	ggml_tensor_t* rel_pos_h, ggml_tensor_t* rel_pos_w, int H, int W)
{
    attn = ggml_nn_arange(ctx, attn);
    q = ggml_nn_arange(ctx, q);
    rel_pos_h = ggml_nn_arange(ctx, rel_pos_h);
    rel_pos_w = ggml_nn_arange(ctx, rel_pos_w);

    ggml_tensor_dump("attn", attn);
    ggml_tensor_dump("q ", q);
    ggml_tensor_dump("rel_pos_h", rel_pos_h);
    ggml_tensor_dump("rel_pos_w", rel_pos_w);

    ggml_tensor_t* r_h = get_rel_pos(ctx, rel_pos_h, H/*qh*/, H/*kh*/);
    ggml_set_name(r_h, "r_h");
    ggml_set_output(r_h);
    ggml_tensor_dump("r_h", r_h);

    ggml_tensor_t* r_w = get_rel_pos(ctx, rel_pos_w, W/*qw*/, W/*kw*/);
    ggml_set_name(r_w, "r_w");
    ggml_set_output(r_w);
    ggml_tensor_dump("r_w", r_w);

    int B = (int)q->ne[2];
    int C = (int)q->ne[0]; // head_dim

    ggml_tensor_t* r_q = ggml_reshape_4d(ctx, q, C, W, H, B);
    ggml_set_name(r_q, "r_q");
    ggml_set_output(r_q);
    ggml_tensor_dump("r_q", r_q);

    // rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, r_h) # [120, 14, 14, 14]
    r_h = ggml_cont(ctx, ggml_transpose(ctx, r_h));
    ggml_tensor_t* rel_h = ggml_nn_mul_mat(ctx, r_q, r_h);
    ggml_set_name(rel_h, "rel_h");
    ggml_set_output(rel_h);
    ggml_tensor_dump("rel_h", rel_h);

    // rel_w = torch.einsum("bhwc,hkc->bhwk", r_q, r_w) # [120, 14, 14, 14]
    int K = r_w->ne[1]; // CKH -- ggml dimensions ...
    r_w = ggml_cont(ctx, ggml_transpose(ctx, r_w));
    r_q = ggml_reshape_4d(ctx, r_q, C, 1, W, B*H);
    ggml_tensor_t* rel_w = ggml_nn_mul_mat(ctx, r_q, r_w);
    rel_w = ggml_reshape_4d(ctx, rel_w, K, W, H, B);
    ggml_set_name(rel_w, "rel_w");
    ggml_set_output(rel_w);
    ggml_tensor_dump("rel_w", rel_w);

    // attn = (
    //     attn.view(B, H*W, H, W) + rel_h.reshape(B, H*W, H, 1) + rel_w.reshape(B, H*W, 1, W)
    // ).view(B, H * W, H * W)


    attn = ggml_reshape_4d(ctx, attn, W, H, W*H, B);
    rel_h = ggml_reshape_4d(ctx, rel_h, 1, H, W*H, B);
    rel_w = ggml_reshape_4d(ctx, rel_w, W, 1, W*H, B);

    attn = ggml_add(ctx, attn, rel_h);
    attn = ggml_add(ctx, attn, rel_w);
    attn = ggml_reshape_3d(ctx, attn, W*H, W*H, B);

    ggml_set_name(attn, "attn");
    ggml_set_output(attn);
    ggml_tensor_dump("attn", attn);

    printf("------------------------------------------------------------\n");

    // attn    f32 [2752, 2752, 6, 1],  (reshaped) (cont)
    // q     f32 [64, 2752, 6, 1],  (reshaped) (cont)
    // rel_pos_h    f32 [64, 63, 1, 1],  (reshaped) (cont)
    // rel_pos_w    f32 [64, 63, 1, 1],  (reshaped) (cont)

    // r_h    f16 [64, 64, 64, 1], r_h
    // r_w    f16 [64, 43, 43, 1], r_w
    // r_q    f32 [64, 43, 64, 6], r_q
    // rel_h    f32 [64, 43, 64, 6], rel_h

    // tensor [attn] size: [6, 2752, 2752], min: 0.0, max: 1.0, mean: 0.5
    // tensor [q] size: [6, 2752, 64], min: 0.0, max: 0.999999, mean: 0.5
    // tensor [rel_pos_h] size: [63, 64], min: 0.0, max: 0.999752, mean: 0.499876
    // tensor [rel_pos_w] size: [63, 64], min: 0.0, max: 0.999752, mean: 0.499876

    // tensor [r_h] size: [64, 64, 64], min: 0.0, max: 0.999752, mean: 0.499876

    // tensor [r_w] size: [43, 43, 64], min: 0.0, max: 0.999752, mean: 0.499876
    // tensor [r_q] size: [6, 64, 43, 64], min: 0.0, max: 0.999999, mean: 0.5
    // tensor [rel_h] size: [6, 64, 43, 64], min: 2e-05, max: 63.482182, mean: 16.443848

    // tensor [rel_w] size: [6, 64, 43, 43], min: 2e-05, max: 63.482182, mean: 16.003044
    // tensor [attn1] size: [6, 2752, 2752], min: 0.0, max: 1.0, mean: 0.5
    // tensor [attn2] size: [6, 2752, 2752], min: 0.000101, max: 127.96431, mean: 32.946892



    return attn;
}

// def upsample_like(src, tar):
//     return F.interpolate(src, size=tar.shape[2:], mode="bilinear", align_corners=True)

ggml_tensor_t *upsample_like(struct ggml_context* ctx, ggml_tensor_t* src, ggml_tensor_t* dst)
{
    int W = (int)dst->ne[0];
    int H = (int)dst->ne[1];
    int C = (int)src->ne[2];
    int B = (int)src->ne[3];

    return ggml_upscale_ext(ctx, src, W, H, C, B);
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
    abs_pos = ggml_upscale_ext(ctx, abs_pos, C, W, H, B);

    return abs_pos;
}


// GGML_API struct ggml_tensor * ggml_get_rel_pos(
//         struct ggml_context * ctx,
//         struct ggml_tensor  * a,
//         int                   qh,
//         int                   kh);
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
