#include "matte.h"

#define GGML_ENGINE_IMPLEMENTATION
#include <ggml_engine.h>
#define GGML_NN_IMPLEMENTATION
#include <ggml_nn.h>


// def attn_add_rel_pos(attn, q, rel_pos_h, rel_pos_w, q_size:Tuple[int, int], k_size:Tuple[int, int]):
//     # tensor [attn] size: [120, 196, 196], min: -8.433104, max: 16.265987, mean: 1.114269
//     # tensor [q] size: [120, 196, 64], min: -7.24711, max: 7.226545, mean: 0.014577
//     # tensor [rel_pos_h] size: [27, 64], min: -0.012905, max: 0.012103, mean: -7.8e-05
//     # tensor [rel_pos_w] size: [27, 64], min: -0.013695, max: 0.012973, mean: -9.8e-05

//     # q_size = (14, 14)
//     # k_size = (14, 14)

//     q_h, q_w = q_size
//     k_h, k_w = k_size
//     # if not (q_h == 14 and q_w == 14 and k_h == 14 and k_w == 14):
//     #     (Pdb) k_size -- (64, 43)

//     r_h = get_rel_pos(rel_pos_h, q_h, k_h)
//     r_w = get_rel_pos(rel_pos_w, q_w, k_w)

//     B, _, dim = q.shape # [120, 196, 64]
//     r_q = q.reshape(B, q_h, q_w, dim)
//     rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, r_h) # [120, 14, 14, 14]
//     # tensor [r_q] size: [6, 64, 43, 64], min: -3.554034, max: 4.482345, mean: 0.045117
//     # tensor [r_h] size: [64, 64, 64], min: -0.172288, max: 0.167465, mean: -0.000355
//     # tensor [rel_h] size: [6, 64, 43, 64], min: -3.037493, max: 5.377393, mean: 0.004754

//     # r_q.size() -- [b=120, h=14, w=14, (c=64)]
//     # r_h.size() -- [h=14, k=14, (c=64)]
//     # ==> [b=120, h=14, w=14, k=14]

//     rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, r_w) # [120, 14, 14, 14]
//     # tensor [r_q] size: [120, 14, 14, 64], min: -5.109172, max: 5.019894, mean: 0.006196
//     # tensor [r_w] size: [14, 14, 64], min: -0.137916, max: 0.143502, mean: 0.000514
//     # tensor [rel_w] size: [120, 14, 14, 14], min: -3.59078, max: 5.638031, mean: -0.029699
//     # r_q = [b, h, w, (c)]
//     # r_w [w, k, (c)]
//     # ==> [b, h, w, k]

//     # B, q_h, q_w, k_h, k_w --  6 64 43 64 43
//     # tensor [attn] size: [6, 2752, 2752], min: -16.951553, max: 20.740995, mean: 1.040415
//     # tensor [rel_h] size: [6, 64, 43, 64], min: -3.037493, max: 5.377393, mean: 0.004754
//     # tensor [rel_w] size: [6, 64, 43, 43], min: -1.91948, max: 4.111428, mean: 0.006052

//     attn = (
//         attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
//     ).view(B, q_h * q_w, k_h * k_w)
//     # [120, 196, 196] -> [120, 14, 14, 14, 14] + ... -> [120, 196, 196]
//     # attn == ggml_add_rel_pos(attn, rel_w, rel_h) ?

//     return attn # attn.size() -- [120, 196, 196]

ggml_tensor_t *attn_add_rel_pos(struct ggml_context* ctx, ggml_tensor_t* attn, ggml_tensor_t* q,
	ggml_tensor_t* rel_pos_h, ggml_tensor_t* rel_pos_w, int H, int W)
{
    //     # tensor [attn] size: [120, 196, 196], min: -8.433104, max: 16.265987, mean: 1.114269
    //     # tensor [q] size: [120, 196, 64], min: -7.24711, max: 7.226545, mean: 0.014577
    //     # tensor [rel_pos_h] size: [27, 64], min: -0.012905, max: 0.012103, mean: -7.8e-05
    //     # tensor [rel_pos_w] size: [27, 64], min: -0.013695, max: 0.012973, mean: -9.8e-05
    ggml_tensor_t* r_h = ggml_get_rel_pos(ctx, rel_pos_h, H/*qh*/, W/*kh*/);
    ggml_tensor_t* r_w = ggml_get_rel_pos(ctx, rel_pos_w, H/*qw*/, W/*kw*/);

    int B = (int)q->ne[2];
    int C = (int)q->ne[0]; // head_dim
    ggml_tensor_t* r_q = ggml_reshape_4d(ctx, q, B, H, W, C);
    // rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, r_h) # [120, 14, 14, 14]
    r_h = ggml_permute(ctx, r_h, 1, 0, 2, 3);
    ggml_tensor_t* rel_h = ggml_nn_mul_mat(ctx, r_q, r_h);

    // rel_w = torch.einsum("bhwc,hkc->bhwk", r_q, r_w) # [120, 14, 14, 14]
    r_w = ggml_permute(ctx, r_w, 1, 0, 2, 3);
    ggml_tensor_t* rel_w = ggml_nn_mul_mat(ctx, r_q, r_w);

    // ------------------------------------------------------------------
    attn = ggml_reshape_4d(ctx, attn, W, H, W*H, B);
    rel_h = ggml_reshape_4d(ctx, rel_h, 1, H, W*H, B);
    rel_w = ggml_reshape_4d(ctx, rel_w, W, 1, W*H, B);

    attn = ggml_add(ctx, attn, rel_h);
    attn = ggml_add(ctx, attn, rel_w);

    return ggml_reshape_3d(ctx, attn, W*H, W*H, B);
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


// def get_abs_pos(abs_pos, hw:Tuple[int, int]):
//     # tensor [abs_pos] size: [1, 197, 384], min: -0.161103, max: 0.160729, mean: 1.1e-05
//     # hw = (64, 43)
//     h, w = hw
//     abs_pos = abs_pos[:, 1:] # ==> [1, 196, 384]
//     xy_num = abs_pos.shape[1] # ===> 196
//     size = int(math.sqrt(xy_num))
//     assert size * size == xy_num
//     assert size == 14

//     # abs_pos.reshape(1, size, size, -1).size() -- [1, 14, 14, 384]
//     new_abs_pos = F.interpolate(
//         abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2), #[] -> [1, 384, 14, 14]
//         size=(h, w),
//         mode="bicubic",
//         align_corners=False,
//     )
//     return new_abs_pos.permute(0, 2, 3, 1) # [1, 384, 64, 43] --> [1, 64, 43, 384]

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
