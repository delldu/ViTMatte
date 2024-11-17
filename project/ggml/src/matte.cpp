#include "matte.h"

#define GGML_ENGINE_IMPLEMENTATION
#include <ggml_engine.h>
#define GGML_NN_IMPLEMENTATION
#include <ggml_nn.h>


// def add_decomposed_rel_pos(attn, q, rel_pos_h, rel_pos_w, q_size:Tuple[int, int], k_size:Tuple[int, int]):
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

//     Rh = get_rel_pos(rel_pos_h, q_h, k_h)
//     Rw = get_rel_pos(rel_pos_w, q_w, k_w)

//     B, _, dim = q.shape # [120, 196, 64]
//     r_q = q.reshape(B, q_h, q_w, dim)
//     rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh) # [120, 14, 14, 14]
//     # tensor [r_q] size: [6, 64, 43, 64], min: -3.554034, max: 4.482345, mean: 0.045117
//     # tensor [Rh] size: [64, 64, 64], min: -0.172288, max: 0.167465, mean: -0.000355
//     # tensor [rel_h] size: [6, 64, 43, 64], min: -3.037493, max: 5.377393, mean: 0.004754

//     # r_q.size() -- [b=120, h=14, w=14, (c=64)]
//     # Rh.size() -- [h=14, k=14, (c=64)]
//     # ==> [b=120, h=14, w=14, k=14]

//     rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw) # [120, 14, 14, 14]
//     # tensor [r_q] size: [120, 14, 14, 64], min: -5.109172, max: 5.019894, mean: 0.006196
//     # tensor [Rw] size: [14, 14, 64], min: -0.137916, max: 0.143502, mean: 0.000514
//     # tensor [rel_w] size: [120, 14, 14, 14], min: -3.59078, max: 5.638031, mean: -0.029699
//     # r_q = [b, h, w, (c)]
//     # Rw [w, k, (c)]
//     # ==> [b, h, w, k]

//     # xxxx_debug
//     # print("B, q_h, q_w, k_h, k_w -- ", B, q_h, q_w, k_h, k_w)
//     # todos.debug.output_var("attn", attn)
//     # todos.debug.output_var("rel_h", rel_h)
//     # todos.debug.output_var("rel_w", rel_w)
//     # print("-" * 80)
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

ggml_tensor_t *add_decomposed_rel_pos(struct ggml_context* ctx, ggml_tensor_t* attn, ggml_tensor_t* q,
	ggml_tensor_t* rel_pos_h, ggml_tensor_t* rel_pos_w, int H, int W)
{

    ggml_tensor_t* Rh = ggml_get_rel_pos(ctx, rel_pos_h, H, W);
    ggml_tensor_t* Rw = ggml_get_rel_pos(ctx, rel_pos_w, H, W);

    int B = (int)q->ne[2];
    int C = (int)q->ne[0]; // head_dim
    // B, _, dim = q.shape # [120, 196, 64]
    // r_q = q.reshape(B, q_h, q_w, dim);
    ggml_tensor_t* r_q = ggml_reshape_4d(ctx, q, B, H, W, C);
    // test_ggml_einsum
    // rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh) # [120, 14, 14, 14]
    // b = b.permute(0, 2, 1) # hkc -> hck/ckh->kch: (1, 0, 2)
    Rh = ggml_permute(ctx, Rh, 1, 0, 2, 3);
    ggml_tensor_t* rel_h = ggml_nn_mul_mat(ctx, r_q, Rh);

    Rw = ggml_permute(ctx, Rw, 1, 0, 2, 3);
    ggml_tensor_t* rel_w = ggml_nn_mul_mat(ctx, r_q, Rw);

    // torch ---
   	// attn = attn.reshape(B, H*W, H, W)
    // rel_h = rel_h.reshape(B, H*W, H, 1).repeat(1, 1, 1, W)
    // rel_w = rel_w.reshape(B, H*W, 1, W).repeat(1, 1, H, 1)    
    attn = ggml_reshape_4d(ctx, attn, W, H, W*H, B);
    rel_h = ggml_reshape_4d(ctx, 1, H, W*H, B);
    rel_w = ggml_reshape_4d(ctx, W, 1, W*H, B);

    // test_add_rel_pos
    // g_y = ggml.ggml_add(ctx, g_attn, g_rel_h)
    // g_y = ggml.ggml_add(ctx, g_y, g_rel_w)
    // g_y = ggml.ggml_reshape_3d(ctx, g_y, H*W, H*W, B)    

    attn = ggml_add(ctx, attn, rel_h);
    attn = ggml_add(ctx, attn, rel_w);

    return ggml_reshape_3d(ctx, attn, W*H, W*H, B);
}