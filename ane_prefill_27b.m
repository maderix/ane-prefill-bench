// ANE Prefill Benchmark — Qwen 3.6 27B on Apple Neural Engine
//
// Standalone demo: reads GGUF directly, dequantizes per-layer to FP16,
// dispatches projections + FFN via ANE conv1x1 with pipelined staging.
// No external dependencies beyond system frameworks + libomp.
//
// Build: ./build.sh
// Run:   ./ane_prefill_27b <model.gguf> [seq_len=256]

#define ACCELERATE_NEW_LAPACK
#import <Foundation/Foundation.h>
#import <Accelerate/Accelerate.h>
#import <IOSurface/IOSurface.h>
#include <math.h>
#include <mach/mach_time.h>
#include <string.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <arm_neon.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/sysctl.h>
#include "ane_bridge.h"

static mach_timebase_info_data_t tb;
static double ticks_to_ms(uint64_t t) {
    return (double)t * tb.numer / tb.denom / 1e6;
}

// ═══════════════════════════════════════════════════════════════════
// Model constants (Qwen 3.6 27B)
// ═══════════════════════════════════════════════════════════════════
#define DIM         5120
#define INTER       17408
#define N_LAYERS    64
#define N_ATTN      16
#define N_DN        48
#define VOCAB_SIZE  248320
#define ATTN_Q_PROJ 12288   // 48 heads × 256 dim (from GGUF tensor dim)
#define KV_DIM      1024    // 4 KV heads × 256 dim
#define ATTN_HD     256
#define ATTN_HQ     24      // GGUF head_count
#define ATTN_HKV    4
#define Q_DIM       6144    // ATTN_HQ * ATTN_HD = 24*256 (= SSM_INNER)
#define SSM_INNER   6144    // ≠ DIM!
#define DN_H        48
#define DN_D        128
#define DN_NORM_DIM 128
#define GATE_OUT    96      // 2 * DN_H (alpha + beta)
#define CONV_DIM    10240   // DIM * 2
#define CONV_K      4
#define ROPE_PAIRS  64
#define ROPE_DIM    128
#define CHUNK_C     64
#define DN_DECAY    0.99f
#define ATTN_INTERVAL 4

static const int ATTN_LAYERS[] = {
    3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63
};
static int is_attn_layer(int idx) {
    for (int i = 0; i < N_ATTN; i++)
        if (ATTN_LAYERS[i] == idx) return 1;
    return 0;
}

// ═══════════════════════════════════════════════════════════════════
// GGUF Direct Reader
// ═══════════════════════════════════════════════════════════════════

enum {
    GGUF_TYPE_UINT8  = 0,  GGUF_TYPE_INT8   = 1,
    GGUF_TYPE_UINT16 = 2,  GGUF_TYPE_INT16  = 3,
    GGUF_TYPE_UINT32 = 4,  GGUF_TYPE_INT32  = 5,
    GGUF_TYPE_FLOAT32= 6,  GGUF_TYPE_BOOL   = 7,
    GGUF_TYPE_STRING = 8,  GGUF_TYPE_ARRAY  = 9,
    GGUF_TYPE_UINT64 = 10, GGUF_TYPE_INT64  = 11,
    GGUF_TYPE_FLOAT64= 12,
};

enum {
    GGML_TYPE_F32  = 0,  GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_K = 12, GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14, GGML_TYPE_Q8_0 = 8,
};

typedef struct {
    char name[128];
    uint32_t n_dims;
    uint64_t dims[4];
    uint32_t type;
    uint64_t offset;
    uint64_t n_elements;
} GGUFTensor;

typedef struct {
    uint8_t *mmap_base;
    size_t mmap_size;
    int fd;
    uint64_t n_tensors;
    GGUFTensor *tensors;
    size_t tensor_data_start;
} GGUFFile;

static const uint8_t *gguf_skip_value(const uint8_t *p, uint32_t type) {
    switch (type) {
        case GGUF_TYPE_UINT8: case GGUF_TYPE_INT8: case GGUF_TYPE_BOOL: return p + 1;
        case GGUF_TYPE_UINT16: case GGUF_TYPE_INT16: return p + 2;
        case GGUF_TYPE_UINT32: case GGUF_TYPE_INT32: case GGUF_TYPE_FLOAT32: return p + 4;
        case GGUF_TYPE_UINT64: case GGUF_TYPE_INT64: case GGUF_TYPE_FLOAT64: return p + 8;
        case GGUF_TYPE_STRING: {
            uint64_t slen;
            memcpy(&slen, p, 8);
            return p + 8 + slen;
        }
        case GGUF_TYPE_ARRAY: {
            uint32_t atype;
            uint64_t alen;
            memcpy(&atype, p, 4); p += 4;
            memcpy(&alen, p, 8); p += 8;
            for (uint64_t i = 0; i < alen; i++)
                p = gguf_skip_value(p, atype);
            return p;
        }
    }
    return p;
}

static int gguf_load(const char *path, GGUFFile *g) {
    g->fd = open(path, O_RDONLY);
    if (g->fd < 0) { perror("open gguf"); return -1; }
    struct stat st;
    fstat(g->fd, &st);
    g->mmap_size = st.st_size;
    g->mmap_base = (uint8_t *)mmap(NULL, g->mmap_size, PROT_READ, MAP_PRIVATE, g->fd, 0);
    if (g->mmap_base == MAP_FAILED) { perror("mmap"); close(g->fd); return -1; }
    madvise(g->mmap_base, g->mmap_size, MADV_NORMAL);

    const uint8_t *p = g->mmap_base;

    uint32_t magic;
    memcpy(&magic, p, 4); p += 4;
    if (magic != 0x46554747) { fprintf(stderr, "Bad GGUF magic\n"); return -1; }

    uint32_t version;
    memcpy(&version, p, 4); p += 4;

    uint64_t n_tensors, n_kv;
    memcpy(&n_tensors, p, 8); p += 8;
    memcpy(&n_kv, p, 8); p += 8;

    printf("  GGUF v%u: %llu tensors, %llu metadata\n",
           version, (unsigned long long)n_tensors, (unsigned long long)n_kv);

    // Skip all KV metadata
    for (uint64_t i = 0; i < n_kv; i++) {
        uint64_t klen;
        memcpy(&klen, p, 8); p += 8 + klen;
        uint32_t vtype;
        memcpy(&vtype, p, 4); p += 4;
        p = gguf_skip_value(p, vtype);
    }

    // Parse tensor info
    g->n_tensors = n_tensors;
    g->tensors = (GGUFTensor *)calloc(n_tensors, sizeof(GGUFTensor));
    for (uint64_t i = 0; i < n_tensors; i++) {
        GGUFTensor *t = &g->tensors[i];
        uint64_t nlen;
        memcpy(&nlen, p, 8); p += 8;
        size_t cpy = nlen < 127 ? nlen : 127;
        memcpy(t->name, p, cpy); t->name[cpy] = 0;
        p += nlen;

        memcpy(&t->n_dims, p, 4); p += 4;
        t->n_elements = 1;
        for (uint32_t d = 0; d < t->n_dims; d++) {
            memcpy(&t->dims[d], p, 8); p += 8;
            t->n_elements *= t->dims[d];
        }
        for (uint32_t d = t->n_dims; d < 4; d++) t->dims[d] = 1;

        memcpy(&t->type, p, 4); p += 4;
        memcpy(&t->offset, p, 8); p += 8;
    }

    g->tensor_data_start = ((size_t)(p - g->mmap_base) + 31) & ~31ULL;
    printf("  Tensor data at offset %zu (%.1f MB header)\n",
           g->tensor_data_start, g->tensor_data_start / 1e6);

    return 0;
}

static GGUFTensor *gguf_find(GGUFFile *g, const char *name) {
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        if (strcmp(g->tensors[i].name, name) == 0)
            return &g->tensors[i];
    }
    return NULL;
}

static const uint8_t *gguf_data(GGUFFile *g, GGUFTensor *t) {
    return g->mmap_base + g->tensor_data_start + t->offset;
}

static size_t gguf_tensor_raw_bytes(GGUFTensor *t) {
    size_t ne = t->n_elements;
    switch (t->type) {
        case GGML_TYPE_F32:  return ne * 4;
        case GGML_TYPE_F16:  return ne * 2;
        case GGML_TYPE_Q4_K: return (ne / 256) * 144;
        case GGML_TYPE_Q5_K: return (ne / 256) * 176;
        case GGML_TYPE_Q6_K: return (ne / 256) * 210;
        case GGML_TYPE_Q8_0: return (ne / 32) * 34;
        default: return ne * 2;
    }
}

static void gguf_prefault_tensor(GGUFFile *g, const char *name) {
    GGUFTensor *t = gguf_find(g, name);
    if (!t) return;
    const uint8_t *base = gguf_data(g, t);
    size_t len = gguf_tensor_raw_bytes(t);
    uintptr_t page_start = (uintptr_t)base & ~(uintptr_t)0xFFF;
    size_t page_len = ((uintptr_t)base + len - page_start + 0xFFF) & ~(size_t)0xFFF;
    madvise((void *)page_start, page_len, MADV_WILLNEED);
}

static void gguf_release_tensor(GGUFFile *g, const char *name) {
    GGUFTensor *t = gguf_find(g, name);
    if (!t) return;
    const uint8_t *base = gguf_data(g, t);
    size_t len = gguf_tensor_raw_bytes(t);
    uintptr_t page_start = (uintptr_t)base & ~(uintptr_t)0xFFF;
    size_t page_len = ((uintptr_t)base + len - page_start + 0xFFF) & ~(size_t)0xFFF;
    madvise((void *)page_start, page_len, MADV_FREE);
}

static void gguf_prefault_layer(GGUFFile *g, int layer, int is_attn) {
    char name[256];
    if (!is_attn) {
        snprintf(name, 256, "blk.%d.attn_qkv.weight", layer);   gguf_prefault_tensor(g, name);
        snprintf(name, 256, "blk.%d.attn_gate.weight", layer);   gguf_prefault_tensor(g, name);
        snprintf(name, 256, "blk.%d.ssm_out.weight", layer);     gguf_prefault_tensor(g, name);
    } else {
        snprintf(name, 256, "blk.%d.attn_q.weight", layer);      gguf_prefault_tensor(g, name);
        snprintf(name, 256, "blk.%d.attn_k.weight", layer);      gguf_prefault_tensor(g, name);
        snprintf(name, 256, "blk.%d.attn_v.weight", layer);      gguf_prefault_tensor(g, name);
        snprintf(name, 256, "blk.%d.attn_output.weight", layer); gguf_prefault_tensor(g, name);
    }
    snprintf(name, 256, "blk.%d.ffn_gate.weight", layer);        gguf_prefault_tensor(g, name);
    snprintf(name, 256, "blk.%d.ffn_up.weight", layer);          gguf_prefault_tensor(g, name);
    snprintf(name, 256, "blk.%d.ffn_down.weight", layer);        gguf_prefault_tensor(g, name);
}

static void gguf_release_layer(GGUFFile *g, int layer, int is_attn) {
    char name[256];
    if (!is_attn) {
        snprintf(name, 256, "blk.%d.attn_qkv.weight", layer);   gguf_release_tensor(g, name);
        snprintf(name, 256, "blk.%d.attn_gate.weight", layer);   gguf_release_tensor(g, name);
        snprintf(name, 256, "blk.%d.ssm_out.weight", layer);     gguf_release_tensor(g, name);
    } else {
        snprintf(name, 256, "blk.%d.attn_q.weight", layer);      gguf_release_tensor(g, name);
        snprintf(name, 256, "blk.%d.attn_k.weight", layer);      gguf_release_tensor(g, name);
        snprintf(name, 256, "blk.%d.attn_v.weight", layer);      gguf_release_tensor(g, name);
        snprintf(name, 256, "blk.%d.attn_output.weight", layer); gguf_release_tensor(g, name);
    }
    snprintf(name, 256, "blk.%d.ffn_gate.weight", layer);        gguf_release_tensor(g, name);
    snprintf(name, 256, "blk.%d.ffn_up.weight", layer);          gguf_release_tensor(g, name);
    snprintf(name, 256, "blk.%d.ffn_down.weight", layer);        gguf_release_tensor(g, name);
}

// ═══════════════════════════════════════════════════════════════════
// GGUF block format dequantization → FP16
// ═══════════════════════════════════════════════════════════════════

static float fp16_to_f32(uint16_t h) {
    _Float16 tmp;
    memcpy(&tmp, &h, 2);
    return (float)tmp;
}

#pragma pack(push, 1)
typedef struct { uint16_t d; uint16_t dmin; uint8_t scales[12]; uint8_t qs[128]; } block_q4_K;
typedef struct { uint16_t d; uint16_t dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; } block_q5_K;
typedef struct { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; uint16_t d; } block_q6_K;
typedef struct { uint16_t d; int8_t qs[32]; } block_q8_0;
#pragma pack(pop)

_Static_assert(sizeof(block_q4_K) == 144, "q4k");
_Static_assert(sizeof(block_q5_K) == 176, "q5k");
_Static_assert(sizeof(block_q6_K) == 210, "q6k");
_Static_assert(sizeof(block_q8_0) == 34, "q8_0");

static void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// Dequantize one row of Q4K GGUF blocks → FP16
// raw = pointer to start of tensor data (all blocks contiguous)
// M = number of rows, K = number of cols (elements per row)
// row = which row to dequantize
static void dequant_row_q4k_fp16(const uint8_t *raw, _Float16 *out,
                                  int64_t row, int64_t K) {
    int64_t n_blocks = K / 256;
    for (int64_t blk = 0; blk < n_blocks; blk++) {
        const block_q4_K *b = (const block_q4_K *)
            (raw + (row * n_blocks + blk) * 144);
        float d = fp16_to_f32(b->d);
        float dmin = fp16_to_f32(b->dmin);
        const uint8_t *q = b->qs;
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc0, m0, sc1, m1;
            get_scale_min_k4(is, b->scales, &sc0, &m0);
            get_scale_min_k4(is + 1, b->scales, &sc1, &m1);
            float d1 = d * sc0, min1 = dmin * m0;
            float d2 = d * sc1, min2 = dmin * m1;
            int base = (int)(blk * 256 + j);
            for (int l = 0; l < 32; l++)
                out[base + l] = (_Float16)(d1 * (q[l] & 0xF) - min1);
            for (int l = 0; l < 32; l++)
                out[base + 32 + l] = (_Float16)(d2 * (q[l] >> 4) - min2);
            q += 32;
            is += 2;
        }
    }
}

static void dequant_row_q5k_fp16(const uint8_t *raw, _Float16 *out,
                                  int64_t row, int64_t K) {
    int64_t n_blocks = K / 256;
    for (int64_t blk = 0; blk < n_blocks; blk++) {
        const block_q5_K *b = (const block_q5_K *)
            (raw + (row * n_blocks + blk) * 176);
        float d = fp16_to_f32(b->d);
        float dmin = fp16_to_f32(b->dmin);
        const uint8_t *ql = b->qs;
        const uint8_t *qh = b->qh;
        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc0, m0, sc1, m1;
            get_scale_min_k4(is, b->scales, &sc0, &m0);
            get_scale_min_k4(is + 1, b->scales, &sc1, &m1);
            float d1 = d * sc0, min1 = dmin * m0;
            float d2 = d * sc1, min2 = dmin * m1;
            int base = (int)(blk * 256 + j);
            for (int l = 0; l < 32; l++)
                out[base + l] = (_Float16)(d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - min1);
            for (int l = 0; l < 32; l++)
                out[base + 32 + l] = (_Float16)(d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - min2);
            ql += 32;
            is += 2;
            u1 <<= 2; u2 <<= 2;
        }
    }
}

static void dequant_row_q6k_fp16(const uint8_t *raw, _Float16 *out,
                                  int64_t row, int64_t K) {
    int64_t n_blocks = K / 256;
    for (int64_t blk = 0; blk < n_blocks; blk++) {
        const block_q6_K *b = (const block_q6_K *)
            (raw + (row * n_blocks + blk) * 210);
        float d = fp16_to_f32(b->d);
        const uint8_t *ql = b->ql;
        const uint8_t *qh = b->qh;
        const int8_t *sc = b->scales;
        _Float16 *y = out + blk * 256;
        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int8_t q1 = (int8_t)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = (int8_t)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l]      = (_Float16)(d * sc[is + 0] * q1);
                y[l + 32] = (_Float16)(d * sc[is + 2] * q2);
                y[l + 64] = (_Float16)(d * sc[is + 4] * q3);
                y[l + 96] = (_Float16)(d * sc[is + 6] * q4);
            }
            y += 128; ql += 64; qh += 32; sc += 8;
        }
    }
}

// Dequantize full tensor [M, K] from GGUF to FP16 row-major
// Dispatches based on tensor type
static void dequant_tensor_fp16(GGUFFile *g, GGUFTensor *t,
                                 _Float16 *out, int64_t M, int64_t K) {
    const uint8_t *raw = gguf_data(g, t);
    int type = t->type;
    dispatch_apply((size_t)M, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0),
        ^(size_t r) {
        _Float16 *row_out = out + r * K;
        switch (type) {
            case GGML_TYPE_Q4_K: dequant_row_q4k_fp16(raw, row_out, (int64_t)r, K); break;
            case GGML_TYPE_Q5_K: dequant_row_q5k_fp16(raw, row_out, (int64_t)r, K); break;
            case GGML_TYPE_Q6_K: dequant_row_q6k_fp16(raw, row_out, (int64_t)r, K); break;
            case GGML_TYPE_F16: {
                const uint16_t *fp16 = (const uint16_t *)raw + r * K;
                memcpy(row_out, fp16, K * 2);
                break;
            }
            case GGML_TYPE_F32: {
                const float *fp32 = (const float *)raw + r * K;
                for (int64_t k = 0; k < K; k++)
                    row_out[k] = (_Float16)fp32[k];
                break;
            }
            default:
                memset(row_out, 0, K * 2);
        }
    });
}

// Extract F32 tensor from GGUF (for norms, biases etc.)
static void extract_f32(GGUFFile *g, const char *name, float *out, int64_t n) {
    GGUFTensor *t = gguf_find(g, name);
    if (!t) {
        fprintf(stderr, "WARNING: tensor '%s' not found, filling 1.0\n", name);
        for (int64_t i = 0; i < n; i++) out[i] = 1.0f;
        return;
    }
    const uint8_t *raw = gguf_data(g, t);
    if (t->type == GGML_TYPE_F32) {
        memcpy(out, raw, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        const uint16_t *fp16 = (const uint16_t *)raw;
        for (int64_t i = 0; i < n; i++) out[i] = fp16_to_f32(fp16[i]);
    } else if (t->type == GGML_TYPE_Q8_0) {
        int64_t n_blocks = n / 32;
        for (int64_t b = 0; b < n_blocks; b++) {
            const block_q8_0 *blk = (const block_q8_0 *)(raw + b * 34);
            float d = fp16_to_f32(blk->d);
            for (int k = 0; k < 32; k++)
                out[b * 32 + k] = d * blk->qs[k];
        }
    } else {
        fprintf(stderr, "Cannot extract F32 from type %d for %s\n", t->type, name);
        for (int64_t i = 0; i < n; i++) out[i] = 1.0f;
    }
}

// ═══════════════════════════════════════════════════════════════════
// Tensor name helpers (Qwen 3.5/3.6 GGUF naming)
// ═══════════════════════════════════════════════════════════════════

static char _tn[256];
static const char *tn_dn_qkv(int l) { snprintf(_tn,256,"blk.%d.attn_qkv.weight",l); return _tn; }
static const char *tn_dn_gate(int l) { snprintf(_tn,256,"blk.%d.attn_gate.weight",l); return _tn; }
static const char *tn_dn_out(int l) { snprintf(_tn,256,"blk.%d.ssm_out.weight",l); return _tn; }
static const char *tn_dn_conv(int l) { snprintf(_tn,256,"blk.%d.ssm_conv1d.weight",l); return _tn; }
static const char *tn_dn_a(int l) { snprintf(_tn,256,"blk.%d.ssm_a",l); return _tn; }
static const char *tn_dn_dt(int l) { snprintf(_tn,256,"blk.%d.ssm_dt.bias",l); return _tn; }
static const char *tn_dn_norm_o(int l) { snprintf(_tn,256,"blk.%d.ssm_norm.weight",l); return _tn; }
static const char *tn_dn_alpha(int l) { snprintf(_tn,256,"blk.%d.ssm_alpha.weight",l); return _tn; }
static const char *tn_dn_beta(int l) { snprintf(_tn,256,"blk.%d.ssm_beta.weight",l); return _tn; }
static const char *tn_attn_q(int l) { snprintf(_tn,256,"blk.%d.attn_q.weight",l); return _tn; }
static const char *tn_attn_k(int l) { snprintf(_tn,256,"blk.%d.attn_k.weight",l); return _tn; }
static const char *tn_attn_v(int l) { snprintf(_tn,256,"blk.%d.attn_v.weight",l); return _tn; }
static const char *tn_attn_o(int l) { snprintf(_tn,256,"blk.%d.attn_output.weight",l); return _tn; }
static const char *tn_attn_qn(int l) { snprintf(_tn,256,"blk.%d.attn_q_norm.weight",l); return _tn; }
static const char *tn_attn_kn(int l) { snprintf(_tn,256,"blk.%d.attn_k_norm.weight",l); return _tn; }
static const char *tn_norm(int l) { snprintf(_tn,256,"blk.%d.attn_norm.weight",l); return _tn; }
static const char *tn_ffn_gate(int l) { snprintf(_tn,256,"blk.%d.ffn_gate.weight",l); return _tn; }
static const char *tn_ffn_up(int l) { snprintf(_tn,256,"blk.%d.ffn_up.weight",l); return _tn; }
static const char *tn_ffn_down(int l) { snprintf(_tn,256,"blk.%d.ffn_down.weight",l); return _tn; }
static const char *tn_ffn_norm(int l) { snprintf(_tn,256,"blk.%d.post_attention_norm.weight",l); return _tn; }

// ═══════════════════════════════════════════════════════════════════
// MIL kernel compilation (dynamic weight matmul)
// ═══════════════════════════════════════════════════════════════════

#define MIL_HDR \
    "program(1.3)\n" \
    "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, " \
    "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, " \
    "{\"coremltools-version\", \"9.0\"}})]\n{\n"

static ANEKernelHandle *compile_dyn_proj(int ic, int oc, int sp) {
    int spw = sp + oc;
    char mil[4096];
    snprintf(mil, sizeof(mil),
        MIL_HDR
        "    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
        "        tensor<int32, [4]> ba = const()[name=string(\"ba\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
        "        tensor<int32, [4]> sa = const()[name=string(\"sa\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n"
        "        tensor<fp16, [1,%d,1,%d]> act = slice_by_size(x=x,begin=ba,size=sa)[name=string(\"act\")];\n"
        "        tensor<int32, [4]> bw = const()[name=string(\"bw\"), val=tensor<int32, [4]>([0,0,0,%d])];\n"
        "        tensor<int32, [4]> sw = const()[name=string(\"sw\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n"
        "        tensor<fp16, [1,%d,1,%d]> wt = slice_by_size(x=x,begin=bw,size=sw)[name=string(\"wt\")];\n"
        "        tensor<int32, [4]> ra = const()[name=string(\"ra\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n"
        "        tensor<fp16, [1,1,%d,%d]> a2 = reshape(shape=ra,x=act)[name=string(\"a2\")];\n"
        "        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"
        "        tensor<fp16, [1,1,%d,%d]> a3 = transpose(perm=pm,x=a2)[name=string(\"a3\")];\n"
        "        tensor<int32, [4]> rw = const()[name=string(\"rw\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n"
        "        tensor<fp16, [1,1,%d,%d]> W = reshape(shape=rw,x=wt)[name=string(\"W\")];\n"
        "        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"
        "        tensor<fp16, [1,1,%d,%d]> yh = matmul(transpose_x=bF,transpose_y=bF,x=a3,y=W)[name=string(\"yh\")];\n"
        "        tensor<fp16, [1,1,%d,%d]> yt = transpose(perm=pm,x=yh)[name=string(\"yt\")];\n"
        "        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n"
        "        tensor<fp16, [1,%d,1,%d]> y = reshape(shape=ro,x=yt)[name=string(\"y\")];\n"
        "    } -> (y);\n}\n",
        ic, spw,
        ic, sp, ic, sp,
        sp, ic, oc, ic, oc,
        ic, sp, ic, sp,
        sp, ic,
        ic, oc, ic, oc,
        sp, oc, oc, sp,
        oc, sp, oc, sp);

    size_t isz = (size_t)ic * spw * 2;
    size_t osz = (size_t)oc * sp * 2;
    ANEKernelHandle *k = ane_bridge_compile(mil, strlen(mil), NULL, 0,
                                             1, &isz, 1, &osz);
    if (!k) printf("  COMPILE FAILED: proj [%d → %d] sp=%d\n", ic, oc, sp);
    return k;
}

// Conv1x1 dynamic weight kernel — 4× faster than matmul MIL at large shapes
// Weight in spatial dim, sliced + transposed to [oc, ic, 1, 1] conv format
static ANEKernelHandle *compile_dyn_conv(int ic, int oc, int sp) {
    int spw = sp + oc;
    char mil[4096];
    snprintf(mil, sizeof(mil),
        MIL_HDR
        "    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
        "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
        "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
        "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
        "        string pt = const()[name=string(\"pt\"), val=string(\"custom\")];\n"
        "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
        "        tensor<int32, [4]> ba = const()[name=string(\"ba\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
        "        tensor<int32, [4]> sa = const()[name=string(\"sa\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n"
        "        tensor<fp16, [1,%d,1,%d]> act = slice_by_size(x=x,begin=ba,size=sa)[name=string(\"act\")];\n"
        "        tensor<int32, [4]> bw = const()[name=string(\"bw\"), val=tensor<int32, [4]>([0,0,0,%d])];\n"
        "        tensor<int32, [4]> sw = const()[name=string(\"sw\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n"
        "        tensor<fp16, [1,%d,1,%d]> wt = slice_by_size(x=x,begin=bw,size=sw)[name=string(\"wt\")];\n"
        "        tensor<int32, [4]> r1 = const()[name=string(\"r1\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n"
        "        tensor<fp16, [1,1,%d,%d]> w2 = reshape(shape=r1,x=wt)[name=string(\"w2\")];\n"
        "        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"
        "        tensor<fp16, [1,1,%d,%d]> w3 = transpose(perm=pm,x=w2)[name=string(\"w3\")];\n"
        "        tensor<int32, [4]> r2 = const()[name=string(\"r2\"), val=tensor<int32, [4]>([%d,%d,1,1])];\n"
        "        tensor<fp16, [%d,%d,1,1]> W = reshape(shape=r2,x=w3)[name=string(\"W\")];\n"
        "        tensor<fp16, [1,%d,1,%d]> y = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W,x=act)[name=string(\"y\")];\n"
        "    } -> (y);\n}\n",
        ic, spw,
        ic, sp, ic, sp,
        sp, ic, oc, ic, oc,
        ic, oc, ic, oc,
        oc, ic,
        oc, ic, oc, ic,
        oc, sp);

    size_t isz = (size_t)ic * spw * 2;
    size_t osz = (size_t)oc * sp * 2;
    ANEKernelHandle *k = ane_bridge_compile(mil, strlen(mil), NULL, 0,
                                             1, &isz, 1, &osz);
    if (!k) printf("  COMPILE FAILED: conv [%d → %d] sp=%d\n", ic, oc, sp);
    return k;
}

// ═══════════════════════════════════════════════════════════════════
// Pre-dequant + pre-transpose for fast weight staging
// ═══════════════════════════════════════════════════════════════════

typedef struct { _Float16 *data; int ic, oc; } PreTransW;

static void transpose_fp16_blocked(_Float16 *dst, const _Float16 *src, int rows, int cols) {
    int nr = (rows + 31) / 32;
    int nc = (cols + 31) / 32;
    dispatch_apply(nr * nc, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0),
                   ^(size_t b) {
        int r0 = ((int)b / nc) * 32;
        int c0 = ((int)b % nc) * 32;
        int rend = r0 + 32 < rows ? r0 + 32 : rows;
        int cend = c0 + 32 < cols ? c0 + 32 : cols;
        for (int r = r0; r < rend; r++)
            for (int c = c0; c < cend; c++)
                dst[c * rows + r] = src[r * cols + c];
    });
}

// Dequant GGUF tensor and pre-transpose for dynamic weight staging
// GGUF stores as [ne[0], ne[1]] where ne[1] = M (rows), ne[0] = K (cols)
// Weight matrix W[M, K] maps input[K] → output[M]
// Pre-transposed for staging: [K, M] (ic=K, oc=M)
static void predequant_gguf(PreTransW *pw, GGUFFile *g, const char *name,
                             int64_t M, int64_t K, _Float16 *tmp) {
    GGUFTensor *t = gguf_find(g, name);
    if (!t) {
        fprintf(stderr, "WARNING: tensor %s not found\n", name);
        pw->ic = (int)K; pw->oc = (int)M;
        pw->data = (_Float16 *)calloc((size_t)K * M, 2);
        return;
    }
    pw->ic = (int)K; pw->oc = (int)M;
    pw->data = (_Float16 *)malloc((size_t)K * M * 2);
    dequant_tensor_fp16(g, t, tmp, M, K);
    transpose_fp16_blocked(pw->data, tmp, (int)M, (int)K);
    printf("    %s [%lld, %lld] %s → pre-transposed\n", name,
           (long long)M, (long long)K,
           t->type == GGML_TYPE_Q4_K ? "Q4K" :
           t->type == GGML_TYPE_Q5_K ? "Q5K" :
           t->type == GGML_TYPE_Q6_K ? "Q6K" : "?");
}

static void predequant_free(PreTransW *pw) {
    if (pw->data) { free(pw->data); pw->data = NULL; }
}

static void fast_stage(ANEKernelHandle *k, const PreTransW *pw,
                       const _Float16 *acts_f16, int sp) {
    int ic = pw->ic, oc = pw->oc;
    int spw = sp + oc;
    IOSurfaceRef inSurf = (IOSurfaceRef)ane_bridge_get_input_surface(k, 0);
    IOSurfaceLock(inSurf, 0, NULL);
    _Float16 *buf = (_Float16 *)IOSurfaceGetBaseAddress(inSurf);
    const _Float16 *wdata = pw->data;
    size_t total = (size_t)ic * (sp + oc) * 2;
    if (total > 4 * 1024 * 1024) {
        int nblk = (ic + 255) / 256;
        dispatch_apply((size_t)nblk,
            dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t b) {
            int d0 = (int)b * 256;
            int d1 = d0 + 256 < ic ? d0 + 256 : ic;
            for (int d = d0; d < d1; d++) {
                memcpy(buf + d * spw, acts_f16 + d * sp, sp * sizeof(_Float16));
                memcpy(buf + d * spw + sp, wdata + d * oc, oc * sizeof(_Float16));
            }
        });
    } else {
        for (int d = 0; d < ic; d++) {
            memcpy(buf + d * spw, acts_f16 + d * sp, sp * sizeof(_Float16));
            memcpy(buf + d * spw + sp, wdata + d * oc, oc * sizeof(_Float16));
        }
    }
    IOSurfaceUnlock(inSurf, 0, NULL);
}

// Forward declaration for pipeline thread
static inline void ffn_down_tile_stage_parallel(ANEKernelHandle *k,
    const _Float16 *acts, int sp, const _Float16 *wdata,
    int oc, int k_start, int k_len);

// ═══════════════════════════════════════════════════════════════════
// Staging pipeline — overlap DMA with ANE compute
// ═══════════════════════════════════════════════════════════════════

typedef enum { PIPE_NONE, PIPE_FAST, PIPE_DOWN } PipeType;

typedef struct {
    PipeType type;
    ANEKernelHandle *kernel;
    const PreTransW *pw;
    const _Float16 *acts;
    int sp;
    const _Float16 *wdata;
    int oc, k_start, k_len;
    bool pending;
    bool shutdown;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
} StagePipeline;

static StagePipeline g_pipe = {
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .cond = PTHREAD_COND_INITIALIZER
};

static void serial_stage(ANEKernelHandle *k, const PreTransW *pw,
                          const _Float16 *acts_f16, int sp) {
    int ic = pw->ic, oc = pw->oc;
    int spw = sp + oc;
    IOSurfaceRef inSurf = (IOSurfaceRef)ane_bridge_get_input_surface(k, 0);
    IOSurfaceLock(inSurf, 0, NULL);
    _Float16 *buf = (_Float16 *)IOSurfaceGetBaseAddress(inSurf);
    const _Float16 *wdata = pw->data;
    for (int d = 0; d < ic; d++) {
        memcpy(buf + d * spw, acts_f16 + d * sp, sp * sizeof(_Float16));
        memcpy(buf + d * spw + sp, wdata + d * oc, oc * sizeof(_Float16));
    }
    IOSurfaceUnlock(inSurf, 0, NULL);
}

static void serial_down_stage(ANEKernelHandle *k, const _Float16 *acts, int sp,
                                const _Float16 *wdata, int oc, int k_start, int k_len) {
    int spw = sp + oc;
    IOSurfaceRef inSurf = (IOSurfaceRef)ane_bridge_get_input_surface(k, 0);
    IOSurfaceLock(inSurf, 0, NULL);
    _Float16 *buf = (_Float16 *)IOSurfaceGetBaseAddress(inSurf);
    for (int d = 0; d < k_len; d++) {
        memcpy(buf + d * spw, acts + (k_start + d) * sp, sp * 2);
        memcpy(buf + d * spw + sp, wdata + (k_start + d) * oc, oc * 2);
    }
    IOSurfaceUnlock(inSurf, 0, NULL);
}

static void *pipe_thread_fn(void *arg) {
    StagePipeline *p = (StagePipeline *)arg;
    pthread_mutex_lock(&p->mutex);
    while (!p->shutdown) {
        while (!p->pending && !p->shutdown)
            pthread_cond_wait(&p->cond, &p->mutex);
        if (p->shutdown) break;
        pthread_mutex_unlock(&p->mutex);

        if (p->type == PIPE_FAST)
            serial_stage(p->kernel, p->pw, p->acts, p->sp);
        else if (p->type == PIPE_DOWN)
            serial_down_stage(p->kernel, p->acts, p->sp,
                               p->wdata, p->oc, p->k_start, p->k_len);

        pthread_mutex_lock(&p->mutex);
        p->pending = false;
        pthread_cond_signal(&p->cond);
    }
    pthread_mutex_unlock(&p->mutex);
    return NULL;
}

static void pipe_async_fast(ANEKernelHandle *k, const PreTransW *pw,
                             const _Float16 *acts, int sp) {
    pthread_mutex_lock(&g_pipe.mutex);
    while (g_pipe.pending)
        pthread_cond_wait(&g_pipe.cond, &g_pipe.mutex);
    g_pipe.type = PIPE_FAST;
    g_pipe.kernel = k;
    g_pipe.pw = pw;
    g_pipe.acts = acts;
    g_pipe.sp = sp;
    g_pipe.pending = true;
    pthread_cond_signal(&g_pipe.cond);
    pthread_mutex_unlock(&g_pipe.mutex);
}

static void pipe_async_down(ANEKernelHandle *k, const _Float16 *acts, int sp,
                              const _Float16 *wdata, int oc, int k_start, int k_len) {
    pthread_mutex_lock(&g_pipe.mutex);
    while (g_pipe.pending)
        pthread_cond_wait(&g_pipe.cond, &g_pipe.mutex);
    g_pipe.type = PIPE_DOWN;
    g_pipe.kernel = k;
    g_pipe.acts = acts;
    g_pipe.sp = sp;
    g_pipe.wdata = wdata;
    g_pipe.oc = oc;
    g_pipe.k_start = k_start;
    g_pipe.k_len = k_len;
    g_pipe.pending = true;
    pthread_cond_signal(&g_pipe.cond);
    pthread_mutex_unlock(&g_pipe.mutex);
}

static void pipe_wait(void) {
    pthread_mutex_lock(&g_pipe.mutex);
    while (g_pipe.pending)
        pthread_cond_wait(&g_pipe.cond, &g_pipe.mutex);
    pthread_mutex_unlock(&g_pipe.mutex);
}

// ═══════════════════════════════════════════════════════════════════
// CPU helper functions
// ═══════════════════════════════════════════════════════════════════

static void cpu_rmsnorm(float *out, const float *x, const float *w, int dim, int n) {
    for (int i = 0; i < n; i++) {
        const float *xi = x + i * dim;
        float *oi = out + i * dim;
        float ss = 0;
        for (int d = 0; d < dim; d++) ss += xi[d] * xi[d];
        float inv = 1.0f / sqrtf(ss / dim + 1e-6f);
        for (int d = 0; d < dim; d++) oi[d] = xi[d] * inv * w[d];
    }
}

static void transpose_to_f16(_Float16 *out, const float *in, int rows, int cols) {
    int nr = (rows + 31) / 32;
    int nc = (cols + 31) / 32;
    dispatch_apply((size_t)(nr * nc),
        dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t b) {
        int r0 = ((int)b / nc) * 32;
        int c0 = ((int)b % nc) * 32;
        int rend = r0 + 32 < rows ? r0 + 32 : rows;
        int cend = c0 + 32 < cols ? c0 + 32 : cols;
        for (int r = r0; r < rend; r++)
            for (int c = c0; c < cend; c++)
                out[c * rows + r] = (_Float16)in[r * cols + c];
    });
}

// Zero-copy read: access output IOSurface directly, NEON transpose [oc,sp] FP16 → [sp,oc] F32
// Uses contiguous-write order (inner loop over oc) to avoid cache thrashing.
static void ane_read_output(ANEKernelHandle *k, float *out, int oc, int sp) {
    IOSurfaceRef outSurf = (IOSurfaceRef)ane_bridge_get_output_surface(k, 0);
    IOSurfaceLock(outSurf, kIOSurfaceLockReadOnly, NULL);
    const _Float16 *src = (const _Float16 *)IOSurfaceGetBaseAddress(outSurf);
    #define BLK 32
    int nb_s = (sp + BLK - 1) / BLK;
    int nb_n = (oc + BLK - 1) / BLK;
    dispatch_apply((size_t)(nb_s * nb_n),
        dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t b) {
        int s0 = ((int)b / nb_n) * BLK;
        int n0 = ((int)b % nb_n) * BLK;
        int se = s0 + BLK < sp ? s0 + BLK : sp;
        int ne = n0 + BLK < oc ? n0 + BLK : oc;
        for (int s = s0; s < se; s++) {
            float *dst = out + s * oc + n0;
            int n = n0;
            for (; n + 3 < ne; n += 4) {
                float16x4_t v = {src[n * sp + s], src[(n+1) * sp + s],
                                 src[(n+2) * sp + s], src[(n+3) * sp + s]};
                vst1q_f32(dst + (n - n0), vcvt_f32_f16(v));
            }
            for (; n < ne; n++)
                dst[n - n0] = (float)src[n * sp + s];
        }
    });
    #undef BLK
    IOSurfaceUnlock(outSurf, kIOSurfaceLockReadOnly, NULL);
}

static void cpu_residual_add(float *hidden, const float *proj, int dim, int n) {
    int total = dim * n;
    int i = 0;
    for (; i + 3 < total; i += 4) {
        float32x4_t h = vld1q_f32(hidden + i);
        float32x4_t p = vld1q_f32(proj + i);
        vst1q_f32(hidden + i, vaddq_f32(h, p));
    }
    for (; i < total; i++) hidden[i] += proj[i];
}

// ═══════════════════════════════════════════════════════════════════
// DeltaNet CPU helpers (same algorithm as 9B, parameterized)
// ═══════════════════════════════════════════════════════════════════

// Gate projection: normed[sp, DIM] × gate_w^T[DIM, GATE_OUT] → raw_gate[sp, GATE_OUT]
static void cpu_gate_proj(const float *normed, const float *gate_w,
                           float *raw_gate, int sp) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                sp, GATE_OUT, DIM, 1.0f,
                normed, DIM, gate_w, DIM, 0.0f, raw_gate, GATE_OUT);
}

// Pre-gates: conv1d + SiLU on QKV output, then split into Q/K/V
//
// Conv output [CONV_DIM=10240] split (matching 9B pattern):
//   [0:2048]       = Q raw (QK_HEADS=16 heads × DN_D=128)
//   [2048:4096]    = K raw (QK_HEADS=16 heads × DN_D=128)
//   [4096:10240]   = V (DN_H=48 heads × DN_D=128 = SSM_INNER=6144)
// Q/K repeated 3× to fill DN_H=48 heads (same as 9B's 2× repeat from 16→32)
#define QK_HEADS  16
#define QK_DIM    (QK_HEADS * DN_D)   // 2048
#define QK_REPEAT (DN_H / QK_HEADS)   // 3

static void cpu_deltanet_pre_gates(
    const float *qkv_buf, const float *raw_gate, const float *conv_w,
    float *conv_state, const float *A_log, const float *dt_bias,
    float *q_out, float *k_out, float *v_out,
    float *beta_out, float *g_out, int sp)
{
    for (int s = 0; s < sp; s++) {
        // Shift conv state + new input (CONV_K=4, stride-4 layout)
        const float *inp = qkv_buf + s * CONV_DIM;
        for (int d = 0; d < CONV_DIM; d++) {
            float *cs = conv_state + d * CONV_K;
            cs[0] = cs[1]; cs[1] = cs[2]; cs[2] = cs[3];
            cs[3] = inp[d];
        }

        // Conv1d + SiLU: each dim is dot(state[d,0:4], weight[d,0:4])
        float conv_out[CONV_DIM];
        for (int d = 0; d < CONV_DIM; d++) {
            float32x4_t vs = vld1q_f32(conv_state + d * CONV_K);
            float32x4_t vw = vld1q_f32(conv_w + d * CONV_K);
            float sum = vaddvq_f32(vmulq_f32(vs, vw));
            conv_out[d] = sum / (1.0f + expf(-sum));
        }

        // Q: conv_out[0:QK_DIM] → 16 heads, repeated 3× to 48
        for (int h = 0; h < QK_HEADS; h++) {
            const float *src = conv_out + h * DN_D;
            for (int r = 0; r < QK_REPEAT; r++)
                memcpy(q_out + s * SSM_INNER + (h * QK_REPEAT + r) * DN_D,
                       src, DN_D * sizeof(float));
        }

        // K: conv_out[QK_DIM:2*QK_DIM] → 16 heads, repeated 3× to 48
        for (int h = 0; h < QK_HEADS; h++) {
            const float *src = conv_out + QK_DIM + h * DN_D;
            for (int r = 0; r < QK_REPEAT; r++)
                memcpy(k_out + s * SSM_INNER + (h * QK_REPEAT + r) * DN_D,
                       src, DN_D * sizeof(float));
        }

        // V: conv_out[2*QK_DIM:CONV_DIM] → SSM_INNER
        memcpy(v_out + s * SSM_INNER, conv_out + 2 * QK_DIM, SSM_INNER * sizeof(float));

        // Beta and g from raw_gate
        const float *rg = raw_gate + s * GATE_OUT;
        for (int h = 0; h < DN_H; h++) {
            float b_val = rg[h];
            beta_out[s * DN_H + h] = 1.0f / (1.0f + expf(-b_val));

            float a_val = rg[DN_H + h];
            float dt = A_log[h] + dt_bias[h];
            float alpha = 1.0f / (1.0f + expf(-a_val));
            g_out[s * DN_H + h] = expf(-expf(dt) * alpha);
        }
    }
}

// DeltaNet recurrence — head-parallel + NEON vectorized
static void blas_deltanet_recurrence_v2(
    float *q, float *k, float *v,
    const float *beta, const float *g,
    float *state, float *o, float *tmp,
    int sp, int nh, int nd)
{
    dispatch_apply((size_t)nh, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0),
        ^(size_t h) {
        float *sh = state + h * nd * nd;
        for (int s = 0; s < sp; s++) {
            float b = beta[s * nh + (int)h];
            float gv = g[s * nh + (int)h];
            float *qh = q + s * nh * nd + (int)h * nd;
            float *kh = k + s * nh * nd + (int)h * nd;
            float *vh = v + s * nh * nd + (int)h * nd;

            // L2 normalize Q and K (NEON)
            float32x4_t vqn = vdupq_n_f32(0), vkn = vdupq_n_f32(0);
            for (int d = 0; d < nd; d += 4) {
                float32x4_t vq = vld1q_f32(qh + d);
                float32x4_t vk = vld1q_f32(kh + d);
                vqn = vfmaq_f32(vqn, vq, vq);
                vkn = vfmaq_f32(vkn, vk, vk);
            }
            float q_inv = 1.0f / (sqrtf(vaddvq_f32(vqn)) + 1e-8f) / sqrtf((float)nd);
            float k_inv = 1.0f / (sqrtf(vaddvq_f32(vkn)) + 1e-8f);
            float32x4_t vqi = vdupq_n_f32(q_inv), vki = vdupq_n_f32(k_inv);
            for (int d = 0; d < nd; d += 4) {
                vst1q_f32(qh + d, vmulq_f32(vld1q_f32(qh + d), vqi));
                vst1q_f32(kh + d, vmulq_f32(vld1q_f32(kh + d), vki));
            }

            float32x4_t vg = vdupq_n_f32(gv);
            for (int i = 0; i < nd; i++) {
                float *si = sh + i * nd;
                float32x4_t vdot = vdupq_n_f32(0);
                for (int j = 0; j < nd; j += 4)
                    vdot = vfmaq_f32(vdot, vld1q_f32(si + j), vld1q_f32(kh + j));
                float coeff = b * (vh[i] - vaddvq_f32(vdot));
                float32x4_t vc = vdupq_n_f32(coeff);
                float32x4_t voh = vdupq_n_f32(0);
                for (int j = 0; j < nd; j += 4) {
                    float32x4_t vnew = vfmaq_f32(vmulq_f32(vg, vld1q_f32(si + j)),
                                                  vc, vld1q_f32(kh + j));
                    vst1q_f32(si + j, vnew);
                    voh = vfmaq_f32(voh, vnew, vld1q_f32(qh + j));
                }
                o[s * nh * nd + (int)h * nd + i] = vaddvq_f32(voh);
            }
        }
    });
}

// Post-gates: output × SiLU(Z) + group norm
static void cpu_deltanet_post_gates(
    const float *o_buf, const float *z_buf, const float *norm_w,
    float *gated, int sp)
{
    for (int s = 0; s < sp; s++) {
        const float *ob = o_buf + s * SSM_INNER;
        const float *zb = z_buf + s * SSM_INNER;
        float *gb = gated + s * SSM_INNER;

        for (int h = 0; h < DN_H; h++) {
            // Group norm on the head's output
            float ss = 0;
            for (int d = 0; d < DN_D; d++) {
                float v = ob[h * DN_D + d];
                ss += v * v;
            }
            float inv = 1.0f / sqrtf(ss / DN_D + 1e-5f);

            for (int d = 0; d < DN_D; d++) {
                float normed = ob[h * DN_D + d] * inv * norm_w[d];
                float z = zb[h * DN_D + d];
                float silu_z = z / (1.0f + expf(-z));
                gb[h * DN_D + d] = normed * silu_z;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Attention CPU helpers (same as 9B, parameterized for 27B)
// ═══════════════════════════════════════════════════════════════════

// Deinterleave Q projection: attn_q [sp, ATTN_Q_PROJ] → Q[sp, Q_DIM] + gate[sp, Q_DIM]
// ATTN_Q_PROJ = 2 * Q_DIM, interleaved as [Q_head0, gate_head0, Q_head1, gate_head1, ...]
static void cpu_attn_deinterleave(const float *raw, float *q, float *gate, int sp) {
    int hd2 = ATTN_HD * 2;
    for (int s = 0; s < sp; s++) {
        const float *src = raw + s * ATTN_Q_PROJ;
        float *dq = q + s * Q_DIM;
        float *dg = gate + s * Q_DIM;
        for (int h = 0; h < ATTN_HQ; h++) {
            memcpy(dq + h * ATTN_HD, src + h * hd2, ATTN_HD * sizeof(float));
            memcpy(dg + h * ATTN_HD, src + h * hd2 + ATTN_HD, ATTN_HD * sizeof(float));
        }
    }
}

// Per-head QK RMSNorm with learned weights
static void cpu_qk_rmsnorm(float *q, float *k, const float *q_norm_w,
                             const float *k_norm_w, int sp) {
    // Q: [sp, ATTN_HQ * ATTN_HD]
    for (int s = 0; s < sp; s++) {
        for (int h = 0; h < ATTN_HQ; h++) {
            float *qh = q + s * Q_DIM + h * ATTN_HD;
            float ss = 0;
            for (int d = 0; d < ATTN_HD; d++) ss += qh[d] * qh[d];
            float inv = 1.0f / sqrtf(ss / ATTN_HD + 1e-6f);
            for (int d = 0; d < ATTN_HD; d++) qh[d] = qh[d] * inv * q_norm_w[d];
        }
    }
    // K: [sp, ATTN_HKV * ATTN_HD]
    for (int s = 0; s < sp; s++) {
        for (int h = 0; h < ATTN_HKV; h++) {
            float *kh = k + s * KV_DIM + h * ATTN_HD;
            float ss = 0;
            for (int d = 0; d < ATTN_HD; d++) ss += kh[d] * kh[d];
            float inv = 1.0f / sqrtf(ss / ATTN_HD + 1e-6f);
            for (int d = 0; d < ATTN_HD; d++) kh[d] = kh[d] * inv * k_norm_w[d];
        }
    }
}

// RoPE (Qwen-style partial rotation)
static void cpu_rope(float *q, float *k, int sp, int pos, float base) {
    for (int s = 0; s < sp; s++) {
        int t = pos + s;
        // Q: ATTN_HQ heads
        for (int h = 0; h < ATTN_HQ; h++) {
            float *qh = q + s * Q_DIM + h * ATTN_HD;
            for (int p = 0; p < ROPE_PAIRS; p++) {
                float freq = 1.0f / powf(base, (float)(2 * p) / ROPE_DIM);
                float angle = t * freq;
                float cs = cosf(angle), sn = sinf(angle);
                float r0 = qh[p * 2], r1 = qh[p * 2 + 1];
                qh[p * 2]     = r0 * cs - r1 * sn;
                qh[p * 2 + 1] = r0 * sn + r1 * cs;
            }
        }
        // K: ATTN_HKV heads
        for (int h = 0; h < ATTN_HKV; h++) {
            float *kh = k + s * KV_DIM + h * ATTN_HD;
            for (int p = 0; p < ROPE_PAIRS; p++) {
                float freq = 1.0f / powf(base, (float)(2 * p) / ROPE_DIM);
                float angle = t * freq;
                float cs = cosf(angle), sn = sinf(angle);
                float r0 = kh[p * 2], r1 = kh[p * 2 + 1];
                kh[p * 2]     = r0 * cs - r1 * sn;
                kh[p * 2 + 1] = r0 * sn + r1 * cs;
            }
        }
    }
}

// Causal attention (CPU, GQA)
static float *g_head_scores = NULL;
static int g_head_scores_cap = 0;

static inline void attn_head_neon(
    const float *qvec, const float *k_cache, const float *v_cache,
    float *ovec, float *sc, int kvh, int seq_len, float scale)
{
    for (int p = 0; p < seq_len; p++) {
        const float *kvec = k_cache + p * KV_DIM + kvh * ATTN_HD;
        float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
        float32x4_t acc2 = vdupq_n_f32(0), acc3 = vdupq_n_f32(0);
        for (int d = 0; d < ATTN_HD; d += 16) {
            acc0 = vfmaq_f32(acc0, vld1q_f32(qvec+d),   vld1q_f32(kvec+d));
            acc1 = vfmaq_f32(acc1, vld1q_f32(qvec+d+4), vld1q_f32(kvec+d+4));
            acc2 = vfmaq_f32(acc2, vld1q_f32(qvec+d+8), vld1q_f32(kvec+d+8));
            acc3 = vfmaq_f32(acc3, vld1q_f32(qvec+d+12),vld1q_f32(kvec+d+12));
        }
        acc0 = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
        sc[p] = vaddvq_f32(acc0) * scale;
    }
    float mx = sc[0];
    for (int p = 1; p < seq_len; p++) if (sc[p] > mx) mx = sc[p];
    float sum = 0;
    for (int p = 0; p < seq_len; p++) { sc[p] = expf(sc[p] - mx); sum += sc[p]; }
    float inv = 1.0f / sum;
    for (int p = 0; p < seq_len; p++) sc[p] *= inv;
    memset(ovec, 0, ATTN_HD * sizeof(float));
    for (int p = 0; p < seq_len; p++) {
        const float *vvec = v_cache + p * KV_DIM + kvh * ATTN_HD;
        float32x4_t w = vdupq_n_f32(sc[p]);
        for (int d = 0; d < ATTN_HD; d += 4)
            vst1q_f32(ovec+d, vfmaq_f32(vld1q_f32(ovec+d), w, vld1q_f32(vvec+d)));
    }
}

static void cpu_causal_attention(
    const float *q, const float *k_cache, const float *v_cache,
    float *out, float *scores_buf, int sp, int kv_len, int pos)
{
    float scale = 1.0f / sqrtf((float)ATTN_HD);
    int gqa_ratio = ATTN_HQ / ATTN_HKV;
    int max_seq = pos + sp;

    int need = ATTN_HQ * (max_seq + 1);
    if (need > g_head_scores_cap) {
        free(g_head_scores);
        g_head_scores = (float *)malloc((size_t)need * sizeof(float));
        g_head_scores_cap = need;
    }

    int par_thresh = 512;
    for (int s = 0; s < sp; s++) {
        int seq_len = pos + s + 1;

        if (seq_len >= par_thresh) {
            dispatch_apply(ATTN_HQ,
                dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t qh) {
                int kvh = (int)qh / gqa_ratio;
                float *sc = g_head_scores + (int)qh * (max_seq + 1);
                attn_head_neon(q + s*Q_DIM + (int)qh*ATTN_HD,
                    k_cache, v_cache,
                    out + s*Q_DIM + (int)qh*ATTN_HD,
                    sc, kvh, seq_len, scale);
            });
        } else {
            for (int qh = 0; qh < ATTN_HQ; qh++) {
                int kvh = qh / gqa_ratio;
                attn_head_neon(q + s*Q_DIM + qh*ATTN_HD,
                    k_cache, v_cache,
                    out + s*Q_DIM + qh*ATTN_HD,
                    scores_buf, kvh, seq_len, scale);
            }
        }
    }
}

static void cpu_attn_output_gate(float *out, const float *gate, int sp) {
    for (int i = 0; i < sp * Q_DIM; i++) {
        float g = 1.0f / (1.0f + expf(-gate[i]));
        out[i] *= g;
    }
}

// ── ANE FFN (dynamic weights, all FP16) ──
// Conv1x1 FFN kernels — 4× faster than matmul MIL path
static ANEKernelHandle *ffn_k_conv = NULL;        // gate/up A: [ic=DIM, oc=INTER, sp=CHUNK]
static ANEKernelHandle *ffn_k_conv_B = NULL;       // gate/up B (double buffer)
static ANEKernelHandle *ffn_k_down_tile = NULL;    // down K-tile A: [ic=DOWN_K_TILE, oc=DIM, sp=CHUNK]
static ANEKernelHandle *ffn_k_down_B = NULL;       // down K-tile B (double buffer)
static ANEKernelHandle *ffn_k_down_rem = NULL;     // down remainder: [ic=DOWN_K_REM, oc=DIM, sp=CHUNK]
static _Float16 *ffn_gate_out = NULL;
static _Float16 *ffn_up_out   = NULL;
static _Float16 *ffn_down_accum = NULL;

#define DOWN_K_TILE 2048
#define DOWN_K_REM  (INTER - (INTER / DOWN_K_TILE) * DOWN_K_TILE)
#define DOWN_N_FULL (INTER / DOWN_K_TILE)


// Blocked GCD-parallel transposed pack: dq_buf[rows, cols] → IOSurface [1, cols, 1, sp+rows]
// Uses 32×32 blocks for L1-friendly cache access (13× faster than naive stride)
static void ffn_conv_stage(ANEKernelHandle *k, const _Float16 *acts_f16,
                            int sp, const _Float16 *dq_buf,
                            int rows, int cols) {
    int spw = sp + rows;
    IOSurfaceRef inSurf = (IOSurfaceRef)ane_bridge_get_input_surface(k, 0);
    IOSurfaceLock(inSurf, 0, NULL);
    _Float16 *buf = (_Float16 *)IOSurfaceGetBaseAddress(inSurf);

    // Activations: [cols, sp] layout — sequential memcpy per channel
    for (int d = 0; d < cols; d++)
        memcpy(buf + d * spw, acts_f16 + d * sp, sp * 2);

    // Weights: blocked transpose directly into IOSurface
    int nr = (rows + 31) / 32;
    int nc = (cols + 31) / 32;
    dispatch_apply(nr * nc, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0),
                   ^(size_t b) {
        int r0 = ((int)b / nc) * 32;
        int c0 = ((int)b % nc) * 32;
        int rend = r0 + 32 < rows ? r0 + 32 : rows;
        int cend = c0 + 32 < cols ? c0 + 32 : cols;
        for (int r = r0; r < rend; r++)
            for (int c = c0; c < cend; c++)
                buf[c * spw + sp + r] = dq_buf[r * cols + c];
    });

    IOSurfaceUnlock(inSurf, 0, NULL);
}

// Stage K-tile of down projection with blocked transposed packing
// dq_buf[oc, full_ic] → pack K-tile slice transposed into IOSurface
static void ffn_down_stage_tile(ANEKernelHandle *k, const _Float16 *acts_f16,
                                 int sp, const _Float16 *dq_buf,
                                 int oc, int full_ic, int k_start, int k_len) {
    int spw = sp + oc;
    IOSurfaceRef inSurf = (IOSurfaceRef)ane_bridge_get_input_surface(k, 0);
    IOSurfaceLock(inSurf, 0, NULL);
    _Float16 *buf = (_Float16 *)IOSurfaceGetBaseAddress(inSurf);

    // Activations: acts_f16[full_ic, sp], take slice [k_start:k_start+k_len, :]
    for (int d = 0; d < k_len; d++)
        memcpy(buf + d * spw, acts_f16 + (k_start + d) * sp, sp * 2);

    // Weight tile: blocked transpose dq_buf[dim, k_start:k_start+k_len] → buf
    int nr = (k_len + 31) / 32;
    int nc = (oc + 31) / 32;
    dispatch_apply(nr * nc, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0),
                   ^(size_t b) {
        int r0 = ((int)b / nc) * 32;  // r = k offset within tile
        int c0 = ((int)b % nc) * 32;  // c = output dim
        int rend = r0 + 32 < k_len ? r0 + 32 : k_len;
        int cend = c0 + 32 < oc ? c0 + 32 : oc;
        for (int c = c0; c < cend; c++)
            for (int r = r0; r < rend; r++)
                buf[r * spw + sp + c] = dq_buf[c * full_ic + k_start + r];
    });

    IOSurfaceUnlock(inSurf, 0, NULL);
}

static void ane_ffn_swiglu(GGUFFile *g, int layer, const float *x,
                            float *out, _Float16 *dq_buf, int sp,
                            const _Float16 *normed_f16_in) {
    GGUFTensor *t_gate = gguf_find(g, tn_ffn_gate(layer));
    GGUFTensor *t_up   = gguf_find(g, tn_ffn_up(layer));
    GGUFTensor *t_down = gguf_find(g, tn_ffn_down(layer));
    if (!t_gate || !t_up || !t_down) {
        fprintf(stderr, "FFN tensors missing for layer %d\n", layer);
        memset(out, 0, (size_t)DIM * sp * sizeof(float));
        return;
    }

    // Gate projection via conv1x1: [DIM → INTER]
    dequant_tensor_fp16(g, t_gate, dq_buf, INTER, DIM);
    ffn_conv_stage(ffn_k_conv, normed_f16_in, sp, dq_buf, INTER, DIM);
    ane_bridge_eval(ffn_k_conv);
    ane_bridge_read_output(ffn_k_conv, 0, ffn_gate_out, (size_t)INTER * sp * 2);

    // Up projection via conv1x1: [DIM → INTER]
    dequant_tensor_fp16(g, t_up, dq_buf, INTER, DIM);
    ffn_conv_stage(ffn_k_conv, normed_f16_in, sp, dq_buf, INTER, DIM);
    ane_bridge_eval(ffn_k_conv);
    ane_bridge_read_output(ffn_k_conv, 0, ffn_up_out, (size_t)INTER * sp * 2);

    // SiLU(gate) * up → ffn_gate_out in-place (all FP16)
    int64_t n = (int64_t)INTER * sp;
    for (int64_t i = 0; i < n; i++) {
        float gv = (float)ffn_gate_out[i];
        ffn_gate_out[i] = (_Float16)((gv / (1.0f + expf(-gv))) * (float)ffn_up_out[i]);
    }

    // Down projection via K-tiled conv1x1: [INTER → DIM]
    dequant_tensor_fp16(g, t_down, dq_buf, DIM, INTER);

    size_t down_out_sz = (size_t)DIM * sp;
    memset(ffn_down_accum, 0, down_out_sz * 2);
    _Float16 *tmp_out = (_Float16 *)malloc(down_out_sz * 2);

    for (int t = 0; t < DOWN_N_FULL; t++) {
        int k_start = t * DOWN_K_TILE;
        ffn_down_stage_tile(ffn_k_down_tile, ffn_gate_out, sp,
                            dq_buf, DIM, INTER, k_start, DOWN_K_TILE);
        ane_bridge_eval(ffn_k_down_tile);
        ane_bridge_read_output(ffn_k_down_tile, 0, tmp_out, down_out_sz * 2);
        for (size_t i = 0; i < down_out_sz; i++)
            ffn_down_accum[i] += tmp_out[i];
    }

    if (DOWN_K_REM > 0 && ffn_k_down_rem) {
        int k_start = DOWN_N_FULL * DOWN_K_TILE;
        ffn_down_stage_tile(ffn_k_down_rem, ffn_gate_out, sp,
                            dq_buf, DIM, INTER, k_start, DOWN_K_REM);
        ane_bridge_eval(ffn_k_down_rem);
        ane_bridge_read_output(ffn_k_down_rem, 0, tmp_out, down_out_sz * 2);
        for (size_t i = 0; i < down_out_sz; i++)
            ffn_down_accum[i] += tmp_out[i];
    }
    free(tmp_out);

    // Transpose [DIM, sp] → [sp, DIM] and convert FP16 → FP32
    dispatch_apply((size_t)DIM, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0),
        ^(size_t d) {
        for (int s = 0; s < sp; s++)
            out[s * DIM + (int)d] = (float)ffn_down_accum[(int)d * sp + s];
    });
}

// Fast ANE FFN with pre-transposed weights: all staging via memcpy (no in-place transpose)
static double ffn_t_gate_stage, ffn_t_gate_eval, ffn_t_gate_read;
static double ffn_t_up_stage, ffn_t_up_eval, ffn_t_up_read;
static double ffn_t_silu, ffn_t_down_stage, ffn_t_down_eval, ffn_t_down_read;
static double ffn_t_transpose;

static inline void ffn_down_tile_stage_parallel(ANEKernelHandle *k,
    const _Float16 *acts, int sp, const _Float16 *wdata,
    int oc, int k_start, int k_len) {
    int spw = sp + oc;
    IOSurfaceRef inSurf = (IOSurfaceRef)ane_bridge_get_input_surface(k, 0);
    IOSurfaceLock(inSurf, 0, NULL);
    _Float16 *buf = (_Float16 *)IOSurfaceGetBaseAddress(inSurf);
    int nblk = (k_len + 255) / 256;
    dispatch_apply((size_t)nblk,
        dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t b) {
        int d0 = (int)b * 256;
        int d1 = d0 + 256 < k_len ? d0 + 256 : k_len;
        for (int d = d0; d < d1; d++) {
            memcpy(buf + d * spw, acts + (k_start + d) * sp, sp * 2);
            memcpy(buf + d * spw + sp, wdata + (k_start + d) * oc, oc * 2);
        }
    });
    IOSurfaceUnlock(inSurf, 0, NULL);
}

static inline void accum_from_iosurface(_Float16 *accum, ANEKernelHandle *k, size_t n) {
    IOSurfaceRef outSurf = (IOSurfaceRef)ane_bridge_get_output_surface(k, 0);
    IOSurfaceLock(outSurf, kIOSurfaceLockReadOnly, NULL);
    const _Float16 *src = (const _Float16 *)IOSurfaceGetBaseAddress(outSurf);
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        float16x8_t a = vld1q_f16((const __fp16 *)(accum + i));
        float16x8_t s = vld1q_f16((const __fp16 *)(src + i));
        vst1q_f16((__fp16 *)(accum + i), vaddq_f16(a, s));
    }
    for (; i < n; i++)
        accum[i] += src[i];
    IOSurfaceUnlock(outSurf, kIOSurfaceLockReadOnly, NULL);
}

static void ane_ffn_swiglu_fast(float *out, int sp, const _Float16 *normed_f16,
                                 const PreTransW *pw_gate, const PreTransW *pw_up,
                                 const PreTransW *pw_down) {
    struct timespec t0, t1;

    // Gate: stage A, then overlap up staging into B during gate eval
    clock_gettime(CLOCK_MONOTONIC, &t0);
    fast_stage(ffn_k_conv, pw_gate, normed_f16, sp);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    ffn_t_gate_stage += (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;

    // Kick off up staging into B while gate eval runs on ANE
    clock_gettime(CLOCK_MONOTONIC, &t0);
    pipe_async_fast(ffn_k_conv_B, pw_up, normed_f16, sp);
    ane_bridge_eval(ffn_k_conv);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    ffn_t_gate_eval += (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;

    clock_gettime(CLOCK_MONOTONIC, &t0);
    ane_bridge_read_output(ffn_k_conv, 0, ffn_gate_out, (size_t)INTER * sp * 2);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    ffn_t_gate_read += (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;

    // Up: B already staged (wait if needed), just eval
    clock_gettime(CLOCK_MONOTONIC, &t0);
    pipe_wait();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    ffn_t_up_stage += (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;

    clock_gettime(CLOCK_MONOTONIC, &t0);
    ane_bridge_eval(ffn_k_conv_B);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    ffn_t_up_eval += (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;

    clock_gettime(CLOCK_MONOTONIC, &t0);
    ane_bridge_read_output(ffn_k_conv_B, 0, ffn_up_out, (size_t)INTER * sp * 2);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    ffn_t_up_read += (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;

    // SiLU(gate) * up → ffn_gate_out in-place
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int64_t n = (int64_t)INTER * sp;
    for (int64_t i = 0; i < n; i++) {
        float gv = (float)ffn_gate_out[i];
        ffn_gate_out[i] = (_Float16)((gv / (1.0f + expf(-gv))) * (float)ffn_up_out[i]);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    ffn_t_silu += (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;

    // Down: K-tiled with double-buffered pipelined staging
    size_t down_out_sz = (size_t)DIM * sp;
    memset(ffn_down_accum, 0, down_out_sz * 2);

    double ds = 0, de = 0, dr = 0;

    // Stage first tile synchronously
    if (DOWN_N_FULL > 0) {
        clock_gettime(CLOCK_MONOTONIC, &t0);
        ffn_down_tile_stage_parallel(ffn_k_down_tile, ffn_gate_out, sp,
                                      pw_down->data, DIM, 0, DOWN_K_TILE);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        ds += (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    }

    for (int t = 0; t < DOWN_N_FULL; t++) {
        ANEKernelHandle *cur = (t % 2 == 0) ? ffn_k_down_tile : ffn_k_down_B;
        ANEKernelHandle *nxt = (t % 2 == 0) ? ffn_k_down_B : ffn_k_down_tile;

        // Overlap: stage next tile into alt kernel during current eval
        if (t + 1 < DOWN_N_FULL) {
            pipe_async_down(nxt, ffn_gate_out, sp,
                            pw_down->data, DIM, (t+1) * DOWN_K_TILE, DOWN_K_TILE);
        }

        clock_gettime(CLOCK_MONOTONIC, &t0);
        ane_bridge_eval(cur);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        de += (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;

        clock_gettime(CLOCK_MONOTONIC, &t0);
        accum_from_iosurface(ffn_down_accum, cur, down_out_sz);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        dr += (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;

        if (t + 1 < DOWN_N_FULL) {
            clock_gettime(CLOCK_MONOTONIC, &t0);
            pipe_wait();
            clock_gettime(CLOCK_MONOTONIC, &t1);
            ds += (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
        }
    }

    if (DOWN_K_REM > 0 && ffn_k_down_rem) {
        int k_start = DOWN_N_FULL * DOWN_K_TILE;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        ffn_down_tile_stage_parallel(ffn_k_down_rem, ffn_gate_out, sp,
                                      pw_down->data, DIM, k_start, DOWN_K_REM);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        ds += (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;

        clock_gettime(CLOCK_MONOTONIC, &t0);
        ane_bridge_eval(ffn_k_down_rem);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        de += (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;

        clock_gettime(CLOCK_MONOTONIC, &t0);
        accum_from_iosurface(ffn_down_accum, ffn_k_down_rem, down_out_sz);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        dr += (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    }
    ffn_t_down_stage += ds;
    ffn_t_down_eval += de;
    ffn_t_down_read += dr;

    // Transpose [DIM, sp] FP16 → [sp, DIM] F32 with contiguous writes
    clock_gettime(CLOCK_MONOTONIC, &t0);
    dispatch_apply((size_t)sp,
        dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t s) {
        float *dst = out + s * DIM;
        int d = 0;
        for (; d + 3 < DIM; d += 4) {
            float16x4_t v = {ffn_down_accum[d * sp + (int)s], ffn_down_accum[(d+1) * sp + (int)s],
                             ffn_down_accum[(d+2) * sp + (int)s], ffn_down_accum[(d+3) * sp + (int)s]};
            vst1q_f32(dst + d, vcvt_f32_f16(v));
        }
        for (; d < DIM; d++)
            dst[d] = (float)ffn_down_accum[d * sp + (int)s];
    });
    clock_gettime(CLOCK_MONOTONIC, &t1);
    ffn_t_transpose += (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
}

// CPU FFN (SwiGLU with BLAS): dequant once per weight, sgemm for matmul
static void cpu_ffn_swiglu(GGUFFile *g, int layer, const float *x,
                            float *out, _Float16 *dq_buf, int sp) {
    GGUFTensor *t_gate = gguf_find(g, tn_ffn_gate(layer));
    GGUFTensor *t_up   = gguf_find(g, tn_ffn_up(layer));
    GGUFTensor *t_down = gguf_find(g, tn_ffn_down(layer));
    if (!t_gate || !t_up || !t_down) {
        fprintf(stderr, "FFN tensors missing for layer %d\n", layer);
        memset(out, 0, (size_t)DIM * sp * sizeof(float));
        return;
    }

    // Reusable FP32 weight buffer — largest is [INTER, DIM] = 89M floats = 356 MB
    static float *w_fp32 = NULL;
    if (!w_fp32) w_fp32 = (float *)malloc((size_t)INTER * DIM * sizeof(float));

    float *gate_out = (float *)malloc((size_t)INTER * sp * sizeof(float));
    float *up_out   = (float *)malloc((size_t)INTER * sp * sizeof(float));

    // gate_proj: x[sp, DIM] × W_gate^T[DIM, INTER] → gate_out[sp, INTER]
    // Dequant W_gate[INTER, DIM] → FP16 → FP32
    dequant_tensor_fp16(g, t_gate, dq_buf, INTER, DIM);
    for (int64_t i = 0; i < (int64_t)INTER * DIM; i++)
        w_fp32[i] = (float)dq_buf[i];
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                sp, INTER, DIM, 1.0f,
                x, DIM, w_fp32, DIM, 0.0f,
                gate_out, INTER);

    // up_proj: x[sp, DIM] × W_up^T[DIM, INTER] → up_out[sp, INTER]
    dequant_tensor_fp16(g, t_up, dq_buf, INTER, DIM);
    for (int64_t i = 0; i < (int64_t)INTER * DIM; i++)
        w_fp32[i] = (float)dq_buf[i];
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                sp, INTER, DIM, 1.0f,
                x, DIM, w_fp32, DIM, 0.0f,
                up_out, INTER);

    // SiLU(gate) * up → gate_out (in-place)
    for (int64_t i = 0; i < (int64_t)INTER * sp; i++) {
        float gv = gate_out[i];
        gate_out[i] = (gv / (1.0f + expf(-gv))) * up_out[i];
    }

    // down_proj: gate_out[sp, INTER] × W_down^T[INTER, DIM] → out[sp, DIM]
    dequant_tensor_fp16(g, t_down, dq_buf, DIM, INTER);
    for (int64_t i = 0; i < (int64_t)DIM * INTER; i++)
        w_fp32[i] = (float)dq_buf[i];
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                sp, DIM, INTER, 1.0f,
                gate_out, INTER, w_fp32, INTER, 0.0f,
                out, DIM);

    free(gate_out);
    free(up_out);
}

// ═══════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════

int main(int argc, char **argv) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&tb);

        if (argc < 2) {
            printf("Usage: %s <path-to-gguf> [seq_len]\n", argv[0]);
            return 1;
        }
        const char *gguf_path = argv[1];
        int S = (argc > 2) ? atoi(argv[2]) : 256;
        int CHUNK = 256;
        if (S < CHUNK) CHUNK = S;
        int n_chunks = S / CHUNK;

        // Detect chip
        char chip[64] = "Apple Silicon";
        size_t chip_sz = sizeof(chip);
        sysctlbyname("machdep.cpu.brand_string", chip, &chip_sz, NULL, 0);
        int ncpu = 0;
        size_t ncpu_sz = sizeof(ncpu);
        sysctlbyname("hw.ncpu", &ncpu, &ncpu_sz, NULL, 0);
        uint64_t memsize = 0;
        size_t mem_sz = sizeof(memsize);
        sysctlbyname("hw.memsize", &memsize, &mem_sz, NULL, 0);

        printf("═══════════════════════════════════════════════════════════════\n");
        printf("  ANE Prefill Benchmark — Qwen 3.6 27B\n");
        printf("  Chip: %s (%d cores, %llu GB)\n", chip, ncpu, memsize >> 30);
        printf("  S=%d, CHUNK=%d, %d layers (%d DN + %d Attn)\n",
               S, CHUNK, N_LAYERS, N_DN, N_ATTN);
        printf("  DIM=%d, SSM_INNER=%d, INTER=%d, CONV_DIM=%d\n",
               DIM, SSM_INNER, INTER, CONV_DIM);
        printf("═══════════════════════════════════════════════════════════════\n\n");

        // ─── Load GGUF ───
        printf("Loading GGUF: %s\n", gguf_path);
        uint64_t t0 = mach_absolute_time();
        GGUFFile gguf = {0};
        if (gguf_load(gguf_path, &gguf) != 0) {
            printf("Failed to load GGUF\n");
            return 1;
        }
        printf("  Loaded in %.0f ms (%.1f GB mmap'd)\n\n",
               ticks_to_ms(mach_absolute_time() - t0),
               gguf.mmap_size / (1024.0 * 1024 * 1024));

        // ─── Compile ANE kernels ───
        ANEKernelHandle *k_proj_qkv = NULL, *k_proj_gate = NULL, *k_proj_ssm_out = NULL;
        ANEKernelHandle *k_proj_attn_q = NULL, *k_proj_kv = NULL, *k_proj_kv_B = NULL;
        ANEKernelHandle *k_proj_attn_o = NULL;

        printf("Compiling ANE kernels...\n");
        ane_bridge_init();
        t0 = mach_absolute_time();

        int compile_count = 0;
        k_proj_qkv     = compile_dyn_proj(DIM, CONV_DIM, CHUNK);
        k_proj_gate    = compile_dyn_proj(DIM, SSM_INNER, CHUNK);
        k_proj_ssm_out = compile_dyn_proj(SSM_INNER, DIM, CHUNK);
        k_proj_attn_q  = compile_dyn_proj(DIM, ATTN_Q_PROJ, CHUNK);
        k_proj_kv      = compile_dyn_proj(DIM, KV_DIM, CHUNK);
        k_proj_kv_B    = compile_dyn_proj(DIM, KV_DIM, CHUNK);
        k_proj_attn_o  = compile_dyn_proj(Q_DIM, DIM, CHUNK);
        compile_count = 7;

        ffn_k_conv = compile_dyn_conv(DIM, INTER, CHUNK);
        ffn_k_conv_B = compile_dyn_conv(DIM, INTER, CHUNK);
        compile_count += 2;
        ffn_k_down_tile = compile_dyn_conv(DOWN_K_TILE, DIM, CHUNK);
        ffn_k_down_B = compile_dyn_conv(DOWN_K_TILE, DIM, CHUNK);
        compile_count += 2;
        if (DOWN_K_REM > 0) {
            ffn_k_down_rem = compile_dyn_conv(DOWN_K_REM, DIM, CHUNK);
            compile_count++;
        }

        ffn_gate_out   = (_Float16 *)malloc((size_t)INTER * CHUNK * 2);
        ffn_up_out     = (_Float16 *)malloc((size_t)INTER * CHUNK * 2);
        ffn_down_accum = (_Float16 *)calloc((size_t)DIM * CHUNK, 2);

        printf("  %d kernels compiled in %.0f ms (6 proj + %d FFN conv)\n",
               compile_count, ticks_to_ms(mach_absolute_time() - t0),
               compile_count - 6);

        // Start staging pipeline thread
        pthread_t pipe_tid;
        pthread_create(&pipe_tid, NULL, pipe_thread_fn, &g_pipe);

        // ─── Working buffers ───
        // Max dequant buffer: largest weight tensor = FFN gate/up [INTER, DIM]
        size_t max_dq = (size_t)INTER * DIM * 2;
        _Float16 *dq_tmp = (_Float16 *)malloc(max_dq);

        float *hidden     = calloc((size_t)DIM * S, sizeof(float));
        float *normed     = calloc((size_t)DIM * CHUNK, sizeof(float));
        float *proj_out   = calloc((size_t)DIM * CHUNK, sizeof(float));
        float *ffn_out    = calloc((size_t)DIM * CHUNK, sizeof(float));
        _Float16 *normed_f16 = (_Float16 *)malloc((size_t)DIM * CHUNK * 2);
        _Float16 *tmp_f16    = (_Float16 *)malloc((size_t)SSM_INNER * CHUNK * 2);

        // DeltaNet buffers
        float *qkv_buf    = calloc((size_t)CONV_DIM * CHUNK, sizeof(float));
        float *gz_buf     = calloc((size_t)SSM_INNER * CHUNK, sizeof(float));
        float *o_buf      = calloc((size_t)SSM_INNER * CHUNK, sizeof(float));
        float *dn_q_buf   = calloc((size_t)SSM_INNER * CHUNK, sizeof(float));
        float *dn_k_buf   = calloc((size_t)SSM_INNER * CHUNK, sizeof(float));
        float *dn_v_buf   = calloc((size_t)SSM_INNER * CHUNK, sizeof(float));
        float *dn_z_buf   = calloc((size_t)SSM_INNER * CHUNK, sizeof(float));
        float *dn_beta    = calloc((size_t)DN_H * CHUNK, sizeof(float));
        float *dn_g       = calloc((size_t)DN_H * CHUNK, sizeof(float));
        float *raw_gate   = calloc((size_t)GATE_OUT * CHUNK, sizeof(float));
        float *dn_gated   = calloc((size_t)SSM_INNER * CHUNK, sizeof(float));
        float *dn_tmp     = calloc((size_t)SSM_INNER, sizeof(float));

        // DeltaNet per-layer conv state [N_DN][CONV_DIM * CONV_K]
        float **conv_states = (float **)calloc(N_DN, sizeof(float *));
        for (int i = 0; i < N_DN; i++)
            conv_states[i] = calloc((size_t)CONV_DIM * CONV_K, sizeof(float));

        // DeltaNet recurrence state [N_DN][DN_H * DN_D * DN_D]
        float **dn_states = (float **)calloc(N_DN, sizeof(float *));
        for (int i = 0; i < N_DN; i++)
            dn_states[i] = calloc((size_t)DN_H * DN_D * DN_D, sizeof(float));

        // Attention buffers
        float *attn_q_raw = calloc((size_t)ATTN_Q_PROJ * CHUNK, sizeof(float));
        float *attn_q     = calloc((size_t)Q_DIM * CHUNK, sizeof(float));
        float *attn_gate  = calloc((size_t)Q_DIM * CHUNK, sizeof(float));
        float *attn_k     = calloc((size_t)KV_DIM * CHUNK, sizeof(float));
        float *attn_v     = calloc((size_t)KV_DIM * CHUNK, sizeof(float));
        float *attn_out   = calloc((size_t)Q_DIM * CHUNK, sizeof(float));
        float *attn_scores = calloc(S, sizeof(float));

        // KV cache per attention layer
        float **kv_caches = (float **)calloc(N_ATTN, sizeof(float *));
        for (int i = 0; i < N_ATTN; i++)
            kv_caches[i] = calloc((size_t)2 * S * KV_DIM, sizeof(float));

        // F32 norm weights (small, load all upfront)
        float *norm_pre[N_LAYERS];
        float *ffn_norm_w[N_LAYERS];
        float *q_norm_w[N_ATTN], *k_norm_w[N_ATTN];
        float *dn_gate_w[N_DN], *dn_conv_w[N_DN];
        float *dn_A_log[N_DN], *dn_dt_bias[N_DN], *dn_norm_o[N_DN];

        printf("\nLoading F32 parameters...\n");
        t0 = mach_absolute_time();
        {
            int di = 0, ai = 0;
            for (int l = 0; l < N_LAYERS; l++) {
                norm_pre[l] = (float *)malloc(DIM * sizeof(float));
                extract_f32(&gguf, tn_norm(l), norm_pre[l], DIM);
                ffn_norm_w[l] = (float *)malloc(DIM * sizeof(float));
                extract_f32(&gguf, tn_ffn_norm(l), ffn_norm_w[l], DIM);

                if (is_attn_layer(l)) {
                    q_norm_w[ai] = (float *)malloc(ATTN_HD * sizeof(float));
                    k_norm_w[ai] = (float *)malloc(ATTN_HD * sizeof(float));
                    extract_f32(&gguf, tn_attn_qn(l), q_norm_w[ai], ATTN_HD);
                    extract_f32(&gguf, tn_attn_kn(l), k_norm_w[ai], ATTN_HD);
                    ai++;
                } else {
                    // Gate weights: alpha[DIM, DN_H] + beta[DIM, DN_H] → gate_w[GATE_OUT * DIM]
                    dn_gate_w[di] = (float *)calloc(GATE_OUT * DIM, sizeof(float));
                    // In 27B, alpha and beta are F32 [DIM, DN_H] = [5120, 48]
                    // GGUF stores as [ne[0]=5120, ne[1]=48], so the matrix is [48, 5120]
                    // But for gate proj: normed[1,DIM] × gate_w^T[DIM, GATE_OUT] → [1, GATE_OUT]
                    // gate_w layout: first DN_H rows = beta, next DN_H rows = alpha
                    // Each row = DIM elements
                    GGUFTensor *t_beta = gguf_find(&gguf, tn_dn_beta(l));
                    GGUFTensor *t_alpha = gguf_find(&gguf, tn_dn_alpha(l));
                    if (t_beta) {
                        const uint8_t *raw = gguf_data(&gguf, t_beta);
                        // GGUF [5120, 48] → ne[0]=5120, ne[1]=48
                        // This is a [48, 5120] matrix (48 rows, 5120 cols)
                        // gate_w[j * DIM + k] for j=0..DN_H-1 (beta)
                        if (t_beta->type == GGML_TYPE_F32) {
                            // Stored as [48][5120] row-major
                            memcpy(dn_gate_w[di], raw, (size_t)DN_H * DIM * sizeof(float));
                        } else {
                            extract_f32(&gguf, tn_dn_beta(l), dn_gate_w[di], DN_H * DIM);
                        }
                    }
                    if (t_alpha) {
                        const uint8_t *raw = gguf_data(&gguf, t_alpha);
                        if (t_alpha->type == GGML_TYPE_F32) {
                            memcpy(dn_gate_w[di] + DN_H * DIM, raw,
                                   (size_t)DN_H * DIM * sizeof(float));
                        } else {
                            extract_f32(&gguf, tn_dn_alpha(l),
                                        dn_gate_w[di] + DN_H * DIM, DN_H * DIM);
                        }
                    }

                    dn_conv_w[di] = (float *)malloc((size_t)CONV_DIM * CONV_K * sizeof(float));
                    extract_f32(&gguf, tn_dn_conv(l), dn_conv_w[di], CONV_DIM * CONV_K);

                    dn_A_log[di] = (float *)malloc(DN_H * sizeof(float));
                    extract_f32(&gguf, tn_dn_a(l), dn_A_log[di], DN_H);

                    dn_dt_bias[di] = (float *)malloc(DN_H * sizeof(float));
                    extract_f32(&gguf, tn_dn_dt(l), dn_dt_bias[di], DN_H);

                    dn_norm_o[di] = (float *)malloc(DN_NORM_DIM * sizeof(float));
                    extract_f32(&gguf, tn_dn_norm_o(l), dn_norm_o[di], DN_NORM_DIM);

                    di++;
                }
            }
        }
        printf("  F32 params loaded in %.0f ms\n", ticks_to_ms(mach_absolute_time() - t0));

        // ─── Initialize hidden state from embedding ───
        printf("\nInitializing embeddings...\n");
        {
            GGUFTensor *t_emb = gguf_find(&gguf, "token_embd.weight");
            if (!t_emb) { printf("FATAL: token_embd.weight not found\n"); return 1; }

            int test_tokens[] = {
                248045, 8678, 198, 2523, 513, 264, 10631, 17313, 13,
                22516, 5707, 303, 6163, 13, 248046, 198, 248045, 846,
                198, 3710, 369, 279, 6511, 314, 9338, 30, 248046, 198,
                248045, 74455, 198
            };
            int n_test = sizeof(test_tokens) / sizeof(test_tokens[0]);
            int n_tokens = (S < n_test) ? S : n_test;

            // Dequant embedding rows on demand
            const uint8_t *emb_raw = gguf_data(&gguf, t_emb);
            _Float16 row_fp16[DIM];
            for (int i = 0; i < n_tokens; i++) {
                int tok = test_tokens[i % n_test];
                if (tok >= 0 && tok < VOCAB_SIZE) {
                    switch (t_emb->type) {
                        case GGML_TYPE_Q4_K:
                            dequant_row_q4k_fp16(emb_raw, row_fp16, tok, DIM);
                            for (int d = 0; d < DIM; d++)
                                hidden[i * DIM + d] = (float)row_fp16[d];
                            break;
                        case GGML_TYPE_F32:
                            memcpy(hidden + i * DIM,
                                   (const float *)emb_raw + tok * DIM,
                                   DIM * sizeof(float));
                            break;
                        default:
                            fprintf(stderr, "Unsupported embed type %d\n", t_emb->type);
                    }
                }
            }
            for (int i = n_tokens; i < S; i++)
                memcpy(hidden + i * DIM, hidden + (n_tokens - 1) * DIM, DIM * sizeof(float));

            printf("  %d tokens embedded\n", n_tokens);
        }

        // ═══════════════════════════════════════════════════════════════
        // Pipeline: per-layer, per-chunk processing
        // Pre-dequant projections per-layer (memory-constrained)
        // ═══════════════════════════════════════════════════════════════

        printf("\nRunning %d-layer pipeline (S=%d, CHUNK=%d)...\n", N_LAYERS, S, CHUNK);

        int is_attn[N_LAYERS];
        for (int l = 0; l < N_LAYERS; l++) is_attn[l] = is_attn_layer(l);

        double total_ane_proj_ms = 0, total_ane_ffn_ms = 0;
        double total_cpu_norm_ms = 0, total_cpu_rec_ms = 0, total_cpu_attn_ms = 0;
        double total_cpu_pregate_ms = 0, total_cpu_postgate_ms = 0;
        double total_dequant_ms = 0, total_cpu_ffn_ms = 0;
        double total_staging_ms = 0, total_transpose_ms = 0;
        double total_proj_stage_ms = 0, total_proj_eval_ms = 0, total_proj_read_ms = 0;
        double total_ffn_stage_ms = 0, total_ffn_eval_ms = 0, total_ffn_read_ms = 0;
        int proj_dispatch_count = 0;

        // Pre-allocate FP16 transpose buffers (avoid malloc/free per layer)
        _Float16 *gated_f16_buf = (_Float16 *)malloc((size_t)SSM_INNER * CHUNK * 2);
        _Float16 *attn_out_f16_buf = (_Float16 *)malloc((size_t)Q_DIM * CHUNK * 2);
        _Float16 *ffn_normed_f16_buf = (_Float16 *)malloc((size_t)DIM * CHUNK * 2);

        // Double-buffered weight prefetch: dequant layer L+1 while layer L runs on ANE.
        typedef struct {
            PreTransW proj[4];
            PreTransW ffn[3];
        } LayerWeights;

        static void (^predequant_layer)(LayerWeights *, GGUFFile *, int, int, _Float16 *) =
            ^(LayerWeights *lw, GGUFFile *g, int layer, int attn, _Float16 *tmp) {
            char name[256];
            if (!attn) {
                snprintf(name, 256, "blk.%d.attn_qkv.weight", layer);
                predequant_gguf(&lw->proj[0], g, name, CONV_DIM, DIM, tmp);
                snprintf(name, 256, "blk.%d.attn_gate.weight", layer);
                predequant_gguf(&lw->proj[1], g, name, SSM_INNER, DIM, tmp);
                snprintf(name, 256, "blk.%d.ssm_out.weight", layer);
                predequant_gguf(&lw->proj[2], g, name, DIM, SSM_INNER, tmp);
            } else {
                snprintf(name, 256, "blk.%d.attn_q.weight", layer);
                predequant_gguf(&lw->proj[0], g, name, ATTN_Q_PROJ, DIM, tmp);
                snprintf(name, 256, "blk.%d.attn_k.weight", layer);
                predequant_gguf(&lw->proj[1], g, name, KV_DIM, DIM, tmp);
                snprintf(name, 256, "blk.%d.attn_v.weight", layer);
                predequant_gguf(&lw->proj[2], g, name, KV_DIM, DIM, tmp);
                snprintf(name, 256, "blk.%d.attn_output.weight", layer);
                predequant_gguf(&lw->proj[3], g, name, DIM, Q_DIM, tmp);
            }
            snprintf(name, 256, "blk.%d.ffn_gate.weight", layer);
            predequant_gguf(&lw->ffn[0], g, name, INTER, DIM, tmp);
            snprintf(name, 256, "blk.%d.ffn_up.weight", layer);
            predequant_gguf(&lw->ffn[1], g, name, INTER, DIM, tmp);
            snprintf(name, 256, "blk.%d.ffn_down.weight", layer);
            predequant_gguf(&lw->ffn[2], g, name, DIM, INTER, tmp);
        };

        static void (^free_layer_weights)(LayerWeights *) = ^(LayerWeights *lw) {
            for (int i = 0; i < 4; i++) predequant_free(&lw->proj[i]);
            for (int i = 0; i < 3; i++) predequant_free(&lw->ffn[i]);
        };

        #define PREFAULT_AHEAD 8
        LayerWeights *lw_buf = (LayerWeights *)calloc(2, sizeof(LayerWeights));
        _Float16 *dq_tmp2 = (_Float16 *)malloc(max_dq);

        uint64_t t_pipeline = mach_absolute_time();

        for (int p = 0; p < PREFAULT_AHEAD && p < N_LAYERS; p++)
            gguf_prefault_layer(&gguf, p, is_attn[p]);

        uint64_t t_dq0 = mach_absolute_time();
        predequant_layer(&lw_buf[0], &gguf, 0, is_attn[0], dq_tmp);
        double initial_dq_ms = ticks_to_ms(mach_absolute_time() - t_dq0);
        printf("  Layer 0 weights pre-dequanted in %.0f ms\n", initial_dq_ms);

        int cur_buf = 0;

        for (int l = 0; l < N_LAYERS; l++) {
            uint64_t t_layer = mach_absolute_time();

            int local_dn = 0, local_attn = 0;
            for (int i = 0; i < l; i++) {
                if (is_attn[i]) local_attn++;
                else local_dn++;
            }

            // Pre-fault GGUF pages PREFAULT_AHEAD layers ahead (async in kernel)
            if (l + PREFAULT_AHEAD < N_LAYERS)
                gguf_prefault_layer(&gguf, l + PREFAULT_AHEAD, is_attn[l + PREFAULT_AHEAD]);

            int next_buf = 1 - cur_buf;
            dispatch_group_t prefetch_group = dispatch_group_create();
            if (l + 1 < N_LAYERS) {
                int next_l = l + 1;
                int next_attn = is_attn[next_l];
                dispatch_group_async(prefetch_group,
                    dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
                    predequant_layer(&lw_buf[next_buf], &gguf, next_l, next_attn, dq_tmp2);
                });
            }

            LayerWeights *lw = &lw_buf[cur_buf];
            PreTransW *pw_proj1 = &lw->proj[0], *pw_proj2 = &lw->proj[1],
                      *pw_proj3 = &lw->proj[2], *pw_proj4 = &lw->proj[3];

            total_dequant_ms += initial_dq_ms;
            initial_dq_ms = 0;

            for (int c = 0; c < n_chunks; c++) {
                int pos = c * CHUNK;
                float *h_chunk = hidden + pos * DIM;
                uint64_t tl;

                // ── RMSNorm ──
                tl = mach_absolute_time();
                cpu_rmsnorm(normed, h_chunk, norm_pre[l], DIM, CHUNK);
                total_cpu_norm_ms += ticks_to_ms(mach_absolute_time() - tl);

                tl = mach_absolute_time();
                transpose_to_f16(normed_f16, normed, CHUNK, DIM);
                total_transpose_ms += ticks_to_ms(mach_absolute_time() - tl);

                #define PROJ_DISPATCH(kern, pw, acts, out_buf, oc_dim, sp_dim) do { \
                    uint64_t _ts = mach_absolute_time(); \
                    fast_stage(kern, pw, acts, sp_dim); \
                    uint64_t _te = mach_absolute_time(); \
                    total_proj_stage_ms += ticks_to_ms(_te - _ts); \
                    ane_bridge_eval(kern); \
                    uint64_t _tr = mach_absolute_time(); \
                    total_proj_eval_ms += ticks_to_ms(_tr - _te); \
                    ane_read_output(kern, out_buf, oc_dim, sp_dim); \
                    total_proj_read_ms += ticks_to_ms(mach_absolute_time() - _tr); \
                    proj_dispatch_count++; \
                } while(0)

                if (!is_attn[l]) {
                    // ═══════════════════════════════════════════
                    // DeltaNet Layer — pipelined QKV+Gate
                    // ═══════════════════════════════════════════
                    tl = mach_absolute_time();
                    {
                        uint64_t _ts = mach_absolute_time();
                        fast_stage(k_proj_qkv, pw_proj1, normed_f16, CHUNK);
                        total_proj_stage_ms += ticks_to_ms(mach_absolute_time() - _ts);

                        // Overlap: stage gate into different kernel during qkv eval
                        pipe_async_fast(k_proj_gate, pw_proj2, normed_f16, CHUNK);
                        _ts = mach_absolute_time();
                        ane_bridge_eval(k_proj_qkv);
                        total_proj_eval_ms += ticks_to_ms(mach_absolute_time() - _ts);

                        _ts = mach_absolute_time();
                        ane_read_output(k_proj_qkv, qkv_buf, CONV_DIM, CHUNK);
                        total_proj_read_ms += ticks_to_ms(mach_absolute_time() - _ts);
                        proj_dispatch_count++;

                        _ts = mach_absolute_time();
                        pipe_wait();
                        total_proj_stage_ms += ticks_to_ms(mach_absolute_time() - _ts);

                        _ts = mach_absolute_time();
                        ane_bridge_eval(k_proj_gate);
                        total_proj_eval_ms += ticks_to_ms(mach_absolute_time() - _ts);

                        _ts = mach_absolute_time();
                        ane_read_output(k_proj_gate, gz_buf, SSM_INNER, CHUNK);
                        total_proj_read_ms += ticks_to_ms(mach_absolute_time() - _ts);
                        proj_dispatch_count++;
                    }
                    total_ane_proj_ms += ticks_to_ms(mach_absolute_time() - tl);

                    tl = mach_absolute_time();
                    cpu_gate_proj(normed, dn_gate_w[local_dn], raw_gate, CHUNK);
                    cpu_deltanet_pre_gates(qkv_buf, raw_gate, dn_conv_w[local_dn],
                                           conv_states[local_dn],
                                           dn_A_log[local_dn], dn_dt_bias[local_dn],
                                           dn_q_buf, dn_k_buf, dn_v_buf,
                                           dn_beta, dn_g, CHUNK);
                    total_cpu_pregate_ms += ticks_to_ms(mach_absolute_time() - tl);

                    memcpy(dn_z_buf, gz_buf, (size_t)SSM_INNER * CHUNK * sizeof(float));

                    tl = mach_absolute_time();
                    blas_deltanet_recurrence_v2(dn_q_buf, dn_k_buf, dn_v_buf,
                                                dn_beta, dn_g,
                                                dn_states[local_dn], o_buf, dn_tmp,
                                                CHUNK, DN_H, DN_D);
                    total_cpu_rec_ms += ticks_to_ms(mach_absolute_time() - tl);

                    tl = mach_absolute_time();
                    cpu_deltanet_post_gates(o_buf, dn_z_buf, dn_norm_o[local_dn],
                                            dn_gated, CHUNK);
                    total_cpu_postgate_ms += ticks_to_ms(mach_absolute_time() - tl);

                    tl = mach_absolute_time();
                    transpose_to_f16(gated_f16_buf, dn_gated, CHUNK, SSM_INNER);
                    PROJ_DISPATCH(k_proj_ssm_out, pw_proj3, gated_f16_buf, proj_out, DIM, CHUNK);
                    total_ane_proj_ms += ticks_to_ms(mach_absolute_time() - tl);

                } else {
                    // ═══════════════════════════════════════════
                    // Attention Layer — pipelined Q→K→V
                    // ═══════════════════════════════════════════
                    tl = mach_absolute_time();
                    {
                        uint64_t _ts = mach_absolute_time();
                        fast_stage(k_proj_attn_q, pw_proj1, normed_f16, CHUNK);
                        total_proj_stage_ms += ticks_to_ms(mach_absolute_time() - _ts);

                        // Overlap: stage K during Q eval
                        pipe_async_fast(k_proj_kv, pw_proj2, normed_f16, CHUNK);
                        _ts = mach_absolute_time();
                        ane_bridge_eval(k_proj_attn_q);
                        total_proj_eval_ms += ticks_to_ms(mach_absolute_time() - _ts);

                        _ts = mach_absolute_time();
                        ane_read_output(k_proj_attn_q, attn_q_raw, ATTN_Q_PROJ, CHUNK);
                        total_proj_read_ms += ticks_to_ms(mach_absolute_time() - _ts);
                        proj_dispatch_count++;

                        _ts = mach_absolute_time();
                        pipe_wait();
                        total_proj_stage_ms += ticks_to_ms(mach_absolute_time() - _ts);

                        // Overlap: stage V into kv_B during K eval
                        pipe_async_fast(k_proj_kv_B, pw_proj3, normed_f16, CHUNK);
                        _ts = mach_absolute_time();
                        ane_bridge_eval(k_proj_kv);
                        total_proj_eval_ms += ticks_to_ms(mach_absolute_time() - _ts);

                        _ts = mach_absolute_time();
                        ane_read_output(k_proj_kv, attn_k, KV_DIM, CHUNK);
                        total_proj_read_ms += ticks_to_ms(mach_absolute_time() - _ts);
                        proj_dispatch_count++;

                        _ts = mach_absolute_time();
                        pipe_wait();
                        total_proj_stage_ms += ticks_to_ms(mach_absolute_time() - _ts);

                        _ts = mach_absolute_time();
                        ane_bridge_eval(k_proj_kv_B);
                        total_proj_eval_ms += ticks_to_ms(mach_absolute_time() - _ts);

                        _ts = mach_absolute_time();
                        ane_read_output(k_proj_kv_B, attn_v, KV_DIM, CHUNK);
                        total_proj_read_ms += ticks_to_ms(mach_absolute_time() - _ts);
                        proj_dispatch_count++;
                    }
                    total_ane_proj_ms += ticks_to_ms(mach_absolute_time() - tl);

                    tl = mach_absolute_time();
                    cpu_attn_deinterleave(attn_q_raw, attn_q, attn_gate, CHUNK);
                    cpu_qk_rmsnorm(attn_q, attn_k, q_norm_w[local_attn],
                                    k_norm_w[local_attn], CHUNK);
                    cpu_rope(attn_q, attn_k, CHUNK, pos, 1e7);

                    float *kv = kv_caches[local_attn];
                    float *k_cache = kv;
                    float *v_cache = kv + S * KV_DIM;
                    for (int s = 0; s < CHUNK; s++) {
                        int t = pos + s;
                        memcpy(k_cache + t * KV_DIM, attn_k + s * KV_DIM, KV_DIM * sizeof(float));
                        memcpy(v_cache + t * KV_DIM, attn_v + s * KV_DIM, KV_DIM * sizeof(float));
                    }
                    total_cpu_attn_ms += ticks_to_ms(mach_absolute_time() - tl);

                    int kv_len = pos + CHUNK;
                    tl = mach_absolute_time();
                    cpu_causal_attention(attn_q, k_cache, v_cache, attn_out,
                                         attn_scores, CHUNK, kv_len, pos);
                    cpu_attn_output_gate(attn_out, attn_gate, CHUNK);
                    total_cpu_attn_ms += ticks_to_ms(mach_absolute_time() - tl);

                    tl = mach_absolute_time();
                    transpose_to_f16(attn_out_f16_buf, attn_out, CHUNK, Q_DIM);
                    PROJ_DISPATCH(k_proj_attn_o, pw_proj4, attn_out_f16_buf, proj_out, DIM, CHUNK);
                    total_ane_proj_ms += ticks_to_ms(mach_absolute_time() - tl);
                }

                cpu_residual_add(h_chunk, proj_out, DIM, CHUNK);

                // ── FFN ──
                tl = mach_absolute_time();
                cpu_rmsnorm(normed, h_chunk, ffn_norm_w[l], DIM, CHUNK);
                total_cpu_norm_ms += ticks_to_ms(mach_absolute_time() - tl);

                tl = mach_absolute_time();
                transpose_to_f16(ffn_normed_f16_buf, normed, CHUNK, DIM);
                ane_ffn_swiglu_fast(ffn_out, CHUNK, ffn_normed_f16_buf,
                                    &lw->ffn[0], &lw->ffn[1], &lw->ffn[2]);
                total_ane_ffn_ms += ticks_to_ms(mach_absolute_time() - tl);

                cpu_residual_add(h_chunk, ffn_out, DIM, CHUNK);

                if (c == 0) {
                    float hmx = 0;
                    int hni = 0, hnn = 0;
                    for (int i = 0; i < DIM * CHUNK; i++) {
                        if (isinf(h_chunk[i])) hni++;
                        if (isnan(h_chunk[i])) hnn++;
                        float av = fabsf(h_chunk[i]);
                        if (av > hmx && isfinite(h_chunk[i])) hmx = av;
                    }
                    if (hnn > 0 || hni > 0)
                        printf("  L%d hidden: max=%.1f, ninf=%d, nnan=%d\n", l, hmx, hni, hnn);
                    else if (l < 4 || l == 63)
                        printf("  L%d hidden: max=%.1f\n", l, hmx);
                }
            }

            // Wait for next layer's prefetch to complete, then swap buffers
            dispatch_group_wait(prefetch_group, DISPATCH_TIME_FOREVER);
            free_layer_weights(&lw_buf[cur_buf]);
            cur_buf = next_buf;

            double layer_ms = ticks_to_ms(mach_absolute_time() - t_layer);
            if (l < 4 || l == 63)
                printf("  Layer %d: %.0f ms\n", l, layer_ms);
        }

        free(dq_tmp2);
        free(gated_f16_buf);
        free(attn_out_f16_buf);
        free(ffn_normed_f16_buf);

        double pipeline_ms = ticks_to_ms(mach_absolute_time() - t_pipeline);
        double tok_per_sec = (double)S / (pipeline_ms / 1000.0);

        printf("\n═══════════════════════════════════════════════════════════════\n");
        printf("  RESULTS\n");
        printf("═══════════════════════════════════════════════════════════════\n");
        printf("  Total: %.0f ms → %.1f tok/s (S=%d)\n", pipeline_ms, tok_per_sec, S);
        printf("  ANE proj:     %.0f ms  (%d dispatches)\n", total_ane_proj_ms, proj_dispatch_count);
        printf("    stage:      %.0f ms (%.1f ms/call)\n", total_proj_stage_ms, total_proj_stage_ms / proj_dispatch_count);
        printf("    eval:       %.0f ms (%.1f ms/call)\n", total_proj_eval_ms, total_proj_eval_ms / proj_dispatch_count);
        printf("    read:       %.0f ms (%.1f ms/call)\n", total_proj_read_ms, total_proj_read_ms / proj_dispatch_count);
        printf("  ANE FFN:      %.0f ms\n", total_ane_ffn_ms);
        printf("    gate:  stg=%.0f eval=%.0f rd=%.0f\n", ffn_t_gate_stage, ffn_t_gate_eval, ffn_t_gate_read);
        printf("    up:    stg=%.0f eval=%.0f rd=%.0f\n", ffn_t_up_stage, ffn_t_up_eval, ffn_t_up_read);
        printf("    silu:  %.0f ms\n", ffn_t_silu);
        printf("    down:  stg=%.0f eval=%.0f rd=%.0f\n", ffn_t_down_stage, ffn_t_down_eval, ffn_t_down_read);
        printf("    xpose: %.0f ms\n", ffn_t_transpose);
        printf("  CPU norm:     %.0f ms\n", total_cpu_norm_ms);
        printf("  CPU recur:    %.0f ms\n", total_cpu_rec_ms);
        printf("  CPU attn:     %.0f ms\n", total_cpu_attn_ms);
        printf("  CPU pregate:  %.0f ms\n", total_cpu_pregate_ms);
        printf("  CPU postgate: %.0f ms\n", total_cpu_postgate_ms);
        printf("  CPU FFN:      %.0f ms\n", total_cpu_ffn_ms);
        printf("  Dequant:      %.0f ms\n", total_dequant_ms);
        printf("  Transpose:    %.0f ms\n", total_transpose_ms);
        printf("  Staging:      %.0f ms\n", total_staging_ms);
        double accounted = total_ane_proj_ms + total_ane_ffn_ms + total_cpu_norm_ms +
            total_cpu_rec_ms + total_cpu_attn_ms + total_cpu_pregate_ms +
            total_cpu_postgate_ms + total_cpu_ffn_ms + total_transpose_ms;
        printf("  Other:        %.0f ms\n", pipeline_ms - accounted);

        // ── Final norm + LM head (sample output) ──
        {
            float *final_norm_w = (float *)malloc(DIM * sizeof(float));
            extract_f32(&gguf, "output_norm.weight", final_norm_w, DIM);

            float *final_hidden = (float *)malloc(DIM * sizeof(float));
            cpu_rmsnorm(final_hidden, hidden + (S - 1) * DIM, final_norm_w, DIM, 1);

            printf("\n  Final hidden[0:4] = [%.4f, %.4f, %.4f, %.4f]\n",
                   final_hidden[0], final_hidden[1], final_hidden[2], final_hidden[3]);

            // LM head: [VOCAB_SIZE, DIM] × hidden → logits
            // Too slow to do full vocab, just check top-5
            GGUFTensor *t_lm = gguf_find(&gguf, "output.weight");
            if (t_lm) {
                const uint8_t *lm_raw = gguf_data(&gguf, t_lm);
                float best_score = -1e30f;
                int best_tok = 0;
                _Float16 row_fp16[DIM];

                // Sample first 1000 + common tokens
                for (int tok = 0; tok < 1000 && tok < VOCAB_SIZE; tok++) {
                    switch (t_lm->type) {
                        case GGML_TYPE_Q6_K:
                            dequant_row_q6k_fp16(lm_raw, row_fp16, tok, DIM);
                            break;
                        case GGML_TYPE_Q4_K:
                            dequant_row_q4k_fp16(lm_raw, row_fp16, tok, DIM);
                            break;
                        default: break;
                    }
                    float dot = 0;
                    for (int d = 0; d < DIM; d++) dot += final_hidden[d] * (float)row_fp16[d];
                    if (dot > best_score) { best_score = dot; best_tok = tok; }
                }
                printf("  Top token (first 1000): %d (score=%.2f)\n", best_tok, best_score);
            }

            free(final_hidden);
            free(final_norm_w);
        }

        printf("\n═══════════════════════════════════════════════════════════════\n");

        // Cleanup
        if (gguf.mmap_base) munmap(gguf.mmap_base, gguf.mmap_size);
        if (gguf.fd >= 0) close(gguf.fd);
        if (gguf.tensors) free(gguf.tensors);
    }
    return 0;
}
