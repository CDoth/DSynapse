#include "DSynapse.h"
#include <dmem.h>
#include <cmath>


DLogs::DLogsContext DSynapse::log_context;

int DSynapse::DSynapseInitLogContext(const char *stream_name)
{
    DLOGS_INIT_DEFAULT_CONTEXT(stream_name);
    log_context.set_log_level(0);
    log_context.set_lvl_cmp_callback(DLogs::default_lvl_cmp__more_oe);
    return 0;
}
DLogs::DLogsContext *DSynapse::DSynapseGetLogContext()
{
    return &log_context;
}
void DSynapse::DSynapseSetLogContext(const DLogs::DLogsContext &c)
{
    log_context = c;
}

DSynapse::rand_callback DSynapse::global_rand_callback = xrands;
DSynapse::ranged_rand_callback DSynapse::global_ranged_rand_callback = xrand;

void DSynapse::DSynapseSetRandCallback(DSynapse::rand_callback c)
{
    global_rand_callback = c;
}
void DSynapse::DSynapseSetRangedRandCallback(DSynapse::ranged_rand_callback c)
{
    global_ranged_rand_callback = c;
}


DSynapse::data_line DSynapse::alloc_buffer(int size)
{
    if(size > 0)
    {
       data_line b =  get_zmem<nvt>(size);
       if(b) return b;
       else DL_BADALLOC(1, "buffer");
    }
    else DL_BADVALUE(1, "size");
    return NULL;
}
DSynapse::data_line DSynapse::realloc_buffer(DSynapse::data_line b, int size)
{
    if(b == NULL) {DL_BADPOINTER(1, "b"); return NULL;}
    if(size <= 0) {DL_BADVALUE(1, "size: [%d]", size); return NULL;}
    if( (b = reget_mem(b, size)) == NULL) {DL_BADALLOC(1, "b"); return NULL;}
    return b;
}
DSynapse::data_line DSynapse::copy_buffer(DSynapse::data_line buffer, int size)
{
    data_line copy = NULL;
    if(buffer)
    {
        if(size > 0)
        {
            if(  (copy = get_zmem<nvt>(size)) == NULL  )
            {
                DL_BADALLOC(1, "copy");
                goto fail;
            }
            copy_mem(copy, buffer, size);
        }
        else
        {
            DL_BADVALUE(1, "size");
            goto fail;
        }
    }
    else
    {
        goto fail;
    }
fail:
    free_mem(copy);
    return NULL;
}

DSynapse::ACTIVATION DSynapse::actt_get_lable_by_index(int index)
{
    switch (index)
    {
        case 0: return ACT_SIGMOID; break;
        case 1: return ACT_GTAN; break;
        case 2: return ACT_RELU; break;
        case 3: return ACT_LEAKYRELU; break;
        case 4: return ACT_GELU; break;
        case 5: return ACT_ELU; break;
        case 6: return ACT_LINE; break;
        default:
        case 7: return ACT_EMPTY; break;
        case 8: return ACT_DEBUG; break;
    }
}
DSynapse::ACTIVATION DSynapse::actt_get_lable_by_callback(DSynapse::ACTF a)
{
    if(a == act_sigmoid) return ACT_SIGMOID;
    if(a == act_gtan) return ACT_GTAN;
    if(a == act_ReLU) return ACT_RELU;
    if(a == act_LINE) return ACT_LINE;
    if(a == act_empty) return ACT_EMPTY;
    if(a == act_debug) return ACT_DEBUG;
    if(a == act_GELU) return ACT_GELU;
    if(a == act_LeakyReLU) return ACT_LEAKYRELU;
    if(a== act_ELU) return ACT_ELU;
    return ACT_EMPTY;
}
int DSynapse::actt_get_index_by_lable(DSynapse::ACTIVATION a)
{
    switch (a)
    {
        case ACT_SIGMOID: return 0; break;
        case ACT_GTAN: return 1; break;
        case ACT_RELU: return 2; break;
        case ACT_LEAKYRELU: return 3; break;
        case ACT_GELU: return 4; break;
        case ACT_ELU: return 5; break;
        case ACT_LINE: return 6; break;
        case ACT_EMPTY: return 7; break;
        case ACT_DEBUG: return 8; break;
        default: return -1; break;
    }
}
int DSynapse::actt_get_index_by_callback(DSynapse::ACTF a)
{
    if(a == act_sigmoid) return 0;
    if(a == act_gtan) return 1;
    if(a == act_ReLU) return 2;
    if(a == act_LeakyReLU) return 3;
    if(a == act_GELU) return 4;
    if(a == act_ELU) return 5;
    if(a == act_LINE) return 6;
    if(a == act_empty) return 7;
    if(a == act_debug) return 8;
    return -1;
}
void DSynapse::actt_get_callback_by_index(int index, ACTF &a, ACTF &da)
{
    switch (index)
    {
        case 0: a = act_sigmoid; da = der_act_sigmoid; break;
        case 1: a = act_gtan; da = der_act_gtan; break;
        case 2: a = act_ReLU; da = der_act_ReLU; break;
        case 3: a = act_LeakyReLU; da = der_act_LeakyReLU; break;
        case 4: a = act_GELU; da = der_act_GELU; break;
        case 5: a = act_ELU; da = der_act_ELU; break;
        case 6: a = act_LINE; da = der_act_LINE; break;
        default:
        case 7: a = act_empty; da = der_act_empty; break;
        case 8: a = act_debug; da = der_act_debug; break;
    }
}
void DSynapse::actt_get_callback_by_lable(DSynapse::ACTIVATION t, DSynapse::ACTF &a, DSynapse::ACTF &da)
{
    switch (t)
    {
        case ACT_SIGMOID: a = act_sigmoid; da = der_act_sigmoid; break;
        case ACT_GTAN: a = act_gtan; da = der_act_gtan; break;
        case ACT_RELU: a = act_ReLU; da = der_act_ReLU; break;
        case ACT_LEAKYRELU: a = act_LeakyReLU; da = der_act_LeakyReLU; break;
        case ACT_GELU: a = act_GELU; da = der_act_GELU; break;
        case ACT_ELU: a = act_ELU; da = der_act_ELU; break;
        case ACT_LINE: a = act_LINE; da = der_act_LINE; break;
        default:
        case ACT_EMPTY: a = act_empty; da = der_act_empty; break;
        case ACT_DEBUG: a = act_debug; da = der_act_debug; break;
    }
}

DSynapse::nvt DSynapse::act_sigmoid(DSynapse::nvt x)
{
    return 1.0/(1.0+exp(-x));
}
DSynapse::nvt DSynapse::der_act_sigmoid(DSynapse::nvt x)
{
    nvt e = exp(-x);
    return e/((e+1.0)*(e+1.0));
}
DSynapse::nvt DSynapse::act_gtan(DSynapse::nvt x)
{
    return (2.0/( 1.0 + exp(-x) )) - 1;
}
DSynapse::nvt DSynapse::der_act_gtan(DSynapse::nvt x)
{
    nvt e = exp(-x);
    return (2.0 * e)/((e+1.0)*(e+1.0));
}
DSynapse::nvt DSynapse::act_ReLU(DSynapse::nvt x)
{
    return (x>0.0) ? x : 0.0 ;
}
DSynapse::nvt DSynapse::der_act_ReLU(DSynapse::nvt x)
{
    return (x>0.0) ? 1.0 : 0.0 ;
}
DSynapse::nvt DSynapse::act_LeakyReLU(DSynapse::nvt x)
{
    return (x>0.0) ? x : (DSYNAPSE_LEAKY_RELU_RATE*x) ;
}
DSynapse::nvt DSynapse::der_act_LeakyReLU(DSynapse::nvt x)
{
    return (x>0.0) ? 1.0 : DSYNAPSE_LEAKY_RELU_RATE ;
}
DSynapse::nvt DSynapse::act_GELU(DSynapse::nvt x)
{
    return 0.5 * x * (
                    1 +
                    tanh(
                        (x + 0.044715 * (x * x * x)) *
                        sqrt(2.0/M_PI)
                    )
                );
}
DSynapse::nvt DSynapse::der_act_GELU(DSynapse::nvt x)
{
    return (
                0.5 * tanh((0.0356774 * x * x * x) + (0.797885 * x)) +
                ((0.0535161 * x * x * x) + 0.398942 * x) * pow( (1.0/cosh((0.0356774 * x * x * x) + 0.797885 * x)), 2 ) +
                0.5
                );
}
DSynapse::nvt DSynapse::act_ELU(DSynapse::nvt x)
{
    return (x>0.0) ? x : (DSYNAPSE_ELU_RATE * (exp(x) - 1.0)) ;
}
DSynapse::nvt DSynapse::der_act_ELU(DSynapse::nvt x)
{
    return (x>0.0) ? 1 : DSYNAPSE_ELU_RATE * exp(x);
}
DSynapse::nvt DSynapse::act_LINE(DSynapse::nvt x)
{
    return x;
}
DSynapse::nvt DSynapse::der_act_LINE(DSynapse::nvt)
{
    return 1.0;
}
DSynapse::nvt DSynapse::act_empty(DSynapse::nvt x)
{
    return x;
}
DSynapse::nvt DSynapse::der_act_empty(DSynapse::nvt x)
{
    return x;
}
DSynapse::nvt DSynapse::act_debug(DSynapse::nvt x)
{
    //debug code here
}
DSynapse::nvt DSynapse::der_act_debug(DSynapse::nvt x)
{
    //debug code here
}


//-------------------------------------------------------------------
DSynapse::WeightMatrix *DSynapse::wm_alloc()
{
    WeightMatrix *m = get_zmem<WeightMatrix>(1);
    return m;
}
int DSynapse::wm_free(DSynapse::WeightMatrix *wm)
{
    if(wm)
    {
        wm_destruct_matrix(wm);
        free_mem(wm);
    }
    return 0;
}
DSynapse::WeightMatrix *DSynapse::wm_copy(DSynapse::WeightMatrix *src)
{
    if(src == NULL) {DL_BADPOINTER(1, "src"); return NULL;}
    WeightMatrix *m = get_zmem<WeightMatrix>(1);
    if( wm_create_matrix(m, src->mainSize, src->biasSize) < 0 )
        goto fail;
    wm_copy_weights_value(m, src, WM_BOTH);
    return m;
fail:
    wm_free(m);
    return NULL;
}
int DSynapse::wm_create_matrix(DSynapse::WeightMatrix *m, int mainSize, int biasSize)
{
    if(m == NULL) {DL_BADPOINTER(1, "m"); return -1;}
    if( (m->main = alloc_buffer(mainSize)) == NULL ) {DL_BADALLOC(1, "main"); goto fail;}
    if( (m->bias = alloc_buffer(biasSize)) == NULL ) {DL_BADALLOC(1, "bias"); goto fail;}
    m->mainSize = mainSize;
    m->biasSize = biasSize;

    return 0;
fail:
    wm_destruct_matrix(m);
    return -1;
}
int DSynapse::wm_create_matrix2(DSynapse::WeightMatrix *m, int currentLayerSize, int nextLayerSize)
{
    return wm_create_matrix(m, currentLayerSize * nextLayerSize, nextLayerSize);
}
int DSynapse::wm_recreate_matrix(DSynapse::WeightMatrix *m, int mainSize, int biasSize)
{
    if(m == NULL) {DL_BADPOINTER(1, "m"); return -1;}
    if( (m->main = realloc_buffer(m->main, mainSize)) == NULL ) {DL_BADALLOC(1, "main"); goto fail;}
    if( (m->bias = realloc_buffer(m->bias, biasSize)) == NULL ) {DL_BADALLOC(1, "bias"); goto fail;}
    m->mainSize = mainSize;
    m->biasSize = biasSize;
    return 0;
fail:
    wm_destruct_matrix(m);
    return -1;
}
int DSynapse::wm_recreate_matrix2(DSynapse::WeightMatrix *m, int currentLayerSize, int nextLayerSize)
{
    return wm_recreate_matrix(m, currentLayerSize * nextLayerSize, nextLayerSize);
}
int DSynapse::wm_destruct_matrix(DSynapse::WeightMatrix *wm)
{
    if(wm == NULL) {DL_BADPOINTER(1, "wm"); return -1;}
    free_mem(wm->main);
    free_mem(wm->bias);
    zero_mem(wm, 1);
    return 0;
}
int DSynapse::wm_zero_matrix(DSynapse::WeightMatrix *m, DSynapse::WM_BUFFER_TYPE t)
{
#ifdef DSYNAPSE_USE_WM_SAVE_CALL
    if(m == NULL) {DL_BADPOINTER(1, "m"); return -1;}
#endif
    if(t == WM_MAIN || t == WM_BOTH) {
#ifdef DSYNAPSE_USE_WM_SAVE_CALL
        if(m->main == NULL) {DL_BADPOINTER(1, "main"); return -1;}
        if(m->mainSize <= 0) {DL_BADVALUE(1, "mainSize: [%d]", m->mainSize); return -1;}
#endif
        zero_mem(m->main, m->mainSize);
    }
    if(t == WM_BIAS || t == WM_BOTH){
#ifdef DSYNAPSE_USE_WM_SAVE_CALL
        if(m->bias == NULL) {DL_BADPOINTER(1, "bias"); return -1;}
        if(m->biasSize <= 0) {DL_BADVALUE(1, "biasSize: [%d]", m->biasSize); return -1;}
#endif
        zero_mem(m->bias, m->biasSize);
    }
    return 0;
}
int DSynapse::wm_copy_weights_value(DSynapse::WeightMatrix *dst, DSynapse::WeightMatrix *src, DSynapse::WM_BUFFER_TYPE t)
{
#ifdef DSYNAPSE_USE_WM_SAVE_CALL
    if(dst == NULL) {DL_BADPOINTER(1, "dst"); return -1;}
    if(src == NULL) {DL_BADPOINTER(1, "src"); return -1;}
#endif
    if(t == WM_MAIN || t == WM_BOTH) {
#ifdef DSYNAPSE_USE_WM_SAVE_CALL
        if(src->main == NULL) {DL_BADPOINTER(1, "src: main"); return -1;}
        if(src->mainSize <= 0) {DL_BADVALUE(1, "src: mainSize: [%d]", src->mainSize); return -1;}
        if(dst->main == NULL) { DL_BADPOINTER(1, "dst: main"); return -1;}
        if(dst->mainSize != src->mainSize) {DL_BADVALUE(1, "different main sizes: dst: [%d] src [%d]", dst->mainSize, src->mainSize); return -1;}
#endif
        copy_mem(dst->main, src->main, src->mainSize);
    }
    if(t == WM_BIAS || t == WM_BOTH){
#ifdef DSYNAPSE_USE_WM_SAVE_CALL
        if(src->bias == NULL) {DL_BADPOINTER(1, "src: bias"); return -1;}
        if(src->biasSize <= 0) {DL_BADVALUE(1, "src: mainSize: [%d]", src->mainSize); return -1;}
        if(dst->bias == NULL) { DL_BADPOINTER(1, "dst: bias"); return -1;}
        if(dst->biasSize != src->biasSize) {DL_BADVALUE(1, "different bias sizes: dst: [%d] src [%d]", dst->biasSize, src->biasSize); return -1;}
#endif
        copy_mem(dst->bias, src->bias, src->biasSize);
    }
    return 0;
}
int DSynapse::wm_set_rand(DSynapse::WeightMatrix *m, WM_BUFFER_TYPE t)
{
#ifdef DSYNAPSE_USE_WM_SAVE_CALL
    if(m == NULL) {DL_BADPOINTER(1, "m"); return -1;}
#endif
    if(t == WM_MAIN || t == WM_BOTH) {
#ifdef DSYNAPSE_USE_WM_SAVE_CALL
        if(m->main == NULL) {DL_BADPOINTER(1, "main"); return -1;}
        if(m->mainSize <= 0) {DL_BADVALUE(1, "mainSize: [%d]", m->mainSize); return -1;}
#endif
        auto b = m->main;
        auto e = m->main + m->mainSize;
        while(b!=e){
            *b++ = global_rand_callback();
        }
    }
    if(t == WM_BIAS || t == WM_BOTH){
#ifdef DSYNAPSE_USE_WM_SAVE_CALL
        if(m->bias == NULL) {DL_BADPOINTER(1, "bias"); return -1;}
        if(m->biasSize <= 0) {DL_BADVALUE(1, "biasSize: [%d]", m->biasSize); return -1;}
#endif
        auto b = m->bias;
        auto e = m->bias + m->biasSize;
        while(b!=e){
            *b++ = global_rand_callback();
        }
    }
    return 0;
}
int DSynapse::wm_set_rand(DSynapse::WeightMatrix *m, DSynapse::WM_BUFFER_TYPE t, DSynapse::nvt bot, DSynapse::nvt top)
{
#ifdef DSYNAPSE_USE_WM_SAVE_CALL
    if(m == NULL) {DL_BADPOINTER(1, "m"); return -1;}
#endif
    if(t == WM_MAIN || t == WM_BOTH) {
#ifdef DSYNAPSE_USE_WM_SAVE_CALL
        if(m->main == NULL) {DL_BADPOINTER(1, "main"); return -1;}
        if(m->mainSize <= 0) {DL_BADVALUE(1, "mainSize: [%d]", m->mainSize); return -1;}
#endif
        auto b = m->main;
        auto e = m->main + m->mainSize;
        while(b!=e){
            *b++ = global_ranged_rand_callback(bot, top);
        }
    }
    if(t == WM_BIAS || t == WM_BOTH){
#ifdef DSYNAPSE_USE_WM_SAVE_CALL
        if(m->bias == NULL) {DL_BADPOINTER(1, "bias"); return -1;}
        if(m->biasSize <= 0) {DL_BADVALUE(1, "biasSize: [%d]", m->biasSize); return -1;}
#endif
        auto b = m->bias;
        auto e = m->bias + m->biasSize;
        while(b!=e){
            *b++ = global_ranged_rand_callback(bot, top);
        }
    }
    return 0;
}
int DSynapse::wm_set_rand(DSynapse::WeightMatrix *m, WM_BUFFER_TYPE t, DSynapse::rand_callback rc)
{
#ifdef DSYNAPSE_USE_WM_SAVE_CALL
    if(m == NULL) {DL_BADPOINTER(1, "m"); return -1;}
    if(!rc) {DL_BADPOINTER(1, "rc"); return -1;}
#endif
    if(t == WM_MAIN || t == WM_BOTH) {
#ifdef DSYNAPSE_USE_WM_SAVE_CALL
        if(m->main == NULL) {DL_BADPOINTER(1, "main"); return -1;}
        if(m->mainSize <= 0) {DL_BADVALUE(1, "mainSize: [%d]", m->mainSize); return -1;}
#endif
        auto b = m->main;
        auto e = m->main + m->mainSize;
        while(b!=e){
            *b++ = rc();
        }
    }
    if(t == WM_BIAS || t == WM_BOTH){
#ifdef DSYNAPSE_USE_WM_SAVE_CALL
        if(m->bias == NULL) {DL_BADPOINTER(1, "bias"); return -1;}
        if(m->biasSize <= 0) {DL_BADVALUE(1, "biasSize: [%d]", m->biasSize); return -1;}
#endif
        auto b = m->bias;
        auto e = m->bias + m->biasSize;
        while(b!=e){
            *b++ = rc();
        }
    }
    return 0;
}
int DSynapse::wm_set_rand(DSynapse::WeightMatrix *m, WM_BUFFER_TYPE t, DSynapse::ranged_rand_callback rrc, nvt bot, nvt top)
{
#ifdef DSYNAPSE_USE_WM_SAVE_CALL
    if(m == NULL) {DL_BADPOINTER(1, "m"); return -1;}
    if(!rrc) {DL_BADPOINTER(1, "rrc"); return -1;}
#endif
    if(t == WM_MAIN || t == WM_BOTH) {
#ifdef DSYNAPSE_USE_WM_SAVE_CALL
        if(m->main == NULL) {DL_BADPOINTER(1, "main"); return -1;}
        if(m->mainSize <= 0) {DL_BADVALUE(1, "mainSize: [%d]", m->mainSize); return -1;}
#endif
        auto b = m->main;
        auto e = m->main + m->mainSize;
        while(b!=e){
            *b++ = rrc(bot, top);
        }
    }
    if(t == WM_BIAS || t == WM_BOTH){
#ifdef DSYNAPSE_USE_WM_SAVE_CALL
        if(m->bias == NULL) {DL_BADPOINTER(1, "bias"); return -1;}
        if(m->biasSize <= 0) {DL_BADVALUE(1, "biasSize: [%d]", m->biasSize); return -1;}
#endif
        auto b = m->bias;
        auto e = m->bias + m->biasSize;
        while(b!=e){
            *b++ = rrc(bot, top);
        }
    }
    return 0;
}
int DSynapse::wm_set_weight_value(DSynapse::WeightMatrix *m, int wpos, DSynapse::nvt value)
{
#ifdef DSYNAPSE_USE_WM_SAVE_CALL
    if(m == NULL) {DL_BADPOINTER(1, "m"); return -1;}
    if(m->main == NULL) {DL_BADPOINTER(1, "main"); return -1;}
#endif
    if(wpos < m->mainSize)
        m->main[wpos] = value;
    return 0;
}
int DSynapse::wm_set_bias_value(DSynapse::WeightMatrix *m, int bpos, DSynapse::nvt value)
{
#ifdef DSYNAPSE_USE_WM_SAVE_CALL
    if(m == NULL) {DL_BADPOINTER(1, "m"); return -1;}
    if(m->bias == NULL) {DL_BADPOINTER(1, "bias"); return -1;}
#endif
    if(bpos < m->biasSize)
        m->bias[bpos] = value;
    return 0;
}
int DSynapse::wm_print(DSynapse::WeightMatrix *m, DSynapse::WM_BUFFER_TYPE t)
{
    if(m)
    {
        if(t == WM_BOTH){
            int nextSize = m->biasSize;
            int currSize = m->mainSize / nextSize;
            FOR_VALUE(nextSize, i)
            {
                FOR_VALUE(currSize, j)
                        std::cout << m->main[i*currSize + j] << " ";
                std::cout << "(bias: " << m->bias[i] << ")" << std::endl;
            }
        }
        else
        {
            if(t == WM_MAIN){
                int nextSize = m->biasSize;
                int currSize = m->mainSize / nextSize;
                FOR_VALUE(nextSize, i)
                {
                    FOR_VALUE(currSize, j)
                            std::cout << m->main[i*currSize + j] << " ";
                }
            }
            if(t == WM_BIAS){
                std::cout << "bias: " << std::endl;
                FOR_VALUE(m->biasSize, i){
                    std::cout << m->bias[i] << " " << std::endl;
                }
            }
        }
    }
    return 0;
}
//-------------------------------------------------------------------
DSynapse::DescentContext *DSynapse::alloc_descent_context()
{
    DescentContext *c = NULL;
    if(   (c = get_zmem<DescentContext>(1)) == NULL   )
    {
        DL_BADALLOC(1, "DescentContext");
        goto fail;
    }

    /*
    c->delta_w = NULL;
    c->delta_b = NULL;
    c->w_size = 0;
    c->b_size = 0;
    */
    c->deltaBuffer = NULL;
    c->special_buffer1 = NULL;
    c->special_buffer2 = NULL;


    c->use_main_buffer = 1;
    c->use_support_buffer = 0;

    c->mu = 0.7;
    c->eps = 1e-8;
    c->t = 0;
    c->beta1 = 0.900;
    c->beta2 = 0.979;
    c->decay_rate = 0.99;

    return c;
fail:
    free_descent_context(c);
    return NULL;
}
DSynapse::DescentContext *DSynapse::copy_descent_context(DSynapse::DescentContext *from)
{
    DescentContext *c = NULL;
    if(from == NULL)
    {
        DL_BADPOINTER(1, "from");
        goto fail;
    }
    if(  (c = alloc_descent_context()) == NULL  )
    {
        DL_BADALLOC(1, "New descent context");
        goto fail;
    }
    if(from->special_buffer1)
    {
        if( (c->special_buffer1 = wm_copy(from->special_buffer1) ) == NULL )
        {
            DL_ERROR(1, "Can't copy special_buffer1");
            goto fail;
        }
    }
    if(from->special_buffer2)
    {
        if( (c->special_buffer2 = wm_copy(from->special_buffer2)) == NULL )
        {
            DL_ERROR(1, "Can't copy special_buffer2");
            goto fail;
        }
    }
    if(from->deltaBuffer){
        if( (c->deltaBuffer = wm_copy(from->deltaBuffer)) == NULL ){
            DL_ERROR(1, "Can't copy deltaBuffer");
            goto fail;
        }
    }
    /*
    if(from->delta_w)
    {
        if( (c->delta_w = alloc_buffer(from->w_size)) == NULL )
        {
            DL_ERROR(1, "Can't copy delta_w");
            goto fail;
        }
    }
    if(from->delta_b)
    {
        if( (c->delta_b = alloc_buffer(from->b_size)) == NULL )
        {
            DL_ERROR(1, "Can't copy delta_b");
            goto fail;
        }
    }
    c->w_size = from->w_size;
    c->b_size = from->b_size;
    */

    c->use_main_buffer = from->use_main_buffer;
    c->use_support_buffer = from->use_support_buffer;
    c->mu = from->mu;
    c->eps = from->eps;
    c->t = from->t;
    c->beta1 = from->beta1;
    c->beta2 = from->beta2;
    c->decay_rate = from->decay_rate;

    return c;

fail:
    free_descent_context(c);
    return NULL;
}
int DSynapse::free_descent_context(DSynapse::DescentContext *c)
{
    if(c)
    {
        /*
        free_mem(c->special_buffer1);
        free_mem(c->special_buffer2);
        free_mem(c->delta_b);
        free_mem(c->delta_w);
        */

        wm_free(c->deltaBuffer);
        wm_free(c->special_buffer1);
        wm_free(c->special_buffer2);


        zero_mem(c, sizeof(DescentContext));
        free_mem(c);
        return 0;
    }
    return -1;
}
int DSynapse::descent_context_clear_buffers(DSynapse::DescentContext *c)
{
    if(c == NULL) {DL_BADPOINTER(1, "context"); return -1;}
    /*
    if(c->w_size > 0)
    {
        if(c->delta_w) zero_mem(c->delta_w, c->w_size);
        if(c->special_buffer1) zero_mem(c->special_buffer1, c->w_size);
        if(c->special_buffer2) zero_mem(c->special_buffer2, c->w_size);
    }
    if(c->delta_b && c->b_size > 0) zero_mem(c->delta_b, c->b_size);
    */

    if(c->deltaBuffer) wm_zero_matrix(c->deltaBuffer, WM_BOTH);
    if(c->special_buffer1) wm_zero_matrix(c->special_buffer1, WM_BOTH);
    if(c->special_buffer2) wm_zero_matrix(c->special_buffer2, WM_BOTH);

    return 0;
}

//-------------------------------------------------------------------
int DSynapse::forward_propagation(DSynapse::data_line out,
                                  DSynapse::const_data_line input, DSynapse::const_data_line w, DSynapse::const_data_line bias,
                                  int in_size, int out_size, DSynapse::ACTF act)
{
#ifdef DSYNAPSE_USE_FORWARD_PROP_SAVE_CALL
    if(out == NULL) { DL_BADPOINTER(1, "out"); return -1; }
    if(input == NULL) { DL_BADPOINTER(1, "input"); return -1; }
    if(w == NULL) { DL_BADPOINTER(1, "w"); return -1; }
    if(bias == NULL) { DL_BADPOINTER(1, "bias"); return -1; }

    if(in_size <= 0) { DL_BADVALUE(1, "in_size: %d", in_size); return -1; }
    if(out_size <= 0) { DL_BADVALUE(1, "out_size: %d", out_size); return -1; }
    if(!act) {DL_BADVALUE(1, "act"); return -1;}
#endif

    auto out_end = out + out_size;
    auto input_it = input;
    auto input_end = input + in_size;
    nvt v = 0;
    while(out != out_end)
    {
        v = *bias;
        input_it = input;
        while(input_it != input_end)
        {
            v += *input_it * *w;
            ++input_it;
            ++w;
        }
        *out = act(v);
        ++out;
        ++bias;
    }
    return 0;
}
int DSynapse::forward_propagation_learn(DSynapse::data_line out, DSynapse::data_line raw, DSynapse::data_line der_out,
                                        DSynapse::const_data_line input, DSynapse::const_data_line w, DSynapse::const_data_line bias,
                                        int in_size, int out_size, DSynapse::ACTF act, DSynapse::ACTF der_act)
{
#ifdef DSYNAPSE_USE_FORWARD_PROP_SAVE_CALL
    if(out == NULL) { DL_BADPOINTER(1, "out"); return -1; }
    if(raw == NULL) { DL_BADPOINTER(1, "raw"); return -1; }
    if(der_out == NULL) { DL_BADPOINTER(1, "der_out"); return -1; }
    if(input == NULL) { DL_BADPOINTER(1, "input"); return -1; }
    if(w == NULL) { DL_BADPOINTER(1, "w"); return -1; }
    if(bias == NULL) { DL_BADPOINTER(1, "bias"); return -1; }

    if(in_size <= 0) { DL_BADVALUE(1, "in_size: %d", in_size); return -1; }
    if(out_size <= 0) { DL_BADVALUE(1, "out_size: %d", out_size); return -1; }
    if(!act) {DL_BADVALUE(1, "act"); return -1;}
    if(!der_act) {DL_BADVALUE(1, "der_act"); return -1;}
#endif

    auto out_end = out + out_size;
    auto input_it = input;
    auto input_end = input + in_size;
    nvt v = 0;
    while(out != out_end)
    {
        v = *bias;
        input_it = input;
        while(input_it != input_end)
        {
            v += *input_it * *w;
            ++input_it;
            ++w;
        }
        *raw = v;
        *out = act(*raw);
        *der_out = der_act(*raw);

        ++raw;
        ++out;
        ++der_out;
        ++bias;
    }
    return 0;
}
int DSynapse::forward_propagation_learn_masked(DSynapse::data_line out, DSynapse::data_line raw, DSynapse::data_line der_out,
                                         DSynapse::const_mask_t outMask,
                                         DSynapse::const_data_line input, DSynapse::const_data_line w, DSynapse::const_data_line bias,
                                         int in_size, int out_size, DSynapse::ACTF act, DSynapse::ACTF der_act)
{
#ifdef DSYNAPSE_USE_FORWARD_PROP_SAVE_CALL
    if(out == NULL) { DL_BADPOINTER(1, "out"); return -1; }
    if(raw == NULL) { DL_BADPOINTER(1, "raw"); return -1; }
    if(der_out == NULL) { DL_BADPOINTER(1, "der_out"); return -1; }
    if(outMask == NULL) { DL_BADPOINTER(1, "outMask"); return -1; }
    if(input == NULL) { DL_BADPOINTER(1, "input"); return -1; }
    if(w == NULL) { DL_BADPOINTER(1, "w"); return -1; }
    if(bias == NULL) { DL_BADPOINTER(1, "bias"); return -1; }

    if(in_size <= 0) { DL_BADVALUE(1, "in_size: %d", in_size); return -1; }
    if(out_size <= 0) { DL_BADVALUE(1, "out_size: %d", out_size); return -1; }
    if(!act) {DL_BADVALUE(1, "act"); return -1;}
    if(!der_act) {DL_BADVALUE(1, "der_act"); return -1;}
#endif

    auto out_end = out + out_size;
    auto input_it = input;
    auto input_end = input + in_size;
    nvt v = 0;
    while(out != out_end)
    {
        if(*outMask)
        {
            input_it = input;
            v = *bias;
            while(input_it != input_end)
            {
                v += *input_it * *w;
                ++input_it;
                ++w;
            }
            *raw = v;
            *out = act(*raw);
            *der_out = der_act(*raw);
        }
        else
        {
            *raw = 0.0;
            *out = 0.0;
            *der_out = 0.0;
        }

        ++outMask;
        ++raw;
        ++out;
        ++der_out;
        ++bias;
    }
    return 0;
}
int DSynapse::back_propagation(DSynapse::data_line curr_error, DSynapse::data_line prev_error, DSynapse::data_line w,
                               DSynapse::const_data_line prev_input, DSynapse::const_data_line curr_der_input, DSynapse::data_line bias, DSynapse::nvt learn_rate,
                               int curr_size, int prev_size, void *special)
{
#ifdef DSYNAPSE_USE_BACK_PROP_SAVE_CALL
    if(curr_error == NULL) { DL_BADPOINTER(1, "curr_error"); return -1; }
    if(prev_error == NULL) { DL_BADPOINTER(1, "prev_error"); return -1; }
    if(w == NULL) { DL_BADPOINTER(1, "w"); return -1; }
    if(prev_input == NULL) { DL_BADPOINTER(1, "prev_input"); return -1; }
    if(curr_der_input == NULL) { DL_BADPOINTER(1, "curr_der_input"); return -1; }
    if(bias == NULL) { DL_BADPOINTER(1, "bias"); return -1; }

    if(curr_size <= 0) { DL_BADVALUE(1, "curr_size: %d", curr_size); return -1; }
    if(prev_size <= 0) { DL_BADVALUE(1, "prev_size: %d", prev_size); return -1; }
    if(special == NULL) {DL_BADVALUE(1, "act"); return -1;}
#endif

    DescentContext *dc = reinterpret_cast<DescentContext*>(special);
    nvt dw;
    nvt delta;
    auto curr_error_end = curr_error + curr_size;
    auto prev_error_it = prev_error;
    auto prev_error_end = prev_error + prev_size;
    auto input_it = prev_input;
    zero_mem(prev_error, prev_size);

    auto support_w_buffer_it = dc->use_support_buffer ? dc->deltaBuffer->main : NULL;
    auto support_b_buffer_it = dc->use_support_buffer ? dc->deltaBuffer->bias : NULL;
    auto main_w_buffer_it = dc->use_main_buffer ? w : NULL;
    auto main_b_buffer_it = dc->use_main_buffer ? bias : NULL;

    while(curr_error != curr_error_end)
    {
        prev_error_it = prev_error;
        input_it = prev_input;

        *curr_error *= *curr_der_input;
        dw = *curr_error * learn_rate;

        while(prev_error_it != prev_error_end)
        {
            *prev_error_it += *curr_error * *w;
            delta = dw * *input_it;


//            std::cout << "back_propagation:"
//                      << " delta: " << delta
//                      << " support_w_buffer_it: " << (void*)support_w_buffer_it
//                      << " main_w_buffer_it: " << (void*)main_w_buffer_it
//                      << " " << *main_w_buffer_it
//                      << std::endl;

            if(support_w_buffer_it)
                *support_w_buffer_it++ += delta;
            if(main_w_buffer_it)
                *main_w_buffer_it++ += delta;

            ++w;
            ++prev_error_it;
            ++input_it;
        }
        if(support_b_buffer_it)
            *support_b_buffer_it++ += dw;
        if(main_b_buffer_it)
            *main_b_buffer_it++ += dw;

        ++bias;
        ++curr_error;
        ++curr_der_input;
    }

    return 0;
}
int DSynapse::back_propagation_masked(DSynapse::data_line curr_error, DSynapse::data_line prev_error, DSynapse::data_line w,
                                DSynapse::const_data_line currMask,
                                DSynapse::const_data_line prev_input, DSynapse::const_data_line curr_der_input, DSynapse::data_line bias, DSynapse::nvt learn_rate,
                                int curr_size, int prev_size, void *special)
{
    /*
#ifdef DSYNAPSE_USE_BACK_PROP_SAVE_CALL
    if(curr_error == NULL) { DL_BADPOINTER(1, "curr_error"); return -1; }
    if(prev_error == NULL) { DL_BADPOINTER(1, "prev_error"); return -1; }
    if(w == NULL) { DL_BADPOINTER(1, "w"); return -1; }
    if(currMask == NULL) { DL_BADPOINTER(1, "currMask"); return -1; }
    if(prev_input == NULL) { DL_BADPOINTER(1, "prev_input"); return -1; }
    if(curr_der_input == NULL) { DL_BADPOINTER(1, "curr_der_input"); return -1; }
    if(bias == NULL) { DL_BADPOINTER(1, "bias"); return -1; }

    if(curr_size <= 0) { DL_BADVALUE(1, "curr_size: %d", curr_size); return -1; }
    if(prev_size <= 0) { DL_BADVALUE(1, "prev_size: %d", prev_size); return -1; }
    if(special == NULL) {DL_BADVALUE(1, "act"); return -1;}
#endif

    DescentContext *dc = reinterpret_cast<DescentContext*>(special);
    nvt dw;
    nvt delta;
    auto curr_error_end = curr_error + curr_size;
    auto prev_error_it = prev_error;
    auto prev_error_end = prev_error + prev_size;
    auto input_it = prev_input;
    zero_mem(prev_error, prev_size);

    auto delta_w_it = dc->delta_w;
    auto delta_b_it = dc->delta_b;

    while(curr_error != curr_error_end)
    {
        if(*currMask)
        {
            prev_error_it = prev_error;
            input_it = prev_input;

            *curr_error *= *curr_der_input;
            dw = *curr_error * learn_rate;

            while(prev_error_it != prev_error_end)
            {
                *prev_error_it += *curr_error * *w;
                delta = dw * *input_it;

                if(dc->use_support_buffer)
                    *delta_w_it++ += delta;
                if(dc->use_main_buffer)
                    *w += delta;

                ++w;
                ++prev_error_it;
                ++input_it;
            }
            if(dc->use_support_buffer)
                *delta_b_it++ += dw;
            if(dc->use_main_buffer)
                *bias += dw;
        }


        ++currMask;
        ++bias;
        ++curr_error;
        ++curr_der_input;
    }
    */
    return 0;
}
int DSynapse::back_propagation_opt(DSynapse::data_line curr_error, DSynapse::data_line prev_error, DSynapse::data_line w,
                                   DSynapse::const_data_line prev_input, DSynapse::const_data_line curr_der_input, DSynapse::data_line bias, DSynapse::nvt learn_rate,
                                   int curr_size, int prev_size, DSynapse::gradient_descent_callback gd_callback, void *special)
{
    /*
#ifdef DSYNAPSE_USE_BACK_PROP_SAVE_CALL
    if(curr_error == NULL) { DL_BADPOINTER(1, "curr_error"); return -1; }
    if(prev_error == NULL) { DL_BADPOINTER(1, "prev_error"); return -1; }
    if(w == NULL) { DL_BADPOINTER(1, "w"); return -1; }
    if(prev_input == NULL) { DL_BADPOINTER(1, "prev_input"); return -1; }
    if(curr_der_input == NULL) { DL_BADPOINTER(1, "curr_der_input"); return -1; }
    if(bias == NULL) { DL_BADPOINTER(1, "bias"); return -1; }

    if(curr_size <= 0) { DL_BADVALUE(1, "curr_size: %d", curr_size); return -1; }
    if(prev_size <= 0) { DL_BADVALUE(1, "prev_size: %d", prev_size); return -1; }
    if(special == NULL) {DL_BADVALUE(1, "act"); return -1;}
#endif

    DescentContext *dc = reinterpret_cast<DescentContext*>(special);
    nvt dw;
    nvt delta;
    auto curr_error_end = curr_error + curr_size;
    auto prev_error_it = prev_error;
    auto prev_error_end = prev_error + prev_size;
    auto input_it = prev_input;
    zero_mem(prev_error, prev_size);

    auto delta_w_it = dc->delta_w;
    auto delta_b_it = dc->delta_b;

    while(curr_error != curr_error_end)
    {
        prev_error_it = prev_error;
        input_it = prev_input;

        *curr_error *= *curr_der_input;
        dw = *curr_error * learn_rate;

        while(prev_error_it != prev_error_end)
        {
            *prev_error_it += *curr_error * *w;
            delta = gd_callback(dc);

            if(dc->use_support_buffer)
                *delta_w_it++ += delta;
            if(dc->use_main_buffer)
                *w += delta;

            ++w;
            ++prev_error_it;
            ++input_it;
        }
        if(dc->use_support_buffer)
            *delta_b_it++ += dw;
        if(dc->use_main_buffer)
            *bias += dw;

        ++bias;
        ++curr_error;
        ++curr_der_input;
    }

    */
    return 0;
}







int DSynapse::loss_function_default(DSynapse::const_data_line out, DSynapse::const_data_line target, DSynapse::data_line out_error, DSynapse::nvt *total_error, int size)
{
#ifdef DSYNAPSE_USE_LOSSF_SAVE_CALL
    if(out == NULL) {DL_BADPOINTER(1, "out"); return -1;}
    if(target == NULL) {DL_BADPOINTER(1, "target"); return -1;}
    if(out_error == NULL) {DL_BADPOINTER(1, "out_error"); return -1;}
    if(total_error == NULL) {DL_BADPOINTER(1, "total_error"); return -1;}
    if(size <= 0) {DL_BADVALUE(1, "size: [%d]", size); return -1;}
#endif
    auto e_out = out + size;
    while(out != e_out){
        *out_error = *target - *out;
        *total_error += fabs(*out_error);
        ++out; ++target; ++out_error;
    }
    return 0;
}
int DSynapse::loss_function_quadratic(DSynapse::const_data_line out, DSynapse::const_data_line target, DSynapse::data_line out_error, DSynapse::nvt *total_error, int size)
{
#ifdef DSYNAPSE_USE_LOSSF_SAVE_CALL
    if(out == NULL) {DL_BADPOINTER(1, "out"); return -1;}
    if(target == NULL) {DL_BADPOINTER(1, "target"); return -1;}
    if(out_error == NULL) {DL_BADPOINTER(1, "out_error"); return -1;}
    if(total_error == NULL) {DL_BADPOINTER(1, "total_error"); return -1;}
    if(size <= 0) {DL_BADVALUE(1, "size: [%d]", size); return -1;}
#endif
    //0.5(t-y)^2
    //C' = y-t
    auto e_out = out + size;
    while(out != e_out){
        *out_error = *out - *target;
        *total_error += 0.5 * pow(*target - *out, 2);
        ++out; ++target; ++out_error;
    }
    return 0;
}
int DSynapse::loss_function_cross_entropy(DSynapse::const_data_line out, DSynapse::const_data_line target, DSynapse::data_line out_error, DSynapse::nvt *total_error, int size)
{
#ifdef DSYNAPSE_USE_LOSSF_SAVE_CALL
    if(out == NULL) {DL_BADPOINTER(1, "out"); return -1;}
    if(target == NULL) {DL_BADPOINTER(1, "target"); return -1;}
    if(out_error == NULL) {DL_BADPOINTER(1, "out_error"); return -1;}
    if(total_error == NULL) {DL_BADPOINTER(1, "total_error"); return -1;}
    if(size <= 0) {DL_BADVALUE(1, "size: [%d]", size); return -1;}
#endif
    auto e_out = out + size;
    while(out != e_out){
        *out_error = *out - *target;
        *total_error += *target * log(*out);
        ++out; ++target; ++out_error;
    }
    return 0;
}











