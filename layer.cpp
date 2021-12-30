#include "layer.h"
#include "net.h"
#include <daran.h>
#include <math.h>

DSynapse::LAYER_P DSynapse::alloc_layer(int _size)
{
    LAYER_P l = NULL;
    if(_size <= 0)
    {
        DL_BADVALUE(1, "size");
        goto fail;
    }
    if(  (l = get_zmem<layer>(1)) == NULL  )
    {
        DL_BADALLOC(1, "layer");
        goto fail;
    }
    l->der_input = NULL;
//    l->weight = NULL;
//    l->bias = NULL;
//    l->w_size = 0;
//    l->b_size = 0;
    l->weigthMatrix = NULL;

    l->next = NULL;
    l->prev = NULL;

    l->learn_rate = 0.0;

    l->size = _size;
    l->act = act_sigmoid;
    l->der_act = der_act_sigmoid;
    l->back_prop = back_propagation;


    if(  (l->input = alloc_buffer(_size)) == NULL )
    {
        DL_BADALLOC(1, "input buffer");
        goto fail;
    }
    if(  (l->error = alloc_buffer(_size)) == NULL )
    {
        DL_BADALLOC(1, "error buffer");
        goto fail;
    }
    if(  (l->raw = alloc_buffer(_size)) == NULL )
    {
        DL_BADALLOC(1, "raw buffer");
        goto fail;
    }
    if(  (l->dcontext = alloc_descent_context()) == NULL  )
    {
        DL_BADALLOC(1, "dcontext");
        goto fail;
    }

    return l;
fail:
    free_layer(l);
    return NULL;
}
DSynapse::LAYER_P DSynapse::layer_copy(const DSynapse::LAYER_P copy_this, int size, int replace_weight_matrix)
{
    LAYER_P dst = NULL;
    LAYER_P src = copy_this;
    if(src == NULL)
    {
        DL_BADPOINTER(1, "src layer");
        goto fail;
    }
    size = size ? size : src->size;
    if(  (dst = alloc_layer(size)) == NULL  )
    {
        DL_BADALLOC(1, "dst layer");
        goto fail;
    }
    if(src->input)
    {
        if(  (dst->input = alloc_buffer(size)) == NULL  ) {DL_BADALLOC(1, "input buffer"); goto fail;}
    }
    if(src->error)
    {
        if(  (dst->error = alloc_buffer(size)) == NULL  ) {DL_BADALLOC(1, "error buffer"); goto fail;}
    }
    if(src->raw)
    {
        if(  (dst->raw = alloc_buffer(size)) == NULL  ) {DL_BADALLOC(1, "raw buffer"); goto fail;}
    }
    /*
    if(src->der_input)
    {
        if(  (dst->der_input = alloc_buffer(size)) == NULL  ) {DL_BADALLOC(1, "der_input buffer"); goto fail;}
    }
    if(src->weight)
    {
        if(replace_weight_matrix)
            dst->weight = src->weight;
        else
        {
            if(  (dst->weight = alloc_buffer(src->w_size)) == NULL  ) {DL_BADALLOC(1, "weight buffer"); goto fail;}
        }
    }
    if(src->bias)
    {
        if(replace_weight_matrix)
            dst->bias = src->bias;
        else
        {
            if(src->next == NULL) {DL_BADPOINTER(1, "next layer"); goto fail;}
            if(  (dst->bias = alloc_buffer(src->next->size)) == NULL  ) {DL_BADALLOC(1, "bias buffer"); goto fail;}
        }
    }
    dst->w_size = src->w_size;
    dst->b_size = src->b_size;
    */

    dst->size = size;
    dst->learn_rate = src->learn_rate;
    dst->act = src->act;
    dst->der_act = src->der_act;
    dst->back_prop = src->back_prop;
    dst->use_act_out = src->use_act_out;

    if(  (dst->dcontext = copy_descent_context(src->dcontext)) == NULL  )
    {
        DL_BADALLOC(1, "dcontext");
        goto fail;
    }


    return dst;
fail:
    free_layer(dst);
    return NULL;
}
DSynapse::LAYER_P DSynapse::layer_unique_copy(DSynapse::LAYER_P copy_this)
{
    return layer_copy(copy_this, 0);
}
DSynapse::LAYER_P DSynapse::layer_shared_copy(DSynapse::LAYER_P copy_this)
{
    return layer_copy(copy_this, 1);
}
DSynapse::LAYER_P DSynapse::layer_modify(DSynapse::LAYER_P l, int new_size)
{
    if(l == NULL) {DL_BADPOINTER(1, "layer"); return NULL;}
    l->size = new_size;

    if(  (l->input = reget_mem(l->input, l->size)) == NULL )
    {
        DL_BADALLOC(1, "input buffer");
        return NULL;
    }
    if(  (l->error = reget_mem(l->error, l->size)) == NULL )
    {
        DL_BADALLOC(1, "error buffer");
        return NULL;
    }
    if(  (l->raw = reget_mem(l->raw, l->size)) == NULL )
    {
        DL_BADALLOC(1, "raw buffer");
        return NULL;
    }

    return l;
}
int DSynapse::free_layer(DSynapse::LAYER_P l)
{
    if(l)
    {
        free_mem(l->input);
        free_mem(l->raw);
        free_mem(l->error);
        free_mem(l->der_input);
//        free_mem(l->weight);
//        free_mem(l->bias);
        wm_free(l->weigthMatrix);

        free_descent_context(l->dcontext);
        zero_mem(l, sizeof(layer));
        free_mem(l);
    }
    return 0;
}
int DSynapse::layer_get_size(DSynapse::LAYER_P l)
{
    if(l == NULL) {DL_BADPOINTER(1, "layer");return -1;}
    return l->size;
}
int DSynapse::layer_alloc_wm(DSynapse::LAYER_P l) {

    if(l == NULL) {DL_BADPOINTER(1, "layer");return -1;}
    if(l->next == nullptr) {DL_BADPOINTER(1, "next"); return -1;}

    if(l->weigthMatrix) {DL_ERROR(1, "weigth matrix already exist"); return -1;}
    if(l->size <= 0) {DL_BADVALUE(1, "current size: [%d]", l->size); return -1;}
    if(l->next->size <= 0) {DL_BADVALUE(1, "next size: [%d]", l->next->size); return -1;}

    if( (l->weigthMatrix = wm_alloc()) == NULL ) {
        DL_BADALLOC(1, "weightMatrix");
        return -1;
    }
    if( wm_create_matrix2(l->weigthMatrix, l->size, l->next->size) < 0 ) {
        DL_FUNCFAIL(1, "wm_create_matrix2");
        return -1;
    }
    return 0;
}
int DSynapse::layer_set_weight(DSynapse::LAYER_P l, int wpos, DSynapse::nvt value)
{
    if(l == NULL) {DL_BADPOINTER(1, "layer");return -1;}
    /*
    if(l->weight == NULL) {DL_BADPOINTER(1, "weight"); return -1;}
    if(wpos < 0 || wpos > l->w_size) {DL_BADVALUE(1, "wpos: [%d] layer w_size: [%d]", wpos, l->w_size); return -1;}
    l->weight[wpos] = value;
    */
    wm_set_weight_value(l->weigthMatrix, wpos, value);
    return 0;
}
int DSynapse::layer_set_weight(DSynapse::LAYER_P l, int currentNeuron, int nextNeuron, DSynapse::nvt value)
{
    if(l == NULL) {DL_BADPOINTER(1, "layer");return -1;}
    /*
    if(l->weight == NULL) {DL_BADPOINTER(1, "weight"); return -1;}
    if(l->next == NULL) {DL_BADPOINTER(1, "next"); return -1;}
    if(currentNeuron < 0 || currentNeuron > l->size) {DL_BADVALUE(1, "currentNeuron: [%d] layer size: [%d]", currentNeuron, l->size); return -1;}
    if(nextNeuron < 0 || nextNeuron > l->next->size) {DL_BADVALUE(1, "nextNeuron: [%d] next layer size: [%d]", nextNeuron, l->next->size); return -1;}
    l->weight[nextNeuron * l->size + currentNeuron] = value;
    */
    wm_set_weight_value(l->weigthMatrix, nextNeuron * l->size + currentNeuron, value);
    return 0;
}
int DSynapse::layer_clear_buffers(DSynapse::LAYER_P l)
{
    if(l == NULL) {DL_BADPOINTER(1, "layer");return -1;}
    if(l->size <=0) {DL_BADVALUE(1, "size"); return -1;}
    if(l->input) zero_mem(l->input, l->size);
    if(l->raw) zero_mem(l->raw, l->size);
    if(l->error) zero_mem(l->error, l->size);
    if(l->der_input) zero_mem(l->der_input, l->size);
    /*
    if(l->weight && l->w_size > 0) zero_mem(l->weight, l->w_size);
    if(l->bias && l->b_size > 0) zero_mem(l->bias, l->b_size);
    */
    wm_zero_matrix(l->weigthMatrix, WM_BOTH);

    descent_context_clear_buffers(l->dcontext);

    return 0;
}
int DSynapse::layer_set_rand_signal(LAYER_P l)
{
    if(l == NULL) {DL_BADPOINTER(1, "layer");return -1;}
    if(l->size <= 0) {DL_BADVALUE(1, "size: [%d]", l->size);return -1;}
    if(l->input == NULL) {DL_BADPOINTER(1, "input");return -1;}

    FOR_VALUE(l->size, i)
    {
        l->input[i] = global_rand_callback();
    }
    if(l->raw)
        copy_mem(l->raw, l->input, l->size);

    return 0;
}
int DSynapse::layer_set_randw(DSynapse::LAYER_P l)
{
    if(l == NULL) {DL_BADPOINTER(1, "layer");return -1;}

    /*
    if(l->weight == NULL || l->bias == NULL)
        return 0;

    FOR_VALUE(l->w_size, i)
    {
        l->weight[i] = global_rand_callback();
    }
    FOR_VALUE(l->b_size, i)
    {
        l->bias[i] = global_rand_callback();
    }
    */
    wm_set_rand(l->weigthMatrix, WM_BOTH);

    return 0;
}
int DSynapse::layer_check(DSynapse::LAYER_P l)
{
    if(l == NULL) { DL_BADPOINTER(1, "layer");return -1; }
    if(l->input == NULL) { DL_BADPOINTER(1, "input");return -1; }
    if(l->raw == NULL) { DL_BADPOINTER(1, "raw");return -1; }
    if(l->error == NULL) { DL_BADPOINTER(1, "error");return -1; }
    if(l->size <= 0) {DL_BADVALUE(1, "size: [%d]", l->size);return -1; }
    if(l->prev == NULL && l->next == NULL) { DL_ERROR(1, "Disconnected layer");return -1; }
    /*
    if(l->next && l->weight == NULL)
    {
        DL_BADPOINTER(1, "weight");
        return -1;
    }
    if(l->next && l->bias == NULL)
    {
        DL_BADPOINTER(1, "bias");
        return -1;
    }
    */
    if(l->weigthMatrix == NULL) { DL_BADPOINTER(1, "weightMatrix"); return -1; }
    if(l->weigthMatrix->main == NULL ) { DL_BADPOINTER(1, "wm: main"); return -1; }
    if(l->weigthMatrix->mainSize <= 0 ) { DL_BADVALUE(1, "wm: mainSize: [%d]", l->weigthMatrix->mainSize); return -1; }
    if(l->weigthMatrix->bias == NULL ) { DL_BADPOINTER(1, "wm: bias"); return -1; }
    if(l->weigthMatrix->biasSize <= 0 ) { DL_BADVALUE(1, "wm: biasSize: [%d]", l->weigthMatrix->biasSize); return -1; }

    if(l->prev && l->der_input == NULL) { DL_BADPOINTER(1, "der_input");return -1; }

    return 0;
}
int DSynapse::layer_free_weight(DSynapse::LAYER_P l)
{
    if(l == NULL){DL_BADPOINTER(1, "layer");return -1;}
    /*
    free_mem(l->weight);
    free_mem(l->bias);
    l->w_size = 0;
    l->b_size = 0;
    */

    wm_free(l->weigthMatrix);
    return 0;
}
int DSynapse::layer_alloc_derinput(DSynapse::LAYER_P l)
{
    if(l == NULL) {DL_BADPOINTER(1, "layer");return -1;}

    if(l->prev)
    {
        if(  (l->der_input = reget_zmem<nvt>(l->der_input, l->size)) == NULL  ) {DL_BADALLOC(1, "der_input"); return -1;}
    }

    return 0;
}
int DSynapse::layer_alloc_connection_buffers(DSynapse::LAYER_P l)
{
    if( layer_alloc_wm(l) < 0 ) {
        DL_FUNCFAIL(1, "layer_alloc_wm");
        return -1;
    }
    layer_alloc_derinput(l);
    return 0;
}

int DSynapse::layer_show_input(DSynapse::LAYER_P l)
{
    if(l == NULL) {DL_BADPOINTER(1, "layer");return -1;}

    if(l->input)
    {
        FOR_VALUE(l->size, i)
            std::cout << l->input[i] << " ";
        std::cout << std::endl;
    }
    else
    {
        DL_BADPOINTER(1, "input");
        return -1;
    }

    return 0;
}
int DSynapse::layer_show_raw(DSynapse::LAYER_P l)
{
    if(l == NULL) {DL_BADPOINTER(1, "layer");return -1;}

    if(l->raw)
    {
        FOR_VALUE(l->size, i)
            std::cout << l->raw[i] << " ";
        std::cout << std::endl;
    }
    else
    {
        DL_BADPOINTER(1, "raw");
        return -1;
    }

    return 0;
}
int DSynapse::layer_show_deri(DSynapse::LAYER_P l)
{
    if(l == NULL) {DL_BADPOINTER(1, "layer");return -1;}

    if(l->der_input)
    {
        FOR_VALUE(l->size, i)
            std::cout << l->der_input[i] << " ";
        std::cout << std::endl;
    }
    else
    {
        DL_BADPOINTER(1, "der_input");
        return -1;
    }

    return 0;
}
int DSynapse::layer_show_signals(DSynapse::LAYER_P l)
{
    layer_show_raw(l);
    layer_show_input(l);
    layer_show_deri(l);

    return 0;
}
int DSynapse::layer_show_error(DSynapse::LAYER_P l)
{
    if(l == NULL) {DL_BADPOINTER(1, "layer");return -1;}

    if(l->error)
    {
        FOR_VALUE(l->size, i)
            std::cout << l->error[i] << " ";
        std::cout << std::endl;
    }
    else
    {
        DL_BADPOINTER(1, "error");
        return -1;
    }

    return 0;
}
int DSynapse::layer_show_w(DSynapse::LAYER_P l)
{
    if(l == NULL) {DL_BADPOINTER(1, "layer");return -1;}

    /*
    if(l->weight && l->bias && l->next)
    {
        std::cout << "layer: " << (void*)l
                  << " size: " << l->size
                  << std::endl;

        FOR_VALUE(l->next->size, i)
        {
            FOR_VALUE(l->size, j)
                    std::cout << l->weight[i*l->size + j] << " ";
            std::cout << "(bias: " << l->bias[i] << ")" << std::endl;
        }
        std::cout << std::endl;
    }
    */
    wm_print(l->weigthMatrix, WM_BOTH);

    return 0;
}
int DSynapse::layer_show_delta(DSynapse::LAYER_P l)
{
    if(l == NULL) {DL_BADPOINTER(1, "layer");return -1;}

    /*
    if(l->weight && l->bias && l->next && l->dcontext)
    {
        std::cout << "layer: " << (void*)l
                  << " size: " << l->size
                  << std::endl;

        FOR_VALUE(l->next->size, i)
        {
            FOR_VALUE(l->size, j)
                    std::cout << l->dcontext->delta_w[i*l->size + j] << " ";
            std::cout << "(bias: " << l->dcontext->delta_b[i] << ")" << std::endl;
        }
        std::cout << std::endl;
    }
    */



    return 0;
}

int DSynapse::layer_alloc_delta(DSynapse::LAYER_P l)
{
    if(l == NULL) {DL_BADPOINTER(1, "layer");return -1;}
    if(l->dcontext == NULL) {DL_BADPOINTER(1, "dcontext");return -1;}
/*
    l->dcontext->w_size = l->w_size;
    l->dcontext->b_size = l->b_size;
    if(l->dcontext->delta_w == NULL)
    {
        if(  (l->dcontext->delta_w = alloc_buffer(l->dcontext->w_size)) == NULL  )
        {
            DL_BADALLOC(1, "delta_w");
            goto fail;
        }
    }
    if(l->dcontext->delta_b == NULL)
    {
        if(  (l->dcontext->delta_b = alloc_buffer(l->dcontext->b_size)) == NULL  )
        {
            DL_BADALLOC(1, "delta_b");
            goto fail;
        }
    }
    return 0;
fail:
    free_mem(l->dcontext->delta_w);
    free_mem(l->dcontext->delta_b);
    return -1;
    */
    if(l->dcontext->deltaBuffer == NULL) {
        if( (l->dcontext->deltaBuffer = wm_alloc()) == NULL ) {DL_BADALLOC(1, "deltaBuffer"); return -1;}
    }
    if(l->weigthMatrix) {
        if( wm_recreate_matrix(l->dcontext->deltaBuffer, l->weigthMatrix->mainSize, l->weigthMatrix->biasSize) < 0 )
            {DL_FUNCFAIL(1, "wm_recreate_matrix"); return -1;}
    }
    else {
        if(l->next == NULL) {DL_BADPOINTER(1, "next"); return -1;}
        if(l->next->size <= 0) {DL_BADVALUE(1, "next: size: [%d]", l->next->size); return -1;}
        if( wm_recreate_matrix2(l->dcontext->deltaBuffer, l->size, l->next->size) < 0 )
            {DL_FUNCFAIL(1, "wm_recreate_matrix2"); return -1;}
    }
    return -1;
}
int DSynapse::layer_free_delta(DSynapse::LAYER_P l)
{
    if(l == NULL) {DL_BADPOINTER(1, "layer");return -1;}
    if(l->dcontext)
    {
        /*
        if(l->dcontext->w_size > 0) zero_mem(l->dcontext->delta_w, l->dcontext->w_size);
        if(l->dcontext->b_size > 0) zero_mem(l->dcontext->delta_b, l->dcontext->b_size);
        free_mem(l->dcontext->delta_w);
        free_mem(l->dcontext->delta_b);
        */
        wm_free(l->dcontext->deltaBuffer);
        l->dcontext->deltaBuffer = NULL;
    }

    return 0;
}
int DSynapse::layer_apply_delta(DSynapse::LAYER_P l)
{
#ifdef LAYER_USE_SAVE_CALL
    if(l == NULL) {DL_BADPOINTER(1, "layer");return -1;}
    if(l->dcontext == NULL) {DL_BADPOINTER(1, "dcontext");return -1;}

    if(l->dcontext->deltaBuffer == NULL) {DL_BADPOINTER(1, "deltaBuffer"); return -1;}
    if(l->weigthMatrix == NULL) {DL_BADPOINTER(1, "weightMatrix"); return -1;}
    if(l->dcontext->deltaBuffer->main == NULL) {DL_BADPOINTER(1, "deltaBuffer: main"); return -1;}
    if(l->dcontext->deltaBuffer->bias == NULL) {DL_BADPOINTER(1, "deltaBuffer: bias"); return -1;}
    if(l->weigthMatrix->main == NULL) {DL_BADPOINTER(1, "weightMatrix: main"); return -1;}
    if(l->weigthMatrix->bias == NULL) {DL_BADPOINTER(1, "weightMatrix: bias"); return -1;}
    if(l->weigthMatrix->mainSize != l->dcontext->deltaBuffer->mainSize) {
        DL_BADVALUE(1, "different main sizes: weightMatrix: [%d] deltaBuffer: [%d]", l->weigthMatrix->mainSize, l->dcontext->deltaBuffer->mainSize);
        return -1;
    }
    if(l->weigthMatrix->biasSize != l->dcontext->deltaBuffer->biasSize) {
        DL_BADVALUE(1, "different bias sizes: weightMatrix: [%d] deltaBuffer: [%d]", l->weigthMatrix->biasSize, l->dcontext->deltaBuffer->biasSize);
        return -1;
    }
    /*
    if(l->next == NULL) {DL_BADPOINTER(1, "next");return -1;}
    if(l->weight == NULL) {DL_BADPOINTER(1, "weight");return -1;}
    if(l->dcontext->delta_w == NULL) { DL_BADPOINTER(1, "delta_w");return -1;}
    if(l->dcontext->delta_b == NULL) {DL_BADPOINTER(1, "delta_b");return -1;}
    if(l->w_size != l->dcontext->w_size) {DL_BADVALUE(1, "w_size: inner: [%d] dcontext: [%d]", l->w_size, l->dcontext->w_size);return -1;}
    if(l->next->size != l->dcontext->b_size) {DL_BADVALUE(1, "b_size: inner: [%d] dcontext: [%d]", l->next->size, l->dcontext->b_size);return -1;}
    */
#endif
    if(l->dcontext->use_support_buffer)
    {
        /*
        FOR_VALUE(l->dcontext->w_size, i)
        {
            l->weight[i] += l->dcontext->delta_w[i];
        }
        FOR_VALUE(l->dcontext->b_size, i)
        {
            l->bias[i] += l->dcontext->delta_b[i];
        }
        */

        FOR_VALUE(l->dcontext->deltaBuffer->mainSize, i){
            l->weigthMatrix->main[i] += l->dcontext->deltaBuffer->main[i];
        }
        FOR_VALUE(l->dcontext->deltaBuffer->biasSize, i){
            l->weigthMatrix->bias[i] += l->dcontext->deltaBuffer->bias[i];
        }
    }

    return 0;
}
int DSynapse::layer_zero_delta(DSynapse::LAYER_P l)
{
#ifdef LAYER_USE_SAVE_CALL
    if(l == NULL) {DL_BADPOINTER(1, "layer"); return -1;}
    if(l->dcontext == NULL) {DL_BADPOINTER(1, "descent context"); return -1;}
    /*
    if(l->dcontext->delta_w == NULL) {DL_BADPOINTER(1, "delta_w"); return -1;}
    if(l->dcontext->delta_b == NULL) {DL_BADPOINTER(1, "delta_b"); return -1;}
    */
    if(l->dcontext->deltaBuffer == NULL) {DL_BADPOINTER(1, "deltaBuffer"); return -1;}
    if(l->dcontext->deltaBuffer->main == NULL) {DL_BADPOINTER(1, "deltaBuffer: main"); return -1;}
    if(l->dcontext->deltaBuffer->bias == NULL) {DL_BADPOINTER(1, "deltaBuffer: bias"); return -1;}
#endif
    /*
    zero_mem(l->dcontext->delta_w, l->dcontext->w_size);
    zero_mem(l->dcontext->delta_b, l->dcontext->b_size);
    */
    wm_zero_matrix(l->dcontext->deltaBuffer, WM_BOTH);

    return 0;
}
int DSynapse::layer_flush_delta(DSynapse::LAYER_P l)
{
    layer_apply_delta(l);
    layer_zero_delta(l);
    return 0;
}
int DSynapse::layer_set_next(DSynapse::LAYER_P l, DSynapse::LAYER_P next)
{
    if(l == NULL) {DL_BADPOINTER(1, "layer");return -1;}
    if(next == NULL) {DL_BADPOINTER(1, "next");return -1;}
    l->next = next;

    return 0;
}
int DSynapse::layer_set_prev(DSynapse::LAYER_P l, DSynapse::LAYER_P prev)
{
    if(l == NULL) {DL_BADPOINTER(1, "layer");return -1;}
    if(prev == NULL) {DL_BADPOINTER(1, "prev");return -1;}
    l->prev = prev;

    return 0;
}
int DSynapse::layer_apply_delta(DSynapse::LAYER_P l, DSynapse::LAYER_P from)
{
#ifdef LAYER_USE_SAVE_CALL
    if(l == NULL) {DL_BADPOINTER(1, "layer");return -1;}
    if(from == NULL)  {DL_BADPOINTER(1, "from");return -1;}
    if(l->dcontext == NULL) {DL_BADPOINTER(1, "dcontext");return -1;}
    if(from->dcontext == NULL) {DL_BADPOINTER(1, "from->dcontext");return -1;}


    /*
    if(l->weight == NULL) {DL_BADPOINTER(1, "weight");return -1;}
    if(from->dcontext->delta_w == NULL) {DL_BADPOINTER(1, "delta_w");return -1;}
    if(from->dcontext->delta_b == NULL){DL_BADPOINTER(1, "delta_b");return -1;}
    if(l->w_size != from->dcontext->w_size)
    {DL_BADVALUE(1, "w_size: inner: [%d] dcontext: [%d]", l->w_size, from->dcontext->w_size);return -1;}
    if(l->next == NULL){DL_BADPOINTER(1, "next");return -1;}
    if(l->next->size != from->dcontext->b_size){DL_BADVALUE(1, "b_size: inner: [%d] dcontext: [%d]", l->next->size, l->dcontext->b_size);return -1;}
    */

    if(l->weigthMatrix == NULL) {DL_BADPOINTER(1, "weightMatrix"); return -1;}
    if(from->dcontext->deltaBuffer == NULL) {DL_BADPOINTER(1, "deltaBuffer"); return -1;}
    if(from->dcontext->deltaBuffer->main == NULL) {DL_BADPOINTER(1, "deltaBuffer: main"); return -1;}
    if(from->dcontext->deltaBuffer->bias == NULL) {DL_BADPOINTER(1, "deltaBuffer: bias"); return -1;}
    if(l->weigthMatrix->main == NULL) {DL_BADPOINTER(1, "weightMatrix: main"); return -1;}
    if(l->weigthMatrix->bias == NULL) {DL_BADPOINTER(1, "weightMatrix: bias"); return -1;}
    if(l->weigthMatrix->mainSize != from->dcontext->deltaBuffer->mainSize) {
        DL_BADVALUE(1, "different main sizes: weightMatrix: [%d] deltaBuffer: [%d]", l->weigthMatrix->mainSize, from->dcontext->deltaBuffer->mainSize);
        return -1;
    }
    if(l->weigthMatrix->biasSize != from->dcontext->deltaBuffer->biasSize) {
        DL_BADVALUE(1, "different bias sizes: weightMatrix: [%d] deltaBuffer: [%d]", l->weigthMatrix->biasSize, from->dcontext->deltaBuffer->biasSize);
        return -1;
    }
#endif
    if(l->dcontext->use_support_buffer)
    {
        /*
        FOR_VALUE(from->dcontext->w_size, i)
        {
            l->weight[i] += from->dcontext->delta_w[i];
        }
        FOR_VALUE(from->dcontext->b_size, i)
        {
            l->bias[i] += from->dcontext->delta_b[i];
        }
        */

        FOR_VALUE(from->dcontext->deltaBuffer->mainSize, i){
            l->weigthMatrix->main[i] += from->dcontext->deltaBuffer->main[i];
        }
        FOR_VALUE(l->dcontext->deltaBuffer->biasSize, i){
            l->weigthMatrix->bias[i] += from->dcontext->deltaBuffer->bias[i];
        }
    }

    return 0;
}
int DSynapse::layer_copy_params(DSynapse::LAYER_P to, DSynapse::LAYER_P from)
{
    if(to == NULL) {DL_BADPOINTER(1, " <to> layer"); return -1;}
    if(from == NULL) {DL_BADPOINTER(1, " <from> layer"); return -1;}
    to->act = from->act;
    to->der_act = from->der_act;
    to->learn_rate = from->learn_rate;
    to->use_act_out = from->use_act_out;
    to->back_prop = from->back_prop;
    free_descent_context(to->dcontext);
    if(  (to->dcontext = copy_descent_context(from->dcontext)) == NULL  )
    {
        DL_BADALLOC(1, "dcontext");
        return -1;
    }

    return 0;
}
int DSynapse::layer_copy_weights(DSynapse::LAYER_P dst, DSynapse::LAYER_P src)
{
#ifdef LAYER_USE_SAVE_CALL
    if(dst == NULL) {DL_BADPOINTER(1, "dst layer"); return -1;}
    if(src == NULL) {DL_BADPOINTER(1, "src layer"); return -1;}
#endif
    /*
    if(dst->weight && src->weight)
        copy_mem(dst->weight, src->weight, src->w_size);
    if(dst->bias && src->bias)
        copy_mem(dst->bias, src->bias, src->b_size);
    */

    return wm_copy_weights_value(dst->weigthMatrix, src->weigthMatrix, WM_BOTH);
}
int DSynapse::layer_set_opt_randw(DSynapse::LAYER_P l, DSynapse::NET_P n)
{
    if(l == NULL) {DL_BADPOINTER(1, "layer");return -1;}
    if(n == NULL) {DL_BADPOINTER(1, "net"); return -1;}
    if(l->next == NULL) {DL_BADPOINTER(1, "next"); return -1;}

    nvt bot, top;
    n->weightInitRangeCallback(l->size, l->next->size, &bot, &top);
    return wm_set_rand(l->weigthMatrix, WM_BOTH, bot, top);
}
int DSynapse::layer_set_ranged_randw(DSynapse::LAYER_P l, DSynapse::NET_P n)
{
    if(l == NULL) {DL_BADPOINTER(1, "layer");return -1;}
    if(n == NULL) {DL_BADPOINTER(1, "net"); return -1;}
    return wm_set_rand(l->weigthMatrix, WM_BOTH, n->rand_weights_bot_value, n->rand_weights_top_value);
}
/*
int DSynapse::layer_set_bpf(DSynapse::LAYER_P l, DSynapse::NET_P n)
{
    if(l == NULL) {DL_BADPOINTER(1, "layer"); return -1;}
    if(n == NULL) {DL_BADPOINTER(1, "net"); return -1;}
    l->back_prop = n->back_prop;
    return 0;
}
int DSynapse::layer_set_actf(DSynapse::LAYER_P l, DSynapse::NET_P n)
{
    if(l == NULL) {DL_BADPOINTER(1, "layer"); return -1;}
    if(n == NULL) {DL_BADPOINTER(1, "net"); return -1;}

//    std::cout << D_FUNC_NAME  << " " << (void*)n->act << std::endl;
    l->act = n->act;
    l->der_act = n->der_act;
    return 0;
}
int DSynapse::layer_use_act_out(DSynapse::LAYER_P l, DSynapse::NET_P n)
{
    if(l == NULL) {DL_BADPOINTER(1, "layer"); return -1;}
    if(n == NULL) {DL_BADPOINTER(1, "net"); return -1;}
    l->use_act_out = n->use_act_out;
    return 0;
}
int DSynapse::layer_set_lr(DSynapse::LAYER_P l, DSynapse::NET_P n)
{
    if(l == NULL) {DL_BADPOINTER(1, "layer"); return -1;}
    if(n == NULL) {DL_BADPOINTER(1, "net"); return -1;}
    l->learn_rate = n->learn_rate;
    return 0;
}
*/
int DSynapse::layer_use_delta_buffers(DSynapse::LAYER_P l, DSynapse::NET_P n)
{
    if(n == NULL) {DL_BADPOINTER(1, "net"); return -1;}
    return layer_use_delta_buffers(l, n->use_support_buffer, n->use_main_buffer);
}
int DSynapse::layer_set_ranged_randw(DSynapse::LAYER_P l, DSynapse::nvt bot, DSynapse::nvt top)
{
    if(l == NULL) {DL_BADPOINTER(1, "layer");return -1;}
    if(l->weigthMatrix == NULL) {DL_BADPOINTER(1, "weightMatrix"); return -1;}
    return wm_set_rand(l->weigthMatrix, WM_BOTH, bot, top);
}
int DSynapse::layer_use_delta_buffers(DSynapse::LAYER_P l, int support, int main)
{
    if(l == NULL) {DL_BADPOINTER(1, "layer"); return -1;}
    l->dcontext->use_support_buffer = support ? 1 : 0;
    l->dcontext->use_main_buffer = main ? 1 : 0;
    return 0;
}
int DSynapse::layer_connect(DSynapse::LAYER_P l, DSynapse::LAYER_P next)
{
    if(l) l->next = next;
    if(next) next->prev = l;
    return 0;
}
int DSynapse::layer_set_actf(DSynapse::LAYER_P l, DSynapse::ACTF a, DSynapse::ACTF d)
{
    if(l == NULL) {DL_BADPOINTER(1, "layer"); return -1;}
    l->act = a;
    l->der_act = d;
    return 0;
}
int DSynapse::layer_set_actf(DSynapse::LAYER_P l, DSynapse::ACTIVATION a)
{
    if(l == NULL) {DL_BADPOINTER(1, "layer"); return -1;}
    switch (a)
    {
        case ACT_SIGMOID: l->act = act_sigmoid; l->der_act = der_act_sigmoid; break;
        case ACT_GTAN: l->act = act_gtan; l->der_act = der_act_gtan; break;
        case ACT_RELU: l->act = act_ReLU; l->der_act = der_act_ReLU; break;
        case ACT_LINE: l->act = act_LINE; l->der_act = der_act_LINE; break;
        case ACT_EMPTY: l->act = act_empty; l->der_act = der_act_empty; break;
        case ACT_DEBUG: l->act = act_debug; l->der_act = der_act_debug; break;
        default: DL_ERROR(1, "Undefined Activation Type: [%d]", (int)a); return -1;
    }
    return 0;
}

int DSynapse::layer_set_actf(DSynapse::LAYER_P l, int activationIndex)
{
    return layer_set_actf(l, actt_get_lable_by_index(activationIndex));
}
int DSynapse::layer_set_lr(DSynapse::LAYER_P l, DSynapse::nvt lr)
{
    if(l == NULL) {DL_BADPOINTER(1, "layer"); return -1;}
    l->learn_rate = lr;
    return 0;
}
int DSynapse::layer_increase_lr(DSynapse::LAYER_P l, DSynapse::nvt value)
{
    if(l == NULL) {DL_BADPOINTER(1, "layer"); return -1;}
    l->learn_rate += value;
    return 0;
}
int DSynapse::layer_decrease_lr(DSynapse::LAYER_P l, DSynapse::nvt value)
{
    if(l == NULL) {DL_BADPOINTER(1, "layer"); return -1;}
    l->learn_rate -= value;
    return 0;
}
int DSynapse::layer_set_bpf(DSynapse::LAYER_P l, DSynapse::BPF f)
{
    if(l == NULL) {DL_BADPOINTER(1, "layer"); return -1;}
    l->back_prop = f;
    return 0;
}
int DSynapse::layer_use_act_out(DSynapse::LAYER_P l, int state)
{
    if(l == NULL) {DL_BADPOINTER(1, "layer"); return -1;}
    l->use_act_out = state;
    return 0;
}
int DSynapse::layer_alloc_mask(DSynapse::LAYER_P l)
{
    if(l == NULL) {DL_BADPOINTER(1, "layer"); return -1;}
    l->mask = reget_zmem(l->mask, l->size);
    return 0;
}
int DSynapse::generate_mask_dropout(DSynapse::LAYER_P l)
{
    if(l == NULL) {DL_BADPOINTER(1, "layer"); return -1;}
    if(l->mask == NULL) layer_alloc_mask(l);
    FOR_VALUE(l->size, i)
    {
        l->mask[i] = ( global_rand_callback() > l->dropoutRate ) ? 1 : 0;
    }

    return 0;
}
int DSynapse::generate_mask_random(DSynapse::LAYER_P l)
{
    if(l == NULL) {DL_BADPOINTER(1, "layer"); return -1;}
    if(l->mask == NULL) layer_alloc_mask(l);
    FOR_VALUE(l->size, i)
    {
        l->mask[i] = ( global_rand_callback() > 0.5 ) ? 1 : 0;
    }

    return 0;
}
int DSynapse::generate_mask_debug(DSynapse::LAYER_P l)
{
    if(l == NULL) {DL_BADPOINTER(1, "layer"); return -1;}
    /*
     * debug code
    */
    return 0;
}
int DSynapse::generate_mask_enable_all(DSynapse::LAYER_P l)
{
    if(l == NULL) {DL_BADPOINTER(1, "layer"); return -1;}
    if(l->mask == NULL) layer_alloc_mask(l);
    FOR_VALUE(l->size, i)
    {
        l->mask[i] = 1;
    }
    return 0;
}
int DSynapse::generate_mask_disable_all(DSynapse::LAYER_P l)
{
    if(l == NULL) {DL_BADPOINTER(1, "layer"); return -1;}
    if(l->mask == NULL) layer_alloc_mask(l);
    set_mem(l->mask, 0, l->size);
    return 0;
}
int DSynapse::layer_set_dropout(DSynapse::LAYER_P l, float rate, DSynapse::DROPOUT t)
{
    if(l == NULL) {DL_BADPOINTER(1, "layer"); return -1;}
    if(rate < 0.0) rate = 0.0;
    else if(rate > 1.0) rate = 1.0;

    l->dropoutRate = rate;
    l->dropoutType = t;

    return 0;
}
int DSynapse::layer_generate_mask(DSynapse::LAYER_P l, DSynapse::LAYERMASK maskType)
{
    if(l == NULL) {DL_BADPOINTER(1, "layer"); return -1;}

    switch (maskType) {
    default:
    case LAYERMASK_EMPTY: break;
    case LAYERMASK_DEBUG: generate_mask_debug(l); break;
    case LAYERMASK_RANDOM: generate_mask_random(l); break;
    case LAYERMASK_DROPOUT: generate_mask_dropout(l); break;
    case LAYERMASK_ENABLEALL: generate_mask_enable_all(l); break;
    case LAYERMASK_DISABLEALL: generate_mask_disable_all(l); break;
    }
    return 0;
}
int DSynapse::layer_set_mask(DSynapse::LAYER_P l, const_mask_t mask, int maskSize)
{
    if(l == NULL) {DL_BADPOINTER(1, "layer"); return -1;}
    if(maskSize <= 0) {DL_BADVALUE(1, "maskSize: [%d]", maskSize); return -1;}
    if(l->mask == NULL) layer_alloc_mask(l);
    int s = maskSize > l->size ? l->size : maskSize;
    FOR_VALUE(s, i)
            l->mask[i] = mask[i] ? 1 : 0;
    return 0;
}
int DSynapse::layer_apply_mask(DSynapse::LAYER_P l)
{
    if(l == NULL) {DL_BADPOINTER(1, "layer"); return -1;}
    if(l->mask == NULL) {DL_BADPOINTER(1, "mask"); return -1;}
    FOR_VALUE(l->size, i)
    {
        l->input[i] *= l->mask[i];
        l->raw[i] *= l->mask[i];
        l->error[i] *= l->mask[i];
        l->der_input[i] *= l->mask[i];
    }
    return 0;
}

int DSynapse::_is_support(DSynapse::LAYER_DELTA_BUFFERS s)
{
    return (s == LAYER_USE_ONLY_SUPPORT_BUFFER || s == LAYER_USE_MAIN_AND_SUPPORT_BUFFER);
}
int DSynapse::_is_main(DSynapse::LAYER_DELTA_BUFFERS s)
{
    return (s == LAYER_USE_ONLY_MAIN_BUFFER || s == LAYER_USE_MAIN_AND_SUPPORT_BUFFER);
}














int DSynapse::layer_filter_activation(DSynapse::LAYER_P l)
{
#ifdef LAYER_FILTER_SAVE_CALL
    if(l == NULL) {DL_BADPOINTER(1, "layer"); return -1;}
    if(l->input == NULL) {DL_BADPOINTER(1, "input"); return -1;}
    if(l->raw == NULL) {DL_BADPOINTER(1, "raw"); return -1;}
    if(!l->act) {DL_BADPOINTER(1, "act"); return -1;}
    if(l->size <= 0) {DL_BADVALUE(1, "size: [%d]", l->size); return -1;}
#endif
    auto b = l->input;
    auto b_raw = l->raw;
    auto e = l->input + l->size;
    while(b!=e){
        *b++ = l->act(*b_raw++);
    }
    return 0;
}
int DSynapse::layer_filter_derivative_activation(DSynapse::LAYER_P l)
{
#ifdef LAYER_FILTER_SAVE_CALL
    if(l == NULL) {DL_BADPOINTER(1, "layer"); return -1;}
    if(l->der_input == NULL) {DL_BADPOINTER(1, "der_input"); return -1;}
    if(l->raw == NULL) {DL_BADPOINTER(1, "raw"); return -1;}
    if(!l->der_act) {DL_BADPOINTER(1, "der_act"); return -1;}
    if(l->size <= 0) {DL_BADVALUE(1, "size: [%d]", l->size); return -1;}
#endif
    auto b = l->der_input;
    auto b_raw = l->raw;
    auto e = l->der_input + l->size;
    while(b!=e){
        *b++ = l->der_act(*b_raw++);
    }
    return 0;
}
int DSynapse::layer_filter_symmetric_activation(DSynapse::LAYER_P l)
{
#ifdef LAYER_FILTER_SAVE_CALL
    if(l == NULL) {DL_BADPOINTER(1, "layer"); return -1;}
    if(l->input == NULL) {DL_BADPOINTER(1, "input"); return -1;}
    if(l->der_input == NULL) {DL_BADPOINTER(1, "der_input"); return -1;}
    if(l->raw == NULL) {DL_BADPOINTER(1, "raw"); return -1;}
    if(!l->act) {DL_BADPOINTER(1, "act"); return -1;}
    if(!l->der_act) {DL_BADPOINTER(1, "der_act"); return -1;}
    if(l->size <= 0) {DL_BADVALUE(1, "size: [%d]", l->size); return -1;}
#endif
    auto b_in = l->input;
    auto b_der = l->der_input;
    auto b_raw = l->raw;
    auto e = l->input + l->size;
    while(b_in!=e){
        *b_in++ = l->act(*b_raw);
        *b_der++ = l->der_act(*b_raw++);
    }
    return 0;
}
int DSynapse::layer_filter_dropout_forward_scaledown(DSynapse::LAYER_P l)
{
#ifdef LAYER_FILTER_SAVE_CALL
    if(l == NULL) {DL_BADPOINTER(1, "layer"); return -1;}
    if(l->input == NULL) {DL_BADPOINTER(1, "input"); return -1;}
    if(l->size <= 0) {DL_BADVALUE(1, "size: [%d]", l->size); return -1;}
#endif
    auto b = l->input;
    auto e = l->input + l->size;
    while(b!=e){
        *b++ *= l->dropoutRate; //survive probapility
    }
    return 0;
}
int DSynapse::layer_filter_dropout_inverted_scaledown(DSynapse::LAYER_P l)
{
#ifdef LAYER_FILTER_SAVE_CALL
    if(l == NULL) {DL_BADPOINTER(1, "layer"); return -1;}
    if(l->input == NULL) {DL_BADPOINTER(1, "input"); return -1;}
    if(l->size <= 0) {DL_BADVALUE(1, "size: [%d]", l->size); return -1;}
#endif
    auto b = l->input;
    auto e = l->input + l->size;
    while(b!=e){
        *b++ *= (1.0 / l->dropoutRate); // (1 / survive probapility)
    }
    return 0;
}
int DSynapse::layer_filter_softmax(DSynapse::LAYER_P l)
{
#ifdef LAYER_FILTER_SAVE_CALL
    if(l == NULL) {DL_BADPOINTER(1, "layer"); return -1;}
    if(l->input == NULL) {DL_BADPOINTER(1, "input"); return -1;}
    if(l->size <= 0) {DL_BADVALUE(1, "size: [%d]", l->size); return -1;}
#endif
    auto b = l->input;
    auto e = l->input + l->size;
    nvt s = 0.0;
    while(b!=e){
        *b = exp(*b);
        s += *b++;
    }
    b = l->input;
    while(b!=e){
        *b++ /= s;
    }

    return 0;
}
int DSynapse::layer_filter_softmax_activation(DSynapse::LAYER_P l)
{
#ifdef LAYER_FILTER_SAVE_CALL
    if(l == NULL) {DL_BADPOINTER(1, "layer"); return -1;}
    if(l->input == NULL) {DL_BADPOINTER(1, "input"); return -1;}
    if(l->der_input == NULL) {DL_BADPOINTER(1, "der_input"); return -1;}
    if(l->size <= 0) {DL_BADVALUE(1, "size: [%d]", l->size); return -1;}
#endif
    auto b = l->input;
    auto e = l->input + l->size;
    nvt s = 0.0;
    while(b!=e){
        *b = exp(*b);
        s += *b++;
    }
    b = l->input;
    auto b_der = l->der_input;
    while(b!=e){
        *b /= s;
        *b_der = *b * (1.0 - *b);
        ++b; ++b_der;
    }

    return 0;
}
int DSynapse::layer_filter_maxout(DSynapse::LAYER_P l)
{

}
int DSynapse::layer_filter_max_blur(DSynapse::LAYER_P l)
{
#ifdef LAYER_FILTER_SAVE_CALL
    if(l == NULL) {DL_BADPOINTER(1, "layer"); return -1;}
    if(l->input == NULL) {DL_BADPOINTER(1, "input"); return -1;}
    if(l->size <= 0) {DL_BADVALUE(1, "size: [%d]", l->size); return -1;}
#endif
    auto b = l->input;
    auto e = l->input + l->size;
    nvt max = *b++;
    while(b!=e){
        if(*b > max)
            max = *b;
        ++b;
    }
    b = l->input;
    while(b!=e){
        *b++ = max;
    }

    return 0;
}
int DSynapse::layer_filter_mean_blur(DSynapse::LAYER_P l)
{
#ifdef LAYER_FILTER_SAVE_CALL
    if(l == NULL) {DL_BADPOINTER(1, "layer"); return -1;}
    if(l->input == NULL) {DL_BADPOINTER(1, "input"); return -1;}
    if(l->size <= 0) {DL_BADVALUE(1, "size: [%d]", l->size); return -1;}
#endif
    auto b = l->input;
    auto e = l->input + l->size;
    nvt s = 0.0;
    while(b!=e){
        s += *b++;
    }
    b = l->input;
    s /= l->size;
    while(b!=e){
        *b++ = s;
    }

    return 0;
}
int DSynapse::layer_filter_exp_blur(DSynapse::LAYER_P l)
{
    return layer_filter_softmax(l);
}











