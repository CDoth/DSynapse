#include "net.h"
#include "daran.h"
#include <cmath>





int __get_next_data_default(DSynapse::data_line _input, DSynapse::data_line _target, const void *opaqueData, int iteration)
{
    if(
               _input == NULL
            || _target == NULL
            || opaqueData == NULL
            || iteration < 0)
        return -1;

    const DSynapse::net *_n = reinterpret_cast<const DSynapse::net*>(opaqueData);
    DSynapse::data_line __data_input = _n->trainInput;
    DSynapse::data_line __data_target = _n->trainTarget;
    if(__data_input && __data_target)
    {
        copy_mem(_input,
                 __data_input + (iteration * _n->input_l->size),
                 _n->input_l->size);
        copy_mem(_target,
                 __data_target + (iteration * _n->output_l->size),
                 _n->output_l->size);
    }
    else
        return -1;
    return 0;
}
int __weight_init_range_default(int currentLayerSize, int nextLayerSize, DSynapse::nvt *bot, DSynapse::nvt *top)
{
    if(bot == NULL || top == NULL)
        return -1;
    if(currentLayerSize <= 0 || nextLayerSize <= 0)
        return -1;

    *top = nextLayerSize / currentLayerSize;
    *bot = -*top;
    return 0;
}
DSynapse::NET_P DSynapse::alloc_net()
{
    NET_P n = NULL;
    if(  (n = get_zmem<net>(1)) == NULL  )
    {
        DL_BADALLOC(1, "net");
        return NULL;
    }
    n->loadSample = __get_next_data_default;
    n->weightInitRangeCallback = __weight_init_range_default;
    return n;
}
DSynapse::LAYER_P net_add_layer_copy(DSynapse::NET_P n, DSynapse::LAYER_P copy_this, int replace_wm)
{
    using namespace DSynapse;
    if(n == NULL) {DL_BADPOINTER(1, "net"); return NULL;}
    LAYER_P l = NULL;
    if(  (l = layer_copy(copy_this, replace_wm)) == NULL  )
    {
        DL_BADALLOC(1, "layer");
        goto fail;
    }
    if(n->size)
    {
        if(n->set == NULL) {DL_BADPOINTER(1, "set");goto fail;}
        LAYER_P prev = n->set[n->size - 1];
        if( layer_connect(prev, l) < 0 ) {DL_FUNCFAIL(1, "layer_connect");goto fail;}
        if( replace_wm == 0 )
        {
            if( layer_alloc_wm(prev) < 0 ) {DL_FUNCFAIL(1, "layer_alloc_weight");goto fail;}
        }
        if( layer_alloc_derinput(l) < 0 ) {DL_FUNCFAIL(1, "layer_alloc_derinput");goto fail;}
    }
    else
    {
        n->in = l->input;
        n->input_l = l;
    }
    if(  (n->set = reget_mem(n->set, n->size+1)) == NULL  ) {DL_BADALLOC(1, "Layers set");goto fail;}
    if(  (n->target = reget_zmem(n->target, l->size)) == NULL  ) {DL_BADALLOC(1, "Target array");goto fail;}
    n->set[n->size++] = l;
    n->out = l->input;
    n->error = l->error;
    n->output_l = l;

    return l;
fail:
    free_layer(l);
    return NULL;
}
DSynapse::NET_P DSynapse::net_copy(DSynapse::NET_P copy_this_net, int replace_wm)
{
    NET_P dst = NULL;
    NET_P src = copy_this_net;

    dst = alloc_net();

    LAYER_P l = *src->set;
    while(l)
    {
        net_add_layer_copy(dst, l, replace_wm);
        l = l->next;
    }
    if(replace_wm == 0)
        net_copy_weights(dst, src);

    if(replace_wm)
    {
        NET_P parentNet = src->baseNet ? src->baseNet : src;
        src->baseNet = parentNet;
        dst->baseNet = parentNet;
        ++parentNet->refCount;
    }

    return dst;
}
DSynapse::NET_P DSynapse::net_shared_copy(DSynapse::NET_P copy_this_net)
{
    return net_copy(copy_this_net, 1);
}
DSynapse::NET_P DSynapse::net_unique_copy(DSynapse::NET_P copy_this_net)
{
    return net_copy(copy_this_net, 0);
}
int DSynapse::net_alloc_wms(DSynapse::NET_P n) {
    if(n == NULL) {DL_BADPOINTER(1, "net"); return -1;}

    if( net_run_conn_layers(n, layer_alloc_wm) < 0 ) {
        DL_FUNCFAIL(1, "net_run_layers");
        return -1;
    }

    return 0;
}
int DSynapse::net_alloc_derinp(DSynapse::NET_P n) {
    if(n == NULL) {DL_BADPOINTER(1, "net"); return -1;}

    if( net_run_layers(n, layer_alloc_derinput) < 0 ) {
        DL_FUNCFAIL(1, "net_run_layers");
        return -1;
    }

    return 0;
}
int DSynapse::net_get_size(DSynapse::NET_P n)
{
    if(n == NULL) {DL_BADPOINTER(1, "net"); return -1;}
    return n->size;
}
int DSynapse::net_detach(DSynapse::NET_P n)
{
    if(n->baseNet && n->baseNet->refCount > 0)
    {
        if( --n->baseNet->refCount == 0 )
            n->baseNet->baseNet = NULL;
        n->baseNet = NULL;
        n->refCount = 0;
        if(n->set == NULL) {DL_BADPOINTER(1, "set"); return -1;}
        FOR_VALUE(n->size, i)
        {
            /*
            n->set[i]->weight = NULL;
            n->set[i]->bias = NULL;
            */
            n->set[i]->weigthMatrix = NULL;
        }

        NET_P currentNet = n->next;
        NET_P prevNet = NULL;
        while(currentNet != n)
        {
            if(currentNet->next == n)
            {
                prevNet = currentNet;
            }
        }
        prevNet->next = n->next;
    }
    return 0;
}
int DSynapse::net_is_unique(DSynapse::NET_P n)
{
    return (n->baseNet == NULL);
}
int DSynapse::net_make_unique(DSynapse::NET_P n)
{
    if(n == NULL) {DL_BADPOINTER(1, "net"); return -1;}
    if(n->baseNet && n->baseNet->refCount > 0)
    {
        if( net_detach(n) < 0 ) {DL_FUNCFAIL(1, "net_detach"); return -1;}
        if( net_run_layers(n, layer_alloc_wm) < 0) {DL_FUNCFAIL(1, "layer_alloc_weight"); return -1;}
        if( net_copy_weights(n, n->baseNet) < 0) {DL_FUNCFAIL(1, "net_copy_weights"); return -1;}
    }
    return 0;
}
void DSynapse::net_free(DSynapse::NET_P n)
{
    if(n)
    {
        if(n->size)
            net_run_layers(n, free_layer);
        free_mem(n->target);
        free_mem(n->input_qrate);
        free_mem(n->output_qrate);
        zero_mem(n, sizeof(net));
        free_mem(n);
    }
}
int DSynapse::net_clear_all(DSynapse::NET_P n)
{
    if(n == NULL) {DL_BADPOINTER(1, "net"); return -1;}
    net_run_layers(n, layer_clear_buffers);
    if(n->target && n->output_l && n->output_l->size > 0) zero_mem(n->target, n->output_l->size);
    if(n->trainInput && n->input_l && n->input_l->size > 0) zero_mem(n->trainInput, n->input_l->size);
    if(n->trainTarget && n->output_l && n->output_l->size > 0) zero_mem(n->trainTarget, n->output_l->size);
    if(n->input_qrate && n->input_l && n->input_l->size > 0) zero_mem(n->input_qrate, n->input_l->size);
    if(n->output_qrate && n->output_l && n->output_l->size > 0) zero_mem(n->output_qrate, n->output_l->size);

    return 0;
}
int DSynapse::net_run_layers(DSynapse::NET_P n, DSynapse::LAYER_ABSTRACT_ACTION a)
{
#ifdef NET_USE_SAVE_CALL
    if(n == NULL) {DL_BADPOINTER(1, "net");return -1;}
    if(n->set == NULL) {DL_BADPOINTER(1, "set"); return -1;}
#endif

    LAYER_P l = *n->set;
    LAYER_P next = NULL;
    while(l)
    {
        next = l->next;
#ifdef NET_USE_SAVE_CALL
        if( a(l) < 0 )
        {
            DL_FUNCFAIL(1, "LAYER_ABSTRACT_ACTION");
            return -1;
        }
#else
        a(l);
#endif
        l = next;
    }

    return 0;
}
int DSynapse::net_run_layers(DSynapse::NET_P n, DSynapse::LAYER_PARENT_ACTION a)
{
#ifdef NET_USE_SAVE_CALL
    if(n == NULL) {DL_BADPOINTER(1, "net");return -1;}
    if(n->set == NULL) {DL_BADPOINTER(1, "set"); return -1;}
#endif

    LAYER_P l = *n->set;
    LAYER_P next = NULL;
    while(l)
    {
        next = l->next;
#ifdef NET_USE_SAVE_CALL
        if( a(l,n) < 0 )
        {
            DL_FUNCFAIL(1, "LAYER_PARENT_ACTION");
            return -1;
        }
#else
        a(l,n);
#endif
        l = next;
    }

    return 0;
}
int DSynapse::net_run_layers(DSynapse::NET_P n1, DSynapse::NET_P n2, DSynapse::LAYER_DEPEND_ACTION a) {
#ifdef NET_USE_SAVE_CALL
    if(n1 == NULL){DL_BADPOINTER(1, "first net");return -1;}
    if(n2 == NULL){DL_BADPOINTER(1, "second net");return -1;}
    if(n1->set == NULL) {DL_BADPOINTER(1, "n1->set"); return -1;}
    if(n2->set == NULL) {DL_BADPOINTER(1, "n2->set"); return -1;}
#endif

    LAYER_P l1 = *n1->set;
    LAYER_P l2 = *n2->set;
    LAYER_P next1 = NULL;
    LAYER_P next2 = NULL;

    while(l1)
    {
        next1 = l1->next;
        next2 = l2->next;
#ifdef NET_USE_SAVE_CALL
        if( a(l1, l2) < 0 )
        {
            DL_FUNCFAIL(1, "LAYER_DEPEND_ACTION (1)");
            return -1;
        }
#else
        a(l1, l2);
#endif
        l1 = next1;
        l2 = next2;
    }

    return 0;
}
int DSynapse::net_run_layers(DSynapse::NET_P n, DSynapse::LAYER_P l, DSynapse::LAYER_DEPEND_ACTION a)
{
#ifdef NET_USE_SAVE_CALL
    if(n == NULL) {DL_BADPOINTER(1, "net");return -1;}
    if(n->set == NULL) {DL_BADPOINTER(1, "set"); return -1;}
#endif
    LAYER_P l1 = *n->set;
    LAYER_P next = NULL;
    while(l1)
    {
        next = l1->next;
#ifdef NET_USE_SAVE_CALL
        if( a(l1, l) < 0 )
        {
            DL_FUNCFAIL(1, "LAYER_DEPEND_ACTION (2)");
            return -1;
        }
#else
        a(l1, l);
#endif
        l1 = next;
    }

    return 0;
}
int DSynapse::net_run_layers(DSynapse::NET_P n, int arg, DSynapse::LAYER_SPECIAL_ACTION1 a)
{
#ifdef NET_USE_SAVE_CALL
    if(n == NULL) {DL_BADPOINTER(1, "net");return -1;}
    if(n->set == NULL) {DL_BADPOINTER(1, "set"); return -1;}
#endif
    LAYER_P l = *n->set;
    LAYER_P next = NULL;
    while(l)
    {
        next = l->next;
#ifdef NET_USE_SAVE_CALL
        if( a(l, arg) < 0 )
        {
            DL_FUNCFAIL(1, "LAYER_SPECIAL_ACTION1");
            return -1;
        }
#else
        a(l, arg);
#endif
        l = next;
    }

    return 0;
}
int DSynapse::net_run_layers(DSynapse::NET_P n, DSynapse::nvt arg, DSynapse::LAYER_SPECIAL_ACTION2 a)
{
#ifdef NET_USE_SAVE_CALL
    if(n == NULL) {DL_BADPOINTER(1, "net");return -1;}
    if(n->set == NULL) {DL_BADPOINTER(1, "set"); return -1;}
#endif
    LAYER_P l = *n->set;
    LAYER_P next = NULL;
    while(l)
    {
        next = l->next;
#ifdef NET_USE_SAVE_CALL
        if( a(l, arg) < 0 )
        {
            DL_FUNCFAIL(1, "LAYER_SPECIAL_ACTION1");
            return -1;
        }
#else
        a(l, arg);
#endif
        l = next;
    }

    return 0;
}
int DSynapse::net_run_conn_layers(DSynapse::NET_P n, DSynapse::LAYER_ABSTRACT_ACTION a) {

#ifdef NET_USE_SAVE_CALL
    if(n == NULL) {DL_BADPOINTER(1, "net");return -1;}
    if(n->set == NULL) {DL_BADPOINTER(1, "set"); return -1;}
#endif

    LAYER_P l = *n->set;
    LAYER_P next = NULL;
    while(l) {

        if( (next = l->next) != NULL ) {



#ifdef NET_USE_SAVE_CALL
            if( a(l) < 0 )
            {
                DL_FUNCFAIL(1, "LAYER_ABSTRACT_ACTION");
                return -1;
            }
#else
            a(l);
#endif
        }

        l = next;
    }

    return 0;
}
int DSynapse::net_run_conn_layers(DSynapse::NET_P n1, DSynapse::NET_P n2, DSynapse::LAYER_DEPEND_ACTION a) {

#ifdef NET_USE_SAVE_CALL
    if(n1 == NULL){DL_BADPOINTER(1, "first net");return -1;}
    if(n2 == NULL){DL_BADPOINTER(1, "second net");return -1;}
    if(n1->set == NULL) {DL_BADPOINTER(1, "n1->set"); return -1;}
    if(n2->set == NULL) {DL_BADPOINTER(1, "n2->set"); return -1;}
#endif

    LAYER_P l1 = *n1->set;
    LAYER_P l2 = *n2->set;
    LAYER_P next1 = NULL;
    LAYER_P next2 = NULL;

    while(l1)
    {
        next1 = l1->next;
        next2 = l2->next;
        if(next1 && next2) {
#ifdef NET_USE_SAVE_CALL
            if( a(l1, l2) < 0 )
            {
                DL_FUNCFAIL(1, "LAYER_DEPEND_ACTION (1)");
                return -1;
            }
#else
            a(l1, l2);
#endif
        }

        l1 = next1;
        l2 = next2;
    }

    return 0;
}
int DSynapse::net_set_randw(DSynapse::NET_P n)
{
    return net_run_conn_layers(n, layer_set_randw);
}
int DSynapse::net_set_opt_randw(DSynapse::NET_P n)
{
    return net_run_layers(n, layer_set_opt_randw);
}
int DSynapse::net_set_ranged_randw(DSynapse::NET_P n)
{
    return net_run_layers(n, layer_set_ranged_randw);
}
int DSynapse::net_set_ranged_randw(DSynapse::NET_P n, DSynapse::nvt bot, DSynapse::nvt top)
{
    if(n == NULL) {DL_BADPOINTER(1, "net"); return -1;}
    n->rand_weights_bot_value = bot;
    n->rand_weights_top_value = top;
    return net_run_layers(n, layer_set_ranged_randw);
}
int DSynapse::net_set_rand_input(DSynapse::NET_P n)
{
    return layer_set_rand_signal(n->input_l);
}
int DSynapse::net_set_actf(DSynapse::NET_P n, DSynapse::ACTIVATION a)
{
    if(n == NULL) {DL_BADPOINTER(1, "net"); return -1;}
//    switch (a)
//    {
//        case ACT_SIGMOID: n->act = act_sigmoid; n->der_act = der_act_sigmoid; break;
//        case ACT_GTAN: n->act = act_gtan; n->der_act = der_act_gtan; break;
//        case ACT_RELU: n->act = act_ReLU; n->der_act = der_act_ReLU; break;
//        case ACT_LINE: n->act = act_LINE; n->der_act = der_act_LINE; break;
//        case ACT_EMPTY: n->act = act_empty; n->der_act = der_act_empty; break;
//        case ACT_DEBUG: n->act = act_debug; n->der_act = der_act_debug; break;
//        default: DL_ERROR(1, "Undefined Activation Type: [%d]", (int)a); break;
//    }
//    return net_run_layers(n, layer_set_actf);

    return net_run_layers(n, actt_get_index_by_lable(a), layer_set_actf);
}
int DSynapse::net_set_lr(DSynapse::NET_P n, DSynapse::nvt lr)
{
    if(n == NULL) {DL_BADPOINTER(1, "net"); return -1;}
//    n->learn_rate = lr;
    return net_run_layers(n, lr, layer_set_lr);
}
int DSynapse::net_set_input_qrate(DSynapse::NET_P n, DSynapse::data_line input)
{
    if(n == NULL) {DL_BADPOINTER(1, "net"); return -1;}
    if(input == NULL) {DL_BADPOINTER(1, "input arg"); return -1;}
    if(n->input_l == NULL) {DL_BADPOINTER(1, "input layer"); return -1;}

    if(n->input_qrate == NULL)
    {
        if(  (n->input_qrate = alloc_buffer(n->input_l->size)) == NULL  )
        {
            DL_BADALLOC(1, "input_qrate");
            return -1;
        }
    }
    copy_mem(n->input_qrate, input, n->input_l->size);
    return 0;
}
int DSynapse::net_set_output_qrate(DSynapse::NET_P n, DSynapse::data_line output)
{
    if(n == NULL) {DL_BADPOINTER(1, "net"); return -1;}
    if(output == NULL) {DL_BADPOINTER(1, "output arg"); return -1;}
    if(n->output_l == NULL) {DL_BADPOINTER(1, "output layer"); return -1;}

    if(n->output_qrate == NULL)
    {
        if(  (n->output_qrate = alloc_buffer(n->output_l->size)) == NULL  )
        {
            DL_BADALLOC(1, "output_qrate");
            return -1;
        }
    }
    copy_mem(n->output_qrate, output, n->output_l->size);
    return 0;
}
int DSynapse::net_fill_input_qrate(DSynapse::NET_P n, DSynapse::nvt qr)
{
    if(n == NULL) {DL_BADPOINTER(1, "net"); return -1;}
    if(n->input_l == NULL) {DL_BADPOINTER(1, "input layer"); return -1;}
    if(n->input_qrate == NULL)
    {
        if(  (n->input_qrate = alloc_buffer(n->input_l->size)) == NULL  )
        {
            DL_BADALLOC(1, "input_qrate");
            return -1;
        }
    }
    FOR_VALUE(n->input_l->size, i)
    {
        n->input_qrate[i] = qr;
    }
    return 0;
}
int DSynapse::net_fill_output_qrate(DSynapse::NET_P n, DSynapse::nvt qr)
{
    if(n == NULL) {DL_BADPOINTER(1, "net"); return -1;}
    if(n->output_l == NULL) {DL_BADPOINTER(1, "output layer"); return -1;}
    if(n->output_qrate == NULL)
    {
        if(  (n->output_qrate = alloc_buffer(n->output_l->size)) == NULL  )
        {
            DL_BADALLOC(1, "output_qrate");
            return -1;
        }
    }
    FOR_VALUE(n->output_l->size, i)
    {
        n->output_qrate[i] = qr;
    }
    return 0;
}
int DSynapse::net_set_epoc_size(DSynapse::NET_P n, int esize)
{
    if(n == NULL) {DL_BADPOINTER(1, "net"); return -1;}
    n->EpocSize = esize;
    return 0;
}
int DSynapse::net_set_bpf(DSynapse::NET_P n, DSynapse::BPF f)
{
    if(n == NULL) {DL_BADPOINTER(1, "net"); return -1;}
    n->back_prop = f;
//    return net_run_layers(n, layer_set_bpf);
    return 0;
}
int DSynapse::net_set_bpf(DSynapse::NET_P n, DSynapse::BACKPROP_FUNCTION bpf)
{
    if(n == NULL) {DL_BADPOINTER(1, "net"); return -1;}
    switch (bpf)
    {
        case BP_DEFAULT: n->back_prop = back_propagation; break;
        default: DL_ERROR(1, "Undefined back prop callback"); return -1;
    }
    return 0;
}
int DSynapse::net_set_loss_function(DSynapse::NET_P n, DSynapse::LOSS_FUNCTION_CALLBACK lf)
{
    if(n == NULL) {DL_BADPOINTER(1, "net"); return -1;}
    n->lossFunction = lf;
    return 0;
}
int DSynapse::net_set_options(DSynapse::NET_P n, DSynapse::nvt learn_rate, DSynapse::ACTIVATION actf, DSynapse::BACKPROP_FUNCTION bpf, bool use_act_out,
                              DSynapse::nvt input_qr, DSynapse::nvt output_qr)
{
    if(n == NULL) {DL_BADPOINTER(1, "net"); return -1;}

    if( net_set_lr(n, learn_rate) < 0 ) {DL_FUNCFAIL(1, "net_set_lr"); return -1;}
    if( net_set_actf(n, actf) < 0 ) {DL_FUNCFAIL(1, "net_set_actf"); return -1;}
    if( net_set_bpf(n, bpf) < 0 ) {DL_FUNCFAIL(1, "net_set_bpf"); return -1;}
    if( net_use_act_out(n, use_act_out) < 0) {DL_FUNCFAIL(1, "net_use_act_out"); return -1;}
    if( net_fill_input_qrate(n, input_qr) < 0 ) {DL_FUNCFAIL(1, "net_fill_input_qrate"); return -1;}
    if( net_fill_output_qrate(n, output_qr) < 0 ) {DL_FUNCFAIL(1, "net_fill_output_qrate"); return -1;}

    return 0;
}
int DSynapse::net_set_options(DSynapse::NET_P n, DSynapse::nvt learn_rate, DSynapse::ACTIVATION actf, DSynapse::BACKPROP_FUNCTION bpf, bool use_act_out,
                              DSynapse::data_line input_qr, DSynapse::data_line output_qr)
{
    if(n == NULL) {DL_BADPOINTER(1, "net"); return -1;}

    if( net_set_lr(n, learn_rate) < 0 ) {DL_FUNCFAIL(1, "net_set_lr"); return -1;}
    if( net_set_actf(n, actf) < 0 ) {DL_FUNCFAIL(1, "net_set_actf"); return -1;}
    if( net_set_bpf(n, bpf) < 0 ) {DL_FUNCFAIL(1, "net_set_bpf"); return -1;}
    if( net_use_act_out(n, use_act_out) < 0) {DL_FUNCFAIL(1, "net_use_act_out"); return -1;}
    if( net_set_input_qrate(n, input_qr) < 0 ) {DL_FUNCFAIL(1, "net_set_input_qrate"); return -1;}
    if( net_set_output_qrate(n, output_qr) < 0 ) {DL_FUNCFAIL(1, "net_set_output_qrate"); return -1;}

    return 0;
}
int DSynapse::net_set_lateflush_size(DSynapse::NET_P n, int size)
{
    if(n == NULL) {DL_BADPOINTER(1, "net"); return -1;}
    if(size < 0) {DL_BADVALUE(1, "size: [%d]", size); return -1;}
    n->delta_flush_freq = size;
    return 0;
}
int DSynapse::net_set_default_load(DSynapse::NET_P n, DSynapse::data_line input, DSynapse::data_line target)
{
    if(n == NULL) {DL_BADPOINTER(1, "net"); return -1;}

    if(input == NULL) {DL_BADPOINTER(1, "input"); return -1;}
    if(target == NULL) {DL_BADPOINTER(1, "target"); return -1;}

    n->trainInput = input;
    n->trainTarget = target;
    n->opaque = n;
    n->loadSample = __get_next_data_default;

    return 0;
}
int DSynapse::net_set_custom_load(DSynapse::NET_P n, DSynapse::LOAD_SAMPLE_CALLBACK callback, const void *data)
{
    if(n == NULL) {DL_BADPOINTER(1, "net"); return -1;}
    if(data == NULL) {DL_BADPOINTER(1, "data. Try to use net_set_global_load"); return -1;}
    n->opaque = data;
    n->loadSample = callback;
    return 0;
}
int DSynapse::net_set_global_load(DSynapse::NET_P n, LOAD_SAMPLE_CALLBACK callback)
{
    if(n == NULL) {DL_BADPOINTER(1, "net"); return -1;}
    n->opaque = NULL;
    n->loadSample = callback;
    return 0;
}
int DSynapse::net_set_export_support_data(DSynapse::NET_P n, const void *data)
{
    if(n == NULL) {DL_BADPOINTER(1, "net"); return -1;}
    n->exportOpaque = data;
    return 0;
}
int DSynapse::net_set_input(DSynapse::NET_P n, DSynapse::nvt value, int index)
{
    if(n == NULL) {DL_BADPOINTER(1, "net"); return -1;}
    if(n->input_l == NULL) {DL_BADPOINTER(1, "input layer"); return -1;}
    if(n->input_l->input == NULL) {DL_BADPOINTER(1, "input buffer"); return -1;}
    if(n->input_l->raw == NULL) {DL_BADPOINTER(1, "raw buffer"); return -1;}
    if(n->input_l->size <= 0) {DL_BADVALUE(1, "input layer size: [%d]", n->input_l->size); return -1;}
    if(index < 0 || index >= n->input_l->size) {DL_BADVALUE(1, "input index: [%d] (layer size: [%d])", index, n->input_l->size); return -1;}

    n->input_l->raw[index] = value;
    n->input_l->input[index] = value;
    return 0;
}
int DSynapse::net_get_output(DSynapse::NET_P n, nvt *value, int index)
{
    if(n == NULL) {DL_BADPOINTER(1, "net"); return -1;}
    if(n->output_l == NULL) {DL_BADPOINTER(1, "output layer"); return -1;}
    if(n->output_l->input == NULL) {DL_BADPOINTER(1, "output buffer"); return -1;}
    if(n->output_l->size <= 0) {DL_BADVALUE(1, "output layer size: [%d]", n->output_l->size); return -1;}
    if(index < 0 || index >= n->output_l->size) {DL_BADVALUE(1, "output index: [%d] (layer size: [%d]", index, n->output_l->size); return -1;}
    if(value == NULL) {DL_BADPOINTER(1, "value arg"); return -1;}

    *value = n->output_l->input[index];
    return 0;
}
int DSynapse::net_remove_neuron(DSynapse::NET_P n, int layerIndex, int npos)
{
    //unchecked code !!!

    /*

    LAYER_P l = net_get_layer(n, layerIndex);
    if(l == NULL) {DL_FUNCFAIL(1, "net_get_layer"); return -1;}
    if(npos < 0 || npos > l->w_size) {DL_BADVALUE(1, "wpos"); return -1;}
    if(l->size == 1)
        return net_remove_layer(n, layerIndex, 0);


    if(l->weight == NULL) {DL_BADPOINTER(1, "weight"); return -1;}
    if(l->prev == NULL) {DL_BADPOINTER(1, "prev"); return -1;}
    if(l->prev->weight == NULL) {DL_BADPOINTER(1, "prev weight"); return -1;}


    // move current layer weights
    auto batch_change_it = l->weight + npos;
    auto batch_src_it = l->weight;
    auto batch_dst_it = l->weight;
    const auto batch_src_end = l->weight + l->w_size;
    while(batch_src_it != batch_src_end)
    {
        memmove(batch_change_it, batch_change_it+1, sizeof(nvt) * (l->size - npos - 1));
        batch_change_it += l->size;

        memmove(batch_dst_it, batch_src_it, sizeof(nvt) * (l->size - 1));
        batch_src_it += l->size;
        batch_dst_it += (l->size-1);
    }

    // move prev layer weights
    auto prev_wm_src = l->prev->weight + ((npos+1) * l->prev->size);
    auto prev_wm_dst = l->prev->weight + (npos * l->prev->size);
    int prev_wm_size = sizeof(nvt) * (l->size - npos - 1);
    memmove(prev_wm_dst, prev_wm_src, prev_wm_size);

    // move prev layer bias
    memmove(l->prev->bias + npos, l->prev->bias + npos + 1, sizeof(nvt) * (l->size - npos - 1));


    layer_modify(l, l->size-1);
    layer_alloc_weight(l->prev);
    layer_alloc_weight(l);

    */

    return 0;
}
int DSynapse::net_remove_neuron_set(DSynapse::NET_P n, int layerIndex, int nstart, int nend)
{

    //unchecked code !!!

    /*

    if(nstart == nend)
        return net_remove_neuron(n, layerIndex, nstart);

    LAYER_P l = net_get_layer(n, layerIndex);
    if(l == NULL) {DL_FUNCFAIL(1, "net_get_layer"); return -1;}
    if(nstart > nend)
        std::swap(nstart, nend);
    if(nstart < 0 || nstart >= l->size) {DL_BADVALUE(1, "nstart: [%d] layer size: [%d]", nstart, l->size); return -1;}
    if(nend < 0 || nend >= l->size) {DL_BADVALUE(1, "nend: [%d] layer size: [%d]", nend, l->size); return -1;}

    int range = nend - nstart;
    auto batch_change_it = l->weight + nstart;
    auto batch_src_it = l->weight;
    auto batch_dst_it = l->weight;
    const auto batch_src_end = l->weight + l->w_size;
    while(batch_src_it != batch_src_end)
    {
        memmove(batch_change_it, batch_change_it + range, sizeof(nvt) * (l->size - nend - 1));
        batch_change_it += l->size;

        memmove(batch_dst_it, batch_src_it, sizeof(nvt) * (l->size - range));
        batch_src_it += l->size;
        batch_dst_it += (l->size - range);
    }

    auto prev_wm_src = l->prev->weight + (nend * l->prev->size);
    auto prev_wm_dst = l->prev->weight + (nstart * l->prev->size);
    int prev_wm_size = sizeof(nvt) * (l->size - nend - 1);
    memmove(prev_wm_dst, prev_wm_src, prev_wm_size);

    memmove(l->prev->bias + nstart, l->prev->bias + nend, sizeof(nvt) * (l->size - nend - 1));

    layer_modify(l, l->size - range);
    layer_alloc_weight(l->prev);
    layer_alloc_weight(l);

    */
    return 0;
}

int DSynapse::net_restore_out(DSynapse::NET_P n)
{
#ifdef NET_USE_SAVE_CALL
    if(n == NULL) {DL_BADPOINTER(1, "net"); return -1;}
    if(n->output_l == NULL) {DL_BADPOINTER(1, "output_l"); return -1;}
    if(n->output_l->input == NULL) {DL_BADPOINTER(1, "output_l->input"); return -1;}
    if(n->output_qrate == NULL) {DL_BADPOINTER(1, "output_qrate"); return -1;}
#endif
    FOR_VALUE(n->output_l->size, i)
    {
        n->output_l->input[i] *= n->output_qrate[i];
    }
    return 0;
}
int DSynapse::net_copy_weights(DSynapse::NET_P dst, DSynapse::NET_P src)
{
#ifdef NET_USE_SAVE_CALL
    if(dst == NULL) {DL_BADPOINTER(1, "dst net"); return -1;}
    if(src == NULL) {DL_BADPOINTER(1, "src net"); return -1;}
#endif
    if(dst->baseNet == NULL || dst->baseNet != src->baseNet)
        return net_run_conn_layers(dst, src, layer_copy_weights);
    return 0;
}
int DSynapse::net_load_sample(DSynapse::NET_P n, int index)
{
//    std::cout << "call: net_load_sample"
//              << std::endl;

#ifdef NET_USE_SAVE_CALL
    if(n == NULL) {DL_BADPOINTER(1, "net"); return -1;}
    if(n->loadSample == NULL) {DL_BADPOINTER(1, "loadSample"); return -1;}
    if(index < 0) {DL_BADVALUE(1, "index: [%d]", index); return -1;}
#endif

//    std::cout << "call: net_load_sample (checked)"
//              << std::endl;
    int r = n->loadSample(n->input_l->raw, n->target, n->opaque, index);

//    std::cout << "call: net_load_sample (loadSample): " << r
//              << std::endl;
    return r;
}
int DSynapse::net_export_error(DSynapse::NET_P n, int index)
{
#ifdef NET_USE_SAVE_CALL
    if(n == NULL) {DL_BADPOINTER(1, "net"); return -1;}
    if(!n->exportError) {DL_BADPOINTER(1, "exportError"); return -1;}
    if(index < 0) {DL_BADVALUE(1, "index: [%d]", index); return -1;}
#endif
    return n->exportError(n->input_l->error, n->exportOpaque, index);
}

DSynapse::LAYER_P DSynapse::net_add_layer(DSynapse::NET_P n, int layer_size, int saveMeta)
{
    if(n == NULL) {DL_BADPOINTER(1, "net"); return NULL;}
    LAYER_P l = NULL;
    if(  (l = alloc_layer(layer_size)) == NULL  )
    {
        DL_BADALLOC(1, "layer");
        goto fail;
    }
    if(n->size)
    {
        if(n->set == NULL) {DL_BADPOINTER(1, "set");goto fail;}
        LAYER_P prev = n->set[n->size - 1];
        if( layer_connect(prev, l) < 0 ) {DL_FUNCFAIL(1, "layer_connect");goto fail;}
//        if( layer_alloc_weight(prev) < 0 ) {DL_FUNCFAIL(1, "layer_alloc_weight");goto fail;}
//        if( layer_alloc_derinput(l) < 0 ) {DL_FUNCFAIL(1, "layer_alloc_derinput");goto fail;}
//        if(n->use_delta)
//        {
//            if( layer_alloc_delta(l) < 0 ) {DL_FUNCFAIL(1, "layer_alloc_delta");goto fail;}
//        }
    }
    else
    {
        n->in = l->input;
        n->input_l = l;
    }
    if(  (n->set = reget_mem(n->set, n->size+1)) == NULL  ) {DL_BADALLOC(1, "Layers set");goto fail;}
    if(  (n->target = reget_zmem(n->target, l->size)) == NULL  ) {DL_BADALLOC(1, "Target array");goto fail;}
    n->set[n->size++] = l;
    n->out = l->input;
    n->error = l->error;
    n->output_l = l;

//    std::cout << __func__ << ": size: " << layer_size << " layer: " << (void*)l << std::endl;
    /* //Meta
    if(saveMeta && n->baseNet)
    {
        NET_P currentNet = n->next;
        while(currentNet != n)
        {
            net_add_layer(currentNet, layer_size, 0);
            currentNet = currentNet->next;
        }
    }
    */

    std::cout << "add layer: " << l << "(" << l->size << ")" << std::endl;
    return l;
fail:
    free_layer(l);
    return NULL;
}
DSynapse::LAYER_P DSynapse::net_insert_layer(DSynapse::NET_P n, int layer_size, int insertAfter, int saveMeta)
{
    if(n == NULL) {DL_BADPOINTER(1, "net"); return NULL;}
    if(n->set == NULL) {DL_BADPOINTER(1, "set"); return NULL;}
    if(insertAfter < 0 || insertAfter >= n->size) {DL_BADVALUE(1, "insertAfter: [%d] net size: [%d]", insertAfter, n->size); return NULL;}
    if(insertAfter == n->size-1)
        return net_add_layer(n, layer_size, saveMeta);

    LAYER_P l = NULL;
    LAYER_P prev = n->set[insertAfter];
    if(  (l = alloc_layer(layer_size)) == NULL  )
    {
        DL_BADALLOC(1, "layer");
        goto fail;
    }
    if( layer_connect(prev, l) < 0 ) {DL_FUNCFAIL(1, "layer_connect");goto fail;}
    if(  (n->set = expand_mem(n->set, n->size++, 1, insertAfter)) == NULL ) {DL_BADALLOC(1, "set"); goto fail;}
    n->set[insertAfter] = l;
    return l;
fail:
    free_layer(l);
    return NULL;
}
int DSynapse::net_modify_layer(DSynapse::NET_P n, int layer_index, int layer_new_size)
{
    if(n == NULL) {DL_BADPOINTER(1, "net"); return NULL;}

    LAYER_P l = net_get_layer(n, layer_index);
    if(l)
    {
        if( layer_modify(l, layer_new_size) == NULL ) {DL_FUNCFAIL(1, "layer_modify"); return -1;}
        LAYER_P prev = l->prev;
        if(prev)
        {
            layer_alloc_connection_buffers(l);
        }
    }

    return 0;
}
int DSynapse::net_remove_layer(DSynapse::NET_P n, int layer_index, int saveMeta)
{
    if(n == NULL) {DL_BADPOINTER(1, "net"); return NULL;}
    if(layer_index < 0 || layer_index >= n->size) {DL_BADVALUE(1, "layer_index: [%d], net size: [%d]", layer_index, n->size); return NULL;}
    if(n->set == NULL) {DL_BADPOINTER(1, "set"); return NULL;}


    /* //Meta
    if(saveMeta == 0)
        net_detach(n);
    */

    LAYER_P l = net_get_layer(n, layer_index);
    std::cout << "remove layer: " << layer_index << ": " << l << "(" << l->size << ")" << std::endl;
//    if(l == NULL) {DL_BADPOINTER(1, "layer: [%d]", layer_index); return -1;}
//    LAYER_P prev = l->prev;
//    LAYER_P next = l->next;
//    layer_connect(prev, next);
//    if(prev)
//    {
//        if(next)
//            layer_alloc_weight(prev);
//        else
//            layer_free_weight(prev);
//    }
//    free_layer(l);
//    if(n->size)
    if(  (n->set = remove_mem(n->set, n->size, 1, layer_index)) == NULL  ) { /*DL_BADALLOC(1, "set, size: [%d]", n->size);*/}
    --n->size;


//    std::cout << __func__ << " set: " << n->set << " size: " << n->size << std::endl;
//    FOR_VALUE(n->size, i)
//    {
//        std::cout << i << ": Layer: " << n->set[i] << " (" << n->set[i]->size << ")" << std::endl;
//    }



    /* //Meta
    if(saveMeta && n->baseNet)
    {
        NET_P currentNet = n->next;
        while (currentNet != n)
        {
            LAYER_P l = net_get_layer(currentNet, layer_index);
            LAYER_P prev = l->prev;
            LAYER_P next = l->next;
            layer_connect(prev, next);
            free_layer(l);
            currentNet = currentNet->next;
        }
    }
    */

    return 0;
}
int DSynapse::net_remove_layer(DSynapse::NET_P n, DSynapse::LAYER_P l)
{
//    net_remove_layer(n, net_get_layer_index(n, l));
}
int DSynapse::net_get_layer_size(DSynapse::NET_P n, int layer_index)
{
    if(n == NULL) {DL_BADPOINTER(1, "net"); return -1;}
    if(layer_index < 0 || layer_index >= n->size) {DL_BADVALUE(1, "layer_index: [%d], net size: [%d]", layer_index, n->size); return -1;}
    if(n->set == NULL) {DL_BADPOINTER(1, "set"); return -1;}
    if(n->set[layer_index])
        return n->set[layer_index]->size;
    else
    {
        DL_BADPOINTER(1, "layer: [%d]", layer_index);
        return -1;
    }
}
int DSynapse::net_get_layer_index(DSynapse::NET_P n, DSynapse::LAYER_P l)
{
    if(n == NULL) {DL_BADPOINTER(1, "net"); return NULL;}
    if(n->set == NULL) {DL_BADPOINTER(1, "set"); return -1;}
    FOR_VALUE(n->size, i)
    {
        if(n->set[i] == l)
            return i;
    }
    return -1;
}
DSynapse::LAYER_P DSynapse::net_get_layer(DSynapse::NET_P n, int layer_index)
{
    if(n == NULL) {DL_BADPOINTER(1, "net"); return NULL;}
    if(layer_index < 0 || layer_index >= n->size) {DL_BADVALUE(1, "layer_index: [%d], net size: [%d]", layer_index, n->size); return NULL;}
    if(n->set == NULL) {DL_BADPOINTER(1, "set"); return NULL;}
    return n->set[layer_index];
}
DSynapse::LAYER_P DSynapse::net_get_input_layer(DSynapse::NET_P n)
{
   if(n == NULL) {DL_BADPOINTER(1, "net"); return NULL;}
   return n->input_l;
}
DSynapse::LAYER_P DSynapse::net_get_output_layer(DSynapse::NET_P n)
{
    if(n == NULL) {DL_BADPOINTER(1, "net"); return NULL;}
    return n->output_l;
}
int DSynapse::net_increase_lr(DSynapse::NET_P n, DSynapse::nvt step)
{
    if(n == NULL) {DL_BADPOINTER(1, "net"); return -1;}
    if(step)
    {
//        n->learn_rate += lr;
//        net_run_layers(n, layer_set_lr);

        net_run_layers(n, step, layer_increase_lr);
    }
    return 0;
}
int DSynapse::net_decrease_lr(DSynapse::NET_P n, DSynapse::nvt step)
{
    if(n == NULL) {DL_BADPOINTER(1, "net"); return -1;}
    if(step)
    {
//        n->learn_rate -= step;
//        net_run_layers(n, layer_set_lr);

        net_run_layers(n, step, layer_decrease_lr);
    }
    return 0;
}
int DSynapse::net_alloc_input_qrate_array(DSynapse::NET_P n)
{
    if(n == NULL) {DL_BADPOINTER(1, "net"); return -1;}
    if(n->input_qrate == NULL)
    {
        if(  (n->input_qrate = alloc_buffer(n->input_l->size)) == NULL  )
        {
            DL_BADALLOC(1, "input_qrate");
            return -1;
        }
    }
    return 0;
}
int DSynapse::net_alloc_output_qrate_array(DSynapse::NET_P n)
{
    if(n == NULL) {DL_BADPOINTER(1, "net"); return -1;}
    if(n->output_qrate == NULL)
    {
        if(  (n->output_qrate = alloc_buffer(n->output_l->size)) == NULL  )
        {
            DL_BADALLOC(1, "output_qrate");
            return -1;
        }
    }
    return 0;
}
int DSynapse::net_check(DSynapse::NET_P n)
{
    if(n == NULL) {DL_BADPOINTER(1, "net"); return -1;}
    if(n->set == NULL)
    {
        DL_BADPOINTER(1, "set");
        return -1;
    }
    if(n->size <= 0)
    {
        DL_BADVALUE(1, "size");
        return -1;
    }
    if(n->input_l == NULL)
    {
        DL_BADPOINTER(1, "input_l");
        return -1;
    }
    if(n->output_l == NULL)
    {
        DL_BADPOINTER(1, "output_l");
        return -1;
    }
    if(n->in == NULL)
    {
        DL_BADPOINTER(1, "in");
        return -1;
    }
    if(n->out == NULL)
    {
        DL_BADPOINTER(1, "out");
        return -1;
    }
    if(n->error == NULL)
    {
        DL_BADPOINTER(1, "error");
        return -1;
    }
    if(n->target == NULL)
    {
        DL_BADPOINTER(1, "target");
        return -1;
    }
    if(n->EpocSize <= 0)
    {
        DL_BADVALUE(1, "EpocSize");
    }
    /*
    if(!n->act)
    {
        DL_BADPOINTER(1, "act");
        return -1;
    }
    if(!n->der_act)
    {
        DL_BADPOINTER(1, "der_act");
        return -1;
    }
    */
    if(!n->back_prop)
    {
        DL_BADPOINTER(1, "back_prop");
        return -1;
    }
    if(!n->loadSample)
    {
        DL_BADPOINTER(1, "loadSample");
        return -1;
    }
    if(n->set)
    {
        for(int i=0;i!=n->size;++i)
        {
            if( layer_check(n->set[i]) < 0 )
            {
                DL_ERROR(1, "bad layer: %p", n->set[i]);
                return -1;
            }
        }
    }

    return 0;
}

int DSynapse::forward_prop(DSynapse::NET_P n)
{
#ifdef NET_USE_SAVE_CALL
    if(n == NULL) {DL_BADPOINTER(1, "net");return -1;}
#endif
    LAYER_P current = *n->set;
    LAYER_P next = current->next;

#ifdef NET_USE_SAVE_CALL
    if(n->input_l == NULL) {DL_BADPOINTER(1, "input layer"); return -1;}
    if(n->input_l->raw == NULL) {DL_BADPOINTER(1, "raw array"); return -1;}
    if(n->input_l->input == NULL) {DL_BADPOINTER(1, "input array (1)"); return -1;}
#endif



//    std::cout << "forward_prop: noq input: ";
    if(n->input_qrate) {
        FOR_VALUE(n->input_l->size, i)
        {
//            std::cout << "[" << n->input_l->raw[i] << " q(" << n->input_qrate[i] << "): ";
            if(n->input_qrate[i]) {
                n->input_l->raw[i] /= n->input_qrate[i];
            }
//            std::cout << n->input_l->raw[i] << "] ";
        }
    }



    copy_mem(n->input_l->input, n->input_l->raw, n->input_l->size);





    while( next )
    {

//        std::cout << "forward_prop " << current << std::endl;

#ifdef NET_USE_SAVE_CALL
    if(next->input == NULL) {DL_BADPOINTER(1, "input array (2)"); return -1;}
    if(next->raw == NULL) {DL_BADPOINTER(1, "next->raw array"); return -1;}
    if(next->der_input == NULL) {DL_BADPOINTER(1, "next->der_input array"); return -1;}
    if(current->input == NULL) {DL_BADPOINTER(1, "current->input array"); return -1;}
    if(current->size <= 0) {DL_BADVALUE(1, "current->size: %d", current->size); return -1;}
    if(next->size <= 0) {DL_BADVALUE(1, "next->size: %d", next->size); return -1;}
    /*
    if(current->weight == NULL) {DL_BADPOINTER(1, "current->weight array"); return -1;}
    if(current->bias == NULL) {DL_BADPOINTER(1, "current->bias array"); return -1;}
    */
    if(current->weigthMatrix == NULL) {DL_BADPOINTER(1, "current: weightMatrix"); return -1;}
#endif

    /*
    if(n->learn_mode)
        forward_propagation_learn(next->input,
                            next->raw,
                            next->der_input,
                            current->input,
                            current->weight,
                            current->bias,
                            current->size,
                            next->size ,
                            current->act,
                            current->der_act);
    else
        forward_propagation(next->input,
                            current->input,
                            current->weight,
                            current->bias,
                            current->size,
                            next->size,
                            current->act);
    */
    if(n->learn_mode)
        forward_propagation_learn(next->input,
                            next->raw,
                            next->der_input,
                            current->input,
                            current->weigthMatrix->main,
                            current->weigthMatrix->bias,
                            current->size,
                            next->size ,
                            current->act,
                            current->der_act);
    else
        forward_propagation(next->input,
                            current->input,
                            current->weigthMatrix->main,
                            current->weigthMatrix->bias,
                            current->size,
                            next->size,
                            current->act);

        current = next;
        next = current->next;
    }



//    std::cout << " t: ";
//    FOR_VALUE(n->output_l->size, i) {
//        std::cout << n->target[i] << " ";
//    }
//    std::cout << " out: ";

//    FOR_VALUE(n->output_l->size, i) {
//        std::cout << n->out[i] << " ";
//    }
//    std::cout << std::endl;

    return 0;
}
int DSynapse::back_prop(DSynapse::NET_P n)
{
#ifdef NET_USE_SAVE_CALL
    if(n == NULL) {DL_BADPOINTER(1, "net");return -1;}
#endif
    LAYER_P current = n->output_l;
    if(current == nullptr) {
        DL_BADPOINTER(1, "output layer");
        return -1;
    }
    LAYER_P prev = current->prev;
#ifdef NET_USE_SAVE_CALL
    if(n->output_l == NULL) {DL_BADPOINTER(1, "output layer"); return -1;}
    if(n->target == NULL) {DL_BADPOINTER(1, "n->target"); return -1;}
    if(n->out == NULL) {DL_BADPOINTER(1, "n->out"); return -1;}
    if(n->error == NULL) {DL_BADPOINTER(1, "n->error"); return -1;}
#endif
    /*
    FOR_VALUE(n->output_l->size, i)
    {
        n->total_error += fabs(n->out[i] - n->target[i]);
        n->error[i] = (n->target[i] - n->out[i]) ;
    }
    */
    if(n->lossFunction == nullptr) {
        DL_BADPOINTER(1, "lossFunction");
        return -1;
    }
    FOR_VALUE(n->output_l->size, i) {

        if(n->output_qrate[i]) {
            n->target[i] /= n->output_qrate[i];
        }
    }

    n->lossFunction(n->out, n->target, n->error, &n->total_error, n->output_l->size);


//    std::cout << "back_prop:" << std::endl;
//    FOR_VALUE(n->output_l->size, i) {
//        std::cout
//                << " o: [" << n->out[i] << "]"
//                << " t: [" << n->target[i] << "]"
//                << " e: [" << n->error[i] << "]"
//                << " q: [" << n->output_qrate[i] << "]"
//                << std::endl
//                ;
//    }
//    std::cout << std::endl;

    while( prev )
    {
//        std::cout << "back_prop " << current << std::endl;

#ifdef NET_USE_SAVE_CALL
        if(current->error == NULL) {DL_BADPOINTER(1, "current->error"); return -1;}
        if(prev->error == NULL) {DL_BADPOINTER(1, "prev->error"); return -1;}
        if(prev->raw == NULL) {DL_BADPOINTER(1, "prev->raw"); return -1;}
        if(prev->input == NULL) {DL_BADPOINTER(1, "prev->input"); return -1;}
        if(current->der_input == NULL) {DL_BADPOINTER(1, "current->der_input"); return -1;}
        if(current->size <= 0) {DL_BADVALUE(1, "current->size: %d", current->size); return -1;}
        if(prev->size <= 0) {DL_BADVALUE(1, "prev->size: %d", prev->size); return -1;}

        /*
        if(prev->weight == NULL) {DL_BADPOINTER(1, "prev->weight"); return -1;}
        if(prev->bias == NULL) {DL_BADPOINTER(1, "prev->bias"); return -1;}
        */
        if(prev->weigthMatrix == NULL) {DL_BADPOINTER(1, "prev: weightMatrix"); return -1;}

#endif
        //dont estimate error for 1st layer (if no need for export)
        /*
        current->back_prop(current->error,
                           prev->error,
                           prev->weight,
                           prev->use_act_out ? prev->input : prev->raw,
                           current->der_input,
                           prev->bias,
                           prev->learn_rate,
                           current->size,
                           prev->size,
                           prev->dcontext
                           );
        */
        current->back_prop(current->error,
                           prev->error,
                           prev->weigthMatrix->main,
                           prev->use_act_out ? prev->input : prev->raw,
                           current->der_input,
                           prev->weigthMatrix->bias,
                           prev->learn_rate,
                           current->size,
                           prev->size,
                           prev->dcontext
                           );


        current = prev;
        prev = current->prev;
    }

    return 0;
}
int DSynapse::net_use_delta_buffers(DSynapse::NET_P n, DSynapse::LAYER_DELTA_BUFFERS s)
{
    if(n == NULL) {DL_BADPOINTER(1, "net");return -1;}
    n->use_support_buffer = _is_support(s) ? 1 : 0;
    n->use_main_buffer = _is_main(s) ? 1 : 0;
    if(n->use_support_buffer)
    {
        if( net_run_layers(n, layer_alloc_delta) < 0 )
        {
            DL_FUNCFAIL(1, "layer_alloc_delta");
            n->use_support_buffer = 0;
            n->use_main_buffer = 1;
            return -1;
        }
    }
    return net_run_layers(n, layer_use_delta_buffers);
}
int DSynapse::net_use_act_out(DSynapse::NET_P n, int state)
{
    if(n == NULL) {DL_BADPOINTER(1, "net");return -1;}
//    n->use_act_out = state > 0 ? 1 : 0;
    return net_run_layers(n, state, layer_use_act_out);
}
/*
int DSynapse::net_use_total_learn_rate(DSynapse::NET_P n, int state)
{
    if(n == NULL) {DL_BADPOINTER(1, "net");return -1;}
    n->use_total_learn_rate = state > 0 ? 1 : 0;
    return 0;
}
*/
int DSynapse::net_use_random_input(DSynapse::NET_P n, int state, DSynapse::nvt bot, DSynapse::nvt top)
{
    if(n == NULL) {DL_BADPOINTER(1, "net");return -1;}
    n->use_random_input = state > 0 ? 1 : 0;
    n->rand_input_bot_value = bot;
    n->rand_input_top_value = top;
    return 0;
}

int DSynapse::net_show_w(DSynapse::NET_P n)
{
    return net_run_layers(n, layer_show_w);
}
int DSynapse::net_show_delta(DSynapse::NET_P n)
{
    return net_run_layers(n, layer_show_delta);
}
int DSynapse::net_show_raw_out(DSynapse::NET_P n)
{
    if(n == NULL) {DL_BADPOINTER(1, "net"); return -1;}
    return layer_show_raw(n->output_l);
}
int DSynapse::net_show_input(DSynapse::NET_P n)
{
    if(n == NULL) {DL_BADPOINTER(1, "net"); return -1;}
    return layer_show_input(n->input_l);
}
int DSynapse::net_show_out(DSynapse::NET_P n)
{
    if(n == NULL) {DL_BADPOINTER(1, "net"); return -1;}
    return layer_show_input(n->output_l);
}
int DSynapse::net_show_target(DSynapse::NET_P n)
{
    if(n == NULL) {DL_BADPOINTER(1, "net"); return -1;}
    if(n->output_l && n->output_l->size)
    {
        FOR_VALUE(n->output_l->size, i)
        {
            std::cout << n->target[i] << ' ';
        }
        std::cout << std::endl;
    }
    return 0;
}
int DSynapse::net_show_out_error(DSynapse::NET_P n)
{
    if(n == NULL) {DL_BADPOINTER(1, "net"); return -1;}
    return layer_show_error(n->output_l);
}
int DSynapse::net_apply_delta(DSynapse::NET_P n)
{
    return net_run_layers(n, layer_apply_delta);
}
int DSynapse::net_zero_delta(DSynapse::NET_P n)
{
    return net_run_layers(n, layer_zero_delta);
}
int DSynapse::net_flush_delta(DSynapse::NET_P n)
{
    if(  net_run_layers(n, layer_apply_delta) < 0  )
    {
        DL_FUNCFAIL(1, "layer_apply_delta");
        return -1;
    }
    if(  net_zero_delta(n) < 0  )
    {
        DL_FUNCFAIL(1, "net_zero_delta");
        return -1;
    }

    return 0;
}
int DSynapse::net_flush_delta(DSynapse::NET_P n, DSynapse::NET_P from)
{
    if(  net_run_layers(n, from, layer_apply_delta) < 0  )
    {
        DL_FUNCFAIL(1, "layer_apply_delta");
        return -1;
    }
    if(  net_zero_delta(from) < 0  )
    {
        DL_FUNCFAIL(1, "net_zero_delta");
        return -1;
    }

    return 0;
}
int DSynapse::net_use_learn_mode(DSynapse::NET_P n, int state)
{
    if(n == NULL) {DL_BADPOINTER(1, "net");return -1;}
    n->learn_mode = state;
    return 0;
}
int DSynapse::net_learn(DSynapse::NET_P n)
{
    if(n == NULL) {DL_BADPOINTER(1, "net");return -1;}
    n->total_error = 0.0f;
    int delta_c = 0;
    int batch_c = 0;
    if(n->EpocSize > 0)
    {
        for(int i=0;i!=n->EpocSize;++i)
        {
            if( net_load_sample(n, i) < 0) {
                DL_FUNCFAIL(1, "net_load_sample");
                break;
            }
//            std::cout << "net_learn: FORWARD" << std::endl;
            if( forward_prop(n) < 0)
            {
                DL_FUNCFAIL(1, "forward_prop");
                break;
            }
//            std::cout << "net_learn: BACK" << std::endl;
            if(batch_c++ == n->batchSize)
            {
                batch_c = 0;
                if( back_prop(n) < 0)
                {
                    DL_FUNCFAIL(1, "back_prop");
                    break;
                }
                ++delta_c;
            }
//            if(delta_c == n->delta_flush_freq){ //avoid when delta_flush_freq == 0
//                delta_c = 0;
//                net_flush_delta(n);
//            }
        }
//        net_flush_delta(n);
    }
    else
    {
        DL_BADVALUE(1, "EpocSize");
        return -1;
    }

    return 0;
}
int DSynapse::net_test(DSynapse::NET_P n)
{
    if(n == NULL) {DL_BADPOINTER(1, "net");return -1;}
    if(n->EpocSize > 0)
    {
        FOR_VALUE(n->EpocSize, i)
        {
            if(net_load_sample(n, i) < 0)
            {
                DL_FUNCFAIL(1, "net_get_sample");
                break;
            }
            if( forward_prop(n) < 0)
            {
                DL_FUNCFAIL(1, "forward_prop");
                break;
            }

            printf("input: ");
            net_show_input(n);

            net_restore_out(n);
            printf("output: ");
            net_show_out(n);
            printf("----------------------- net: %p\n", n);
        }
    }
    return 0;
}























