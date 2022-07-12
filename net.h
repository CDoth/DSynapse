#ifndef NET_H
#define NET_H
#include <DSynapse.h>
#include <layer.h>




namespace DSynapse
{
typedef int (*LOAD_SAMPLE_CALLBACK)(
          data_line _input
        , data_line _target
        , const void *opaqueData
        , int iteration
        );
typedef int (*EXPORT_ERROR_CALLBACK)(
            data_line _error
        ,   const void *opaque
        ,   int iteration
        );

typedef int (*WEIGHT_INIT_RANGE_CALLBACK)(
        int currentLayerSize
        , int nextLayerSize
        , nvt *bot
        , nvt *top
        );

typedef struct net {
    // Main:
    int learn_mode;
    ///
    /// \brief set
    /// Set of layers
    LAYER_P *set;
    ///
    /// \brief size
    /// Layers in net (include input layer, all hidden layers and output layer)
    int size;

    ///
    /// \brief input_l, output_l
    /// First and last layers in set
    LAYER_P input_l;
    LAYER_P output_l;

    ///
    /// \brief in, out, error
    /// Pointers to real buffers in layers
    /// input_l contain: in
    /// output_l contain: out, error
    data_line in;
    data_line out;
    data_line error;

    ///
    /// \brief target
    /// Alloced buffer for target output
    data_line target;

    ///
    /// \brief total_error
    /// Sum of all errors (abs value) by last epoc
    nvt total_error;
//    ///
//    /// \brief total_learn_rate
//    /// Alternative for layer-specific learn rate
//    nvt learn_rate;
    ///
    /// \brief EpocCounter
    /// Increase by one each learn function call
    int EpocCounter;
    ///
    /// \brief EpocSize
    /// Number of forward&back propagations for one learn function call
    /// In general: dataset size.
    int EpocSize;

    // Learn tools:
//    ///
//    /// \brief act
//    /// Activation function, like:
//    /// act_sigmoid, act_ReLU
//    ACTF act;
//    ///
//    /// \brief der_act
//    /// Derivative of activation function, like:
//    /// der_act_sigmoid, der_act_ReLU
//    ACTF der_act;
    ///
    /// \brief back_prop
    /// Back propagation function which point to wich descent tool to use, like:
    /// back_propagation_adagrad, back_propagation_rmsprop, back_propagation (default)
    BPF back_prop;

    // Input loading tools:
    /// There is three ways for get input data for net while it works.

    ///
    /// \brief GetNextSample
    /// This is callback function for getting input and target values for learning.
    /// Net call this when it need new data from data-set for forward and back propagations.
    ///
    /// 1. Default callback. By default 'GetNextData' points to '__get_next_data_default' function.
    /// You can use it if your input data can exist in line mode (C-style). So you have to
    /// set trainInput and trainTarget pointers to your data-set storage
    /// eg:
    /// trainInput is C-style array and contains input data like this:
    /// [InputData<0>][InputData<1>]...[InputData<EpocSize-1>], and InputData is sub-array with size == input_l->size (size of elements, multiply it with sizeof(nvt) to get bytes size)
    /// same for trainTarget:
    /// [TargetOutputData<0>][TargetOutputData<1>]...[TargetOutputData<EpocSize-1>], and TargetOutputData is sub-array with size == output_l->size
    /// By using default callback opaque pointer will automatically set to address of current net and can't changed
    ///
    /// 2. Custom input storage. You can set your custom callback to 'GetNextData' which will use 'opaque' pointer to get input data.
    /// 3. Global input storage. You can set your custom callback to 'GetNextData' and set NULL to 'opaque', so in this case your
    /// callback should generate new input data in real time or get it from global storage.
    LOAD_SAMPLE_CALLBACK loadSample;
    ///
    /// \brief opaque
    /// Pointer to your data-set.
    /// 1. Set it to NULL to use 'global method' (read GetNextData brief).
    /// 2. Set it to data-set address to use 'custom method' (read GetNextData brief).
    /// 3. If net use 'default method' - opaque pointer should to point to this net. This value
    /// will set automatically after calling 'net_set_default_load'.
    /// Default value: current net
    const void *opaque;
    ///
    /// \brief trainInput, trainTarget
    /// Pointers to your data-set which exist in C-style (need for 'default method', read GetNextData brief).
    /// Default values: EMPTY_DATA_LINE
    data_line trainInput;
    data_line trainTarget;

    EXPORT_ERROR_CALLBACK exportError;
    const void *exportOpaque;


    LOSS_FUNCTION_CALLBACK lossFunction;

    // Progressive Learning mode:
    ///
    /// \brief delta_flush_freq
    /// Frequence of flushing weight delta by one epoc
    /// eg:
    /// delta_flush_freq == 1: flush deltas after each back propagation
    /// delta_flush_freq == 3: flush deltas after each third back propagation
    int delta_flush_freq;

    int batchSize;
    // Other:
    ///
    /// \brief input_qrate, output_qrate
    /// Values for rationing input and output, if it needed:
    /// input_qrate value means all inputs will divide by this value before forward propagation
    ///
    /// output_qrate value means all targets will divide by this value before back propagation
    /// and net outputs NEED to restoring after each epoc by call 'net_restore_out' function or automatically after
    /// each back(forward?) propagation (read use_auto_out_restoring)
    ///
    /// both values using for division and SHOULD TO HAVE NON-ZERO VALUES
    data_line input_qrate;
    data_line output_qrate;


    ///
    /// \brief rand_input_bot_value, rand_input_top_value
    /// To set rand inputs (debug mode)
    nvt rand_input_bot_value;
    nvt rand_input_top_value;
    WEIGHT_INIT_RANGE_CALLBACK weightInitRangeCallback;
    nvt rand_weights_bot_value;
    nvt rand_weights_top_value;

    // Multithreading:
    ///
    /// Don't set this field directly, use 'net_use_mt' and 'net_start_fp_daemon_threads' functions

//    MultiThreadContext *mt_context;
//    int ThreadIndex;
//    int ThreadStartPos;
//    int ThreadTasks;
//    NET_LEARN_CALLBACK mt_learn_callback;
//    NET_P support_net;

    // Metanet:
    NET_P baseNet;
    NET_P next;
    int refCount;
    // 'Use' options:
    ///
    /// \brief use_delta
    /// Set use_delta to '1' for manage of flushing process for weights delta (read delta_flush_freq)
    int use_delta;
//    ///
//    /// \brief use_daemon_fp
//    /// Set it with 'net_use_mt' function
//    int use_daemon_fp;
//    ///
//    /// \brief use_total_learn_rate
//    /// use_total_learn_rate points to which learn rate should to use:
//    /// 0: use layer specific learn rate
//    /// 1: use common net specific learn rate
//    int use_total_learn_rate;
    ///
    /// \brief use_random_input
    /// Debug mode
    int use_random_input;
//    ///
//    /// \brief use_act_out
//    /// use_act_out points to which input (in layer) data use for learning while back propagation:
//    /// 0: use raw input
//    /// 1: use activated input
//    int use_act_out;
//    ///
//    /// \brief use_global_activation
//    /// use_global_activation points to which activation and derivativation functions to use:
//    /// 0: use layer specific functions
//    /// 1: use common net specific functions
//    int use_global_activation;
//    ///
//    /// \brief use_global_backprop
//    /// use_global_backprop points to which bp function to use:
//    /// 0: use layer specific function
//    /// 1: use common net specific function
//    int use_global_backprop;
    ///
    /// \brief use_auto_out_restoring
    /// use_auto_out_restoring points to how output data will restoring if you use 'output_qrate':
    /// 0: output will not auto resoring and you should to call 'net_restore_out' after each FP
    /// if you use 'output_qrate' (output_qrate != 1)
    /// 1: output will auto restoring after each FP. So it means all output values will multiply with 'output_qrate'
    int use_auto_out_restoring;
    int use_support_buffer;
    int use_main_buffer;
    //--------------------------------------------
} *NET_P;
NET_P alloc_net();
NET_P net_copy(NET_P copy_this_net, int replace_wm);
NET_P net_shared_copy(NET_P copy_this_net); // call net_copy(copy_this_net, 1)
NET_P net_unique_copy(NET_P copy_this_net); // call net_copy(copy_this_net, 0);

int net_alloc_wms(NET_P n);
int net_alloc_derinp(NET_P n);

int net_get_size(NET_P n);

int net_detach(NET_P n);
int net_is_unique(NET_P n);
int net_make_unique(NET_P n);
void net_free(NET_P n);
int net_clear_all(NET_P n);



int net_run_layers(NET_P n, LAYER_ABSTRACT_ACTION a);
int net_run_layers(NET_P n, LAYER_PARENT_ACTION a);
int net_run_layers(NET_P n1, NET_P n2, LAYER_DEPEND_ACTION a);
int net_run_layers(NET_P n, LAYER_P l, LAYER_DEPEND_ACTION a);
int net_run_layers(NET_P n,  int arg, LAYER_SPECIAL_ACTION1 a);
int net_run_layers(NET_P n, nvt arg, LAYER_SPECIAL_ACTION2 a);

int net_run_conn_layers(NET_P n, LAYER_ABSTRACT_ACTION a);
int net_run_conn_layers(NET_P n1, NET_P n2, LAYER_DEPEND_ACTION a);



int net_set_randw(NET_P n);
int net_set_opt_randw(NET_P n);
int net_set_ranged_randw(NET_P n);
int net_set_ranged_randw(NET_P n, nvt bot, nvt top);

int net_set_rand_input(NET_P n);
int net_set_actf(NET_P n, ACTIVATION a);
int net_set_lr(NET_P n, nvt lr);
int net_set_input_qrate(NET_P n, data_line input);
int net_set_output_qrate(NET_P n, data_line output);
int net_fill_input_qrate(NET_P n, nvt qr);
int net_fill_output_qrate(NET_P n, nvt qr);

int net_set_epoc_size(NET_P n, int esize);
int net_set_bpf(NET_P n, BPF f);
int net_set_bpf(NET_P n, BACKPROP_FUNCTION bpf);
int net_set_loss_function(NET_P n, LOSS_FUNCTION_CALLBACK lf);
int net_set_options(NET_P n, nvt learn_rate, ACTIVATION actf, BACKPROP_FUNCTION bpf, bool use_act_out, nvt input_qr, nvt output_qr);
int net_set_options(NET_P n, nvt learn_rate, ACTIVATION actf, BACKPROP_FUNCTION bpf, bool use_act_out, data_line input_qr, data_line output_qr);
int net_set_lateflush_freq(NET_P n, int size);

int net_set_default_load(NET_P n, data_line input, data_line target);
int net_set_custom_load(NET_P n, LOAD_SAMPLE_CALLBACK callback, const void *data);
int net_set_global_load(NET_P n, LOAD_SAMPLE_CALLBACK callback);
int net_set_export_support_data(NET_P n, const void *data);


int net_set_input(NET_P n, nvt value, int index);
int net_get_output(NET_P n, nvt *value, int index);


//-------------------
int net_remove_neuron(NET_P n, int layerIndex, int npos);
int net_remove_neuron_set(NET_P n, int layerIndex, int nstart, int nend);

LAYER_P net_add_layer(NET_P n, int layer_size, int saveMeta);
LAYER_P net_insert_layer(NET_P n, int layer_size, int insertAfter, int saveMeta);
int net_modify_layer(NET_P n, int layer_index, int layer_new_size);
int net_remove_layer(NET_P n, int layer_index, int saveMeta);
int net_remove_layer(NET_P n, LAYER_P l);
//-------------------

int net_restore_out(NET_P n);
int net_copy_weights(NET_P dst, NET_P src);
int net_load_sample(NET_P n, int index);
int net_export_error(NET_P n, int index);
int net_get_layer_size(NET_P n, int layer_index);
int net_get_layer_index(NET_P n, LAYER_P l);
LAYER_P net_get_layer(NET_P n, int layer_index);
LAYER_P net_get_input_layer(NET_P n);
LAYER_P net_get_output_layer(NET_P n);
int net_increase_lr(NET_P n, nvt step);
int net_decrease_lr(NET_P n, nvt step);
int net_alloc_input_qrate_array(NET_P n);
int net_alloc_output_qrate_array(NET_P n);
int net_check(NET_P n);

int forward_prop(NET_P n);
int back_prop(NET_P n);
//--------------------------------
int net_use_delta_buffers(NET_P n, LAYER_DELTA_BUFFERS s);
int net_use_act_out(NET_P n, int state);
int net_use_random_input(NET_P n, int state, nvt bot = 0, nvt top = 0);
int net_use_learn_mode(NET_P n, int state);
//-------------------------------
int net_show_w(NET_P n);
int net_show_delta(NET_P n);
int net_show_raw_out(NET_P n);
int net_show_input(NET_P n);
int net_show_out(NET_P n);
int net_show_target(NET_P n);
int net_show_out_error(NET_P n);

int net_apply_delta(NET_P n);
int net_zero_delta(NET_P n);
int net_flush_delta(NET_P n);
int net_flush_delta(NET_P n, NET_P from);

int net_learn(NET_P n);
int net_test(NET_P n);


#define NET_USE_SAVE_CALL
}

#endif // NET_H
