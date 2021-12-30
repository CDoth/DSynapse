#ifndef LAYER_H
#define LAYER_H
#include <DSynapse.h>


namespace DSynapse
{
typedef struct net *NET_P;
typedef struct layer
{
    ///
    /// \brief input
    /// Main signal in layer. It is activated signal from 'raw'
    /// Size of this array is size of this layer, value stored in 'size' field.
    data_line input;
    ///
    /// \brief raw
    /// Not activated signal (only weight + bias)
    /// Size of this array is size of this layer, value stored in 'size' field.
    data_line raw;
    ///
    /// \brief error
    /// Layer error. Need only for learning (back propagation)
    /// Size of this array is size of this layer, value stored in 'size' field.
    data_line error;
    ///
    /// \brief der_input
    /// Raw signal after derivative of actiovation function. Need only for learning (back propagation)
    /// Size of this array is size of this layer, value stored in 'size' field.
    data_line der_input;
    mask_t mask;

    /*
    ///
    /// \brief weight
    /// Weights between this layer and 'next' layer. This weight use for forward
    /// propagation from this layer to 'next' layer.
    /// Need for forward and back propagations.
    /// Size of this array is product of size of this layer ('size' filed)
    /// and size of 'next' layer (next->size)
    /// In memory this looks like:
    /// [InputWeights<0>][InputWeights<1>]...[InputWeights<next->size-1>],
    /// where InputWeights<N> is C-style array wich store 'size' weights
    /// between all input neurons (current layer) and one single out neuron (next layer) from next layer
    /// with 'N' index.
    ///
    data_line weight;
    ///
    /// \brief bias
    /// Weights with "bias" function, need for shift out values.
    /// Size of this array is size of 'next' layer (next->size)
    /// In memory this looks like:
    /// [Bias<0>][Bias<1>]...[Bias<next->size-1>],
    /// where Bias<N> is single value with type "nvt" and each value
    /// associated with each output neuron from 'next' layer.
    data_line bias;
    ///
    /// \brief w_size
    /// Number of weights (without bias) which this layer include.
    /// No "bytes" size, just number of weights.
    /// w_size equal 'size' * 'next->size'.
    int w_size;
    int b_size;
    */
    WeightMatrix *weigthMatrix;

    ///
    /// \brief size
    /// Size of this layer. Size of neurons in this layer.
    /// No "bytes" size, just number of neurons.
    int size;


    ///
    /// \brief act & der_act
    /// Activation and derivative of activation functions for this layer. Each layer can use
    /// own function or you can use global activation function (read 'net' struct brief, 'act' field).
    ACTF act;
    ACTF der_act;

    ///
    /// \brief learn_rate
    /// Learn rate for this layer. Each layer can use own learn rate or you can use global learn rate
    /// (read 'net' struct brief, 'global_learn_rate' field).
    nvt learn_rate;
    ///
    /// \brief use_act_out
    /// This flag points to which input array ('raw' or 'input') layer will use for
    /// back propagation function. This flag can be local or global (read 'net' struct brief, 'use_act_out' field)
    bool use_act_out;

    ///
    /// \brief next
    /// Pointer no next layer in net.
    layer *next;
    ///
    /// \brief prev
    /// Pointer to previous layer in net.
    layer *prev;

    ///
    /// \brief back_prop
    /// Back propagation function. Each layer can use own function or you can
    /// set global function for all layers in net (read 'net' struct brief, 'back_prop' field).
    BPF back_prop;

    ///
    /// \brief dcontext
    /// Additional context for special back propagation functions and some other functions.
    /// Deafult value: NULL
    /// Alloc and free this with 'alloc_descent_context' and 'free_descent_context'.
    DescentContext *dcontext;
    // Dropout:
    float dropoutRate;
    DROPOUT dropoutType;

    NET_P parent;
public:
} *LAYER_P;
///
/// \brief alloc_layer
/// \param _size
///  Number of neurons in this layer
/// \return
/// Pointer to new layer
///
/// Get memory for 'layer' struct, set default values, and return pointer to it.
LAYER_P alloc_layer(int _size);
///
/// \brief layer_copy
/// \param copy_this
/// \param replace_weight_matrix
/// This flag points what layer->weight does mean
/// 0: alloc 'weight' array by default, with 'w_size'
/// 1: set 'weight' pointer value to copy_this->weight, so both layers has common 'weight' array,
/// use it if you need common weights.
///
/// \return
/// Get memoru for 'struct', set values from original layer 'copy_this', and return pointer to it
LAYER_P layer_copy(const LAYER_P copy_this, int size = 0, int replace_weight_matrix = 0);
///
/// \brief layer_unique_copy
/// Wrapper for layer_copy(copy_this, 0)
/// \param copy_this
/// \return
///
LAYER_P layer_unique_copy(LAYER_P copy_this);
///
/// \brief layer_shared_copy
/// Wrapper for layer_copy(copy_this, 1)
/// \param copy_this
/// \return
///
LAYER_P layer_shared_copy(LAYER_P copy_this);
LAYER_P layer_modify(LAYER_P l, int new_size);
int free_layer(LAYER_P l);
int layer_get_size(LAYER_P l);
int layer_alloc_wm(LAYER_P l);

int layer_set_weight(LAYER_P l, int wpos, nvt value);
int layer_set_weight(LAYER_P l, int currentNeuron, int nextNeuron, nvt value);
//---------------------------------- LAYER ABSTRACT TOOLS:
int layer_clear_buffers(LAYER_P l);
int layer_set_rand_signal(LAYER_P l); // set random values to layer->input and layer->raw arrays (for debug)
int layer_set_randw(LAYER_P l); //set random values to layer->weight. This function set integer values in [0-100) range (for debug)
int layer_check(LAYER_P l);
int layer_free_weight(LAYER_P l);
int layer_alloc_derinput(LAYER_P l);
int layer_alloc_connection_buffers(LAYER_P l);
// show:
int layer_show_input(LAYER_P l);
int layer_show_raw(LAYER_P l);
int layer_show_deri(LAYER_P l);
int layer_show_signals(LAYER_P l);
int layer_show_error(LAYER_P l);
int layer_show_w(LAYER_P l);
int layer_show_delta(LAYER_P l);
// delta:
int layer_alloc_delta(LAYER_P l); //get mem for layer->dcontext->delta_w and layer->dcontext->delta_b (using sizes from 'l')
int layer_free_delta(LAYER_P l);
int layer_apply_delta(LAYER_P l); // add delta (layer->dcontext->delta_w and layer->dcontext->delta_b) to layer->weight and layer->bias
int layer_zero_delta(LAYER_P l); //clear deltas arrays (set zero values)
int layer_flush_delta(LAYER_P l);
//---------------------------------- LAYER DEPEND TOOLS:
int layer_set_next(LAYER_P l, LAYER_P next); //set 'next' pointer to 'l' and alloc 'weight' and 'bias' arrays in 'l'
int layer_set_prev(LAYER_P l, LAYER_P prev); //set 'prev' pointer to 'l' and alloc 'der_input' array in 'l'
int layer_apply_delta(LAYER_P l, LAYER_P from); //add delta from 'from' layer to l->weight and l->bias
int layer_copy_params(LAYER_P to, LAYER_P from); //copy main parameters (include dcontext with its arrays)
int layer_copy_weights(LAYER_P dst, LAYER_P src);// copy src->weight values to dst->weight (and copy bias)
//---------------------------------- LAYER PARENT TOOLS:
int layer_set_opt_randw(LAYER_P l, NET_P n);
int layer_set_ranged_randw(LAYER_P l, NET_P n);
/*
int layer_set_bpf(LAYER_P l, NET_P n); //set back propagation function to 'l' (l->back_prop) using value from net (n->back_prop)
int layer_set_actf(LAYER_P l, NET_P n); //set actiovation and der_activation functions to 'l' (l->act, l->der_act) using values from net (n->act, n->der_act)
int layer_use_act_out(LAYER_P l, NET_P n); //set use_act_out flag using net value (n->use_act_out)
int layer_set_lr(LAYER_P l, NET_P n); //set learn rate to layer using net value (n->global_learn_rate)
*/
int layer_use_delta_buffers(LAYER_P l, NET_P n);
//---------------------------------- LAYER SPECIAL TOOLS:
int layer_set_ranged_randw(LAYER_P l, nvt bot, nvt top);
int layer_use_delta_buffers(LAYER_P l, int support, int main);
int layer_connect(LAYER_P l, LAYER_P next);
int layer_set_actf(LAYER_P l, ACTF a, ACTF d); //set actiovation functions
int layer_set_actf(LAYER_P l, ACTIVATION a); //set actiovation functions
int layer_set_actf(LAYER_P l, int activationIndex);
int layer_set_lr(LAYER_P l, nvt lr); //set learn rate value
int layer_increase_lr(LAYER_P l, nvt value); //increase learn rate by 'value'
int layer_decrease_lr(LAYER_P l, nvt value); //decrease learn rate by 'value'
int layer_set_bpf(LAYER_P l, BPF f); //set back propagation function
int layer_use_act_out(LAYER_P l, int state); //set l->use_act_out flag


int layer_alloc_mask(LAYER_P l);
typedef int (*generate_mask_callback)(LAYER_P);
int generate_mask_dropout(LAYER_P l);
int generate_mask_random(LAYER_P l);
int generate_mask_debug(LAYER_P l);
int generate_mask_enable_all(LAYER_P l);
int generate_mask_disable_all(LAYER_P l);

int layer_set_dropout(LAYER_P l, float rate, DROPOUT t);
int layer_generate_mask(LAYER_P l, LAYERMASK maskType);
int layer_set_mask(LAYER_P l, const_mask_t mask, int maskSize);
int layer_apply_mask(LAYER_P l);


enum LAYER_DELTA_BUFFERS
{
    LAYER_USE_ONLY_MAIN_BUFFER
    ,LAYER_USE_ONLY_SUPPORT_BUFFER
    ,LAYER_USE_MAIN_AND_SUPPORT_BUFFER
    ,LAYER_IGNORE_BUFFERS
};
int _is_support(LAYER_DELTA_BUFFERS s);
int _is_main(LAYER_DELTA_BUFFERS s);
///
/// \brief LAYER_ABSTRACT_ACTION
/// Type of pointer to function, which has only one layer to do something with it.
typedef int (*LAYER_ABSTRACT_ACTION)(LAYER_P);
///
/// \brief LAYER_PARENT_ACTION
/// Type of pointer to function, which has pointer to layer and pointer to its net (parent net for this layer)
typedef int (*LAYER_PARENT_ACTION)(LAYER_P, NET_P);
///
/// \brief LAYER_DEPEND_ACTION
/// Type of pointer to fucntion, which has pointers to two layers to provide some communication between them
typedef int (*LAYER_DEPEND_ACTION)(LAYER_P, LAYER_P);

typedef int (*LAYER_SPECIAL_ACTION1)(LAYER_P, int);
typedef int (*LAYER_SPECIAL_ACTION2)(LAYER_P, nvt);





typedef LAYER_ABSTRACT_ACTION LAYER_FILTER;
int layer_filter_activation(LAYER_P l);
int layer_filter_derivative_activation(LAYER_P l);
int layer_filter_symmetric_activation(LAYER_P l);
int layer_filter_dropout_forward_scaledown(LAYER_P l);
int layer_filter_dropout_inverted_scaledown(LAYER_P l);
int layer_filter_softmax(LAYER_P l);
int layer_filter_softmax_activation(LAYER_P l);
int layer_filter_maxout(LAYER_P l);

int layer_filter_max_blur(LAYER_P l);
int layer_filter_mean_blur(LAYER_P l);
int layer_filter_exp_blur(LAYER_P l);

#define LAYER_USE_SAVE_CALL
#define LAYER_FILTER_SAVE_CALL
}

#endif // LAYER_H
