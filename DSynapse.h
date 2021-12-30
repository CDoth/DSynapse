#ifndef DSYNAPSE_H
#define DSYNAPSE_H
#include <stdio.h>
#include <DLogs.h>
#include <DRand.h>
#include <daran.h>

namespace DSynapse
{


// 1. Late delta flush  +
// 2. Batch size +
// 3. Soft max +
// 4. Weight init +
// 5. Error export callback +
// 6. Error estimation callback (loss function) +
// 7. Drop out +
// 8. L2 reg
// 9. Batch norm
// 10. ensembles (merge nets)
// 11. Label smoothing

int DSynapseInitLogContext(const char *stream_name);
DLogs::DLogsContext* DSynapseGetLogContext();
void DSynapseSetLogContext(const DLogs::DLogsContext &c);

///
/// \brief nn_value_type
/// Main type of values in neuron network. Type of
/// signals and errors.
typedef float nn_value_type, nvt;
///
/// \brief data_line
/// Array of nvt values.
typedef nvt* data_line;
typedef const nvt* const_data_line;
typedef int* mask_t;
typedef const int* const_mask_t;


typedef nvt (*rand_callback)();
typedef nvt (*ranged_rand_callback)(nvt bot, nvt top);
///
/// \brief global_rand_callback
/// This callback should return rand value in [0.0, 1.0] range
extern rand_callback global_rand_callback;
///
/// \brief global_ranged_rand_callback
/// This callback should return rand value in [bot, top] range
extern ranged_rand_callback global_ranged_rand_callback;
void DSynapseSetRandCallback(rand_callback c);
void DSynapseSetRangedRandCallback(ranged_rand_callback c);

data_line alloc_buffer(int size);
data_line realloc_buffer(data_line b, int size);
data_line copy_buffer(data_line buffer, int size);
//----------------------------------------------------------------------
///
/// \brief ACTF
/// Type of pointer to activation or der_activation function
typedef nvt (*ACTF)(nvt);
///
/// \brief The ACTIVATION enum
/// List of short names for activation functions.
enum ACTIVATION
{
    ACT_SIGMOID
    ,ACT_GTAN
    ,ACT_RELU
    ,ACT_LEAKYRELU
    ,ACT_GELU
    ,ACT_ELU
    ,ACT_LINE
    ,ACT_EMPTY
    ,ACT_DEBUG
};
ACTIVATION actt_get_lable_by_index(int index);
ACTIVATION actt_get_lable_by_callback(ACTF a);
int actt_get_index_by_lable(ACTIVATION a);
int actt_get_index_by_callback(ACTF a);
void actt_get_callback_by_index(int index, ACTF &a, ACTF &da);
void actt_get_callback_by_lable(ACTIVATION t, ACTF &a, ACTF &da);
///
/// activation and their derivative functions:
#define DSYNAPSE_LEAKY_RELU_RATE (0.001)
#define DSYNAPSE_ELU_RATE (0.001)

#define ACTIVATION_WRAPPER(activation_name) \
    nvt act_##activation_name(nvt x); \
    nvt der_act_##activation_name(nvt x);

ACTIVATION_WRAPPER(sigmoid)
ACTIVATION_WRAPPER(gtan)
ACTIVATION_WRAPPER(ReLU)
ACTIVATION_WRAPPER(LeakyReLU)
ACTIVATION_WRAPPER(GELU)
ACTIVATION_WRAPPER(ELU)
ACTIVATION_WRAPPER(LINE)
ACTIVATION_WRAPPER(empty)
ACTIVATION_WRAPPER(debug)
//----------------------------------------------------------------------
enum DROPOUT
{
    DROPOUT_NO
    ,DROPOUT_FW
    ,DROPOUT_IV
};
enum LAYERMASK
{
    LAYERMASK_EMPTY
    ,LAYERMASK_DROPOUT
    ,LAYERMASK_RANDOM
    ,LAYERMASK_DEBUG
    ,LAYERMASK_ENABLEALL
    ,LAYERMASK_DISABLEALL
};


typedef int (*FPF)(data_line OUT
                   ,const_data_line INPUT
                   ,const_data_line WEIGHT
                   ,const_data_line BIAS
                   ,int IN_SIZE
                   ,int OUT_SIZE
                   ,ACTF act);
///
/// Type of pointer to back propagation function
///
typedef int (*BPF)(data_line CURR_ERROR,
                    data_line PREV_ERROR,
                    data_line WEIGHT,
                    const_data_line INPUT_SIGNAL,
                    const_data_line DERACT_INPUT,
                    data_line BIAS,
                    nvt LEARN_RATE,
                    int CURR_LAYER_SIZE,
                    int PREV_LAYER_SIZE,
                    void *SPECIAL);

struct WeightMatrix
{
    int mainSize;
    int biasSize;
    data_line main;
    data_line bias;
};
enum WM_BUFFER_TYPE
{
    WM_MAIN
    , WM_BIAS
    , WM_BOTH
    , WM_IGNORE
};
#define DSYNAPSE_USE_WM_SAVE_CALL
WeightMatrix* wm_alloc();
int wm_free(WeightMatrix *wm);
WeightMatrix* wm_copy(WeightMatrix *src);

int wm_create_matrix(WeightMatrix *m, int mainSize, int biasSize);
int wm_create_matrix2(WeightMatrix *m, int currentLayerSize, int nextLayerSize);
int wm_recreate_matrix(WeightMatrix *m, int mainSize, int biasSize);
int wm_recreate_matrix2(WeightMatrix *m, int currentLayerSize, int nextLayerSize);
int wm_destruct_matrix(WeightMatrix *m);

int wm_zero_matrix(WeightMatrix *m, WM_BUFFER_TYPE t);
int wm_copy_weights_value(WeightMatrix *dst, WeightMatrix *src, WM_BUFFER_TYPE t);
int wm_set_rand(WeightMatrix *m, WM_BUFFER_TYPE t);
int wm_set_rand(WeightMatrix *m, WM_BUFFER_TYPE t, nvt bot, nvt top);
int wm_set_rand(WeightMatrix *m, WM_BUFFER_TYPE t, rand_callback rc);
int wm_set_rand(WeightMatrix *m, WM_BUFFER_TYPE t, ranged_rand_callback rrc, nvt bot, nvt top);
int wm_set_weight_value(WeightMatrix *m, int wpos, nvt value);
int wm_set_bias_value(WeightMatrix *m, int bpos, nvt value);

int wm_print(WeightMatrix *m, WM_BUFFER_TYPE t);





///
/// \brief The DescentContext struct
/// Special struct for each layer. It store some
/// arrays and settings for back propagation functions and
/// other methods like 'delta method'.
/// Create and free objects of this type by call 'alloc_descent_context' and 'free_descent_context'.
struct DescentContext
{
    /*
    int w_size;
    int b_size;
    ///
    /// \brief delta_w
    /// Deltas of weights for current layer.
    data_line delta_w;
    ///
    /// \brief delta_b
    /// Deltas of bias for current layer.
    data_line delta_b;
    */
    WeightMatrix *deltaBuffer;


    int use_support_buffer;
    int use_main_buffer;
    data_line *mt_error;

    //----------
    ///
    /// \brief special_store1
    /// Special array. Mean of this array depends on kind of back propagation function.
    /// Size of 'special_store1' and 'special_store2' is layer->w_size (layer which contain pointer to this struct)
//    data_line special_buffer1;
    WeightMatrix *special_buffer1;

    ///
    /// Special hyperparameters for some back propagation functions:
    /// Info by https://cs231n.github.io/neural-networks-3/#ada
    ///
    /// \brief mu
    /// mu: additional hyperparameter, this variable is in optimization referred to as momentum.
    /// Effectively, this variable damps the velocity and reduces the kinetic energy of the system,
    /// or otherwise the particle would never come to a stop at the bottom of a hill.
    /// When cross-validated, this parameter is usually set to values such as [0.5, 0.9, 0.95, 0.99].
    double mu; //momentum, nesterov
    ///
    /// \brief eps
    /// The smoothing term eps (usually set somewhere in range from 1e-4 to 1e-8) avoids division by zero.
    double eps; //adagrad, rms, adam
    ///
    /// \brief decay_rate
    /// decay_rate is a hyperparameter and typical values are [0.9, 0.99, 0.999]
    double decay_rate; //rms
    //adam:
//    data_line special_buffer2;
    WeightMatrix *special_buffer2;
    ///
    /// \brief beta1 & beta2
    /// beta1 = 0.9, beta2 = 0.999
    nvt beta1;
    nvt beta2;
    ///
    /// \brief t
    /// t is your iteration counter going from 1 to infinity
    int t;
};
DescentContext* alloc_descent_context();
DescentContext* copy_descent_context(DescentContext *from);
int free_descent_context(DescentContext *c);
int descent_context_clear_buffers(DescentContext *c);



int forward_propagation(
                        data_line out,
                        const_data_line input, const_data_line w, const_data_line bias,
                        int in_size, int out_size, ACTF act
        );
int forward_propagation_learn(
                        data_line out, data_line raw, data_line der_out,
                        const_data_line input, const_data_line w, const_data_line bias,
                        int in_size, int out_size, ACTF act, ACTF der_act
        );
int forward_propagation_learn_masked(
                        data_line out, data_line raw, data_line der_out,
                        const_mask_t outMask,
                        const_data_line input, const_data_line w, const_data_line bias,
                        int in_size, int out_size, ACTF act, ACTF der_act
        );
///
/// \brief The BACKPROP_FUNCTION enum
/// List of short names of back propagation functions
enum BACKPROP_FUNCTION
{
    BP_DEFAULT
    ,BP_MOMENTUM
    ,BP_NESTEROV
    ,BP_ADAGRAD
    ,BP_RMSPROP
    ,BP_ADAM1
    ,BP_ADAM2
};
int back_propagation(data_line curr_error, data_line prev_error, data_line w,
                    const_data_line prev_input, const_data_line curr_der_input, data_line bias,
                    nvt learn_rate, int curr_size, int prev_size, void *special = NULL
        );
int back_propagation_masked(data_line curr_error, data_line prev_error, data_line w,
                      const_data_line currMask,
                    const_data_line prev_input, const_data_line curr_der_input, data_line bias,
                    nvt learn_rate, int curr_size, int prev_size, void *special = NULL
        );
typedef nvt (*gradient_descent_callback)(DescentContext *c);
int back_propagation_opt(data_line curr_error, data_line prev_error, data_line w,
                    const_data_line prev_input, const_data_line curr_der_input, data_line bias,
                    nvt learn_rate, int curr_size, int prev_size,
                         gradient_descent_callback gd_callback,
                         void *special = NULL
        );
#define DSYNAPSE_USE_FORWARD_PROP_SAVE_CALL
#define DSYNAPSE_USE_BACK_PROP_SAVE_CALL
#define DSYNAPSE_USE_LOSSF_SAVE_CALL

typedef int (*LOSS_FUNCTION_CALLBACK)(
            const_data_line out
        ,   const_data_line target
        ,   data_line out_error
        ,   nvt *total_error
        ,   int size
        );

int loss_function_default(const_data_line out, const_data_line target, data_line out_error, nvt *total_error, int size);
int loss_function_quadratic(const_data_line out, const_data_line target, data_line out_error, nvt *total_error, int size);
int loss_function_cross_entropy(const_data_line out, const_data_line target, data_line out_error, nvt *total_error, int size);


extern DLOGS_DEFINE_DEFAULT_CONTEXT

}
#endif // DSYNAPSE_H
