use burn::tensor::backend::Backend;
use crate::fc_model::{FcValueNetwork, INPUT_SIZE};
use crate::nnue::NnueWeights;

/// Extract weights from a trained burn FcValueNetwork into NnueWeights.
///
/// This export format requires exactly 3 layers (2 hidden + 1 output),
/// i.e., the model must have been created with `hidden_sizes = &[l1, l2]`.
///
/// Burn `Linear` stores weights as `[d_input, d_output]` row-major (O = I * W).
/// L1: burn [INPUT_SIZE, l1_size] -- NNUE reads column-major, same flat layout (no transpose).
/// L2/L3: burn [in, out] row-major -> NNUE [out, in] row-major (transpose).
pub fn export_weights<B: Backend>(model: &FcValueNetwork<B>, l1_size: usize, l2_size: usize) -> NnueWeights {
    assert_eq!(model.num_layers(), 3,
        "export_weights requires exactly 3 layers (2 hidden + 1 output), got {}. \
         The model must have been created with hidden_sizes = &[l1, l2].",
        model.num_layers());

    let l1_weight: Vec<f32> = model.layers[0].weight.val().into_data().to_vec().expect("fc1 weight");
    assert_eq!(l1_weight.len(), INPUT_SIZE * l1_size, "L1 weight size mismatch");
    let l1_bias: Vec<f32> = model.layers[0].bias.as_ref().expect("fc1 bias").val().into_data().to_vec().expect("fc1 bias");

    let fc2_burn: Vec<f32> = model.layers[1].weight.val().into_data().to_vec().expect("fc2 weight");
    let l2_weight = transpose(&fc2_burn, l1_size, l2_size);
    let l2_bias: Vec<f32> = model.layers[1].bias.as_ref().expect("fc2 bias").val().into_data().to_vec().expect("fc2 bias");

    let fc3_burn: Vec<f32> = model.layers[2].weight.val().into_data().to_vec().expect("fc3 weight");
    let l3_weight = transpose(&fc3_burn, l2_size, 1);
    let l3_bias: Vec<f32> = model.layers[2].bias.as_ref().expect("fc3 bias").val().into_data().to_vec().expect("fc3 bias");

    NnueWeights { l1_size, l2_size, l1_weight, l1_bias, l2_weight, l2_bias, l3_weight, l3_bias }
}

fn transpose(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(data.len(), rows * cols);
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}
