use burn::tensor::backend::Backend;
use crate::fc_model::{FcValueNetwork, INPUT_SIZE, L1_SIZE, L2_SIZE};
use crate::nnue::NnueWeights;

/// Extract weights from a trained burn FcValueNetwork into NnueWeights.
///
/// burn's Linear stores weight as [d_input, d_output] and computes O = X @ W + b.
/// The NNUE engine expects weights in [d_output, d_input] row-major layout,
/// where l1_weight[i * d_input + j] is the weight from input j to output i.
/// So we need to transpose the weight matrices during export.
pub fn export_weights<B: Backend>(model: &FcValueNetwork<B>) -> NnueWeights {
    // fc1: burn shape [INPUT_SIZE, L1_SIZE] -> NNUE shape [L1_SIZE, INPUT_SIZE]
    let fc1_weight_burn: Vec<f32> = model
        .fc1
        .weight
        .val()
        .into_data()
        .to_vec()
        .expect("fc1 weight");
    let l1_weight = transpose(&fc1_weight_burn, INPUT_SIZE, L1_SIZE);

    let l1_bias: Vec<f32> = model
        .fc1
        .bias
        .as_ref()
        .expect("fc1 has no bias")
        .val()
        .into_data()
        .to_vec()
        .expect("fc1 bias");

    // fc2: burn shape [L1_SIZE, L2_SIZE] -> NNUE shape [L2_SIZE, L1_SIZE]
    let fc2_weight_burn: Vec<f32> = model
        .fc2
        .weight
        .val()
        .into_data()
        .to_vec()
        .expect("fc2 weight");
    let l2_weight = transpose(&fc2_weight_burn, L1_SIZE, L2_SIZE);

    let l2_bias: Vec<f32> = model
        .fc2
        .bias
        .as_ref()
        .expect("fc2 has no bias")
        .val()
        .into_data()
        .to_vec()
        .expect("fc2 bias");

    // fc3: burn shape [L2_SIZE, 1] -> NNUE shape [1, L2_SIZE]
    let fc3_weight_burn: Vec<f32> = model
        .fc3
        .weight
        .val()
        .into_data()
        .to_vec()
        .expect("fc3 weight");
    let l3_weight = transpose(&fc3_weight_burn, L2_SIZE, 1);

    let l3_bias: Vec<f32> = model
        .fc3
        .bias
        .as_ref()
        .expect("fc3 has no bias")
        .val()
        .into_data()
        .to_vec()
        .expect("fc3 bias");

    NnueWeights {
        l1_weight,
        l1_bias,
        l2_weight,
        l2_bias,
        l3_weight,
        l3_bias,
    }
}

/// Transpose a row-major matrix from [rows, cols] to [cols, rows].
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
