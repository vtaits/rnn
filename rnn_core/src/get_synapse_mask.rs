use ndarray::Array2;

use crate::structures::{SynapseMask, SynapseParams};

const MIN_BETA: f32 = 0.001;
const MAX_OFFSET: usize = 100;

fn get_beta(synapse_params: &SynapseParams, distance: f32) -> f32 {
    return 1.0 / (1.0 + synapse_params.alpha * distance.powf(1.0 / synapse_params.h as f32));
}

fn get_max_offset(synapse_params: &SynapseParams) -> usize {
    for i in 1..=MAX_OFFSET {
        let beta = get_beta(synapse_params, i as f32);

        if beta < MIN_BETA {
            return i - 1;
        }
    }

    return MAX_OFFSET;
} 

pub fn get_synapse_mask(synapse_params: &SynapseParams) -> SynapseMask {
    let offset = get_max_offset(synapse_params);

    let size = 1 + 2 * offset;

    let mut mask = Array2::<f32>::zeros([size, size]);

    for i in 0..size {
        for j in 0..size {
            let i_diff = (i as i32) - (offset as i32);
            let j_diff = (j as i32) - (offset as i32);
            let distance = ((i_diff * i_diff + j_diff * j_diff) as f32).sqrt();

            let beta = get_beta(synapse_params, distance);

            mask[[i, j]] = beta;
        }
    }

    return SynapseMask {
        mask,
        offset,
        size,
    };
}
