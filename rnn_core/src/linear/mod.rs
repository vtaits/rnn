use crate::structures::LayerParams;

pub fn get_next_field(params: &LayerParams, field_x: usize, field_y: usize) -> (usize, usize) {
    let layer_2_to_1_x = {
        // last field in row
        if field_x == params.layer_width - 1 {
            0
        } else {
            field_x + 1
        }
    };

    let layer_2_to_1_y = {
        if field_x == params.layer_width - 1 {
            if field_y == params.layer_height - 1 {
                0
            } else {
                field_y + 1
            }
        } else {
            field_y
        }
    };

    (layer_2_to_1_x, layer_2_to_1_y)
}

pub fn get_last_field(params: &LayerParams) -> (usize, usize) {
    (params.layer_width - 1, params.layer_height - 1)
}

pub fn get_output_field(params: &LayerParams, output_field_index: usize) -> (usize, usize) {
    let output_field_y = output_field_index / params.layer_width;
    let output_field_x_rest = output_field_index % params.layer_width;

    let output_field_x = output_field_x_rest;

    (output_field_x, output_field_y)
}
