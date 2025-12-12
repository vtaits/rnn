use crate::structures::LayerParams;

pub fn get_next_field(params: &LayerParams, field_x: usize, field_y: usize) -> (usize, usize) {
    if field_x == 0 || field_y == params.layer_height - 1 {
        let next_x = field_y + field_x + 1;

        if next_x < params.layer_width {
            return (next_x, 0);
        }

        let next_y = next_x - params.layer_width + 1;

        return (params.layer_width - 1, next_y);
    }

    (field_x - 1, field_y + 1)
}

pub fn get_last_field(params: &LayerParams) -> (usize, usize) {
    (params.layer_width - 1, params.layer_height - 1)
}

pub fn get_output_field(params: &LayerParams, output_field_index: usize) -> (usize, usize) {
    let mut res_x = 0;
    let mut res_y = 0;

    for _ in 0..output_field_index {
        let next_field = get_next_field(params, res_x, res_y);

        res_x = next_field.0;
        res_y = next_field.1;
    }

    (res_x, res_y)
}
