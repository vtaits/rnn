use crate::structures::NetworkParams;

pub fn get_next_field(params: &NetworkParams, field_x: usize, field_y: usize) -> (usize, usize) {
    let layer_2_to_1_x = if field_y % 2 == 0 {
        // last field in row
        if field_x == params.layer_width - 1 {
            if field_y == params.layer_height - 1 {
                0
            } else {
                field_x
            }
        } else {
            field_x + 1
        }
    } else {
        // first field in row
        if field_x == 0 {
            field_x
        } else {
            field_x - 1
        }
    };

    let layer_2_to_1_y = if field_y % 2 == 0 {
        if field_x == params.layer_width - 1 {
            if field_y == params.layer_height - 1 {
                0
            } else {
                field_y + 1
            }
        } else {
            field_y
        }
    } else {
        if field_x == 0 {
            if field_y == params.layer_height - 1 {
                0
            } else {
                field_y + 1
            }
        } else {
            field_y
        }
    };

    return (layer_2_to_1_x, layer_2_to_1_y);
}
