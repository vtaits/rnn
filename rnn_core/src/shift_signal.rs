use crate::LayerParams;

pub fn shift_signal(
    bit_vec: &[bool],
    field_size: usize,
    layer_params: &LayerParams,
    shift: &(i8, i8),
) -> Vec<bool> {
    let mut res = vec![false; field_size];
    let field_width = layer_params.field_width;
    let field_height = layer_params.field_height;

    for (index, value) in bit_vec.iter().enumerate() {
        if *value {
            let x = index % field_width;
            let y = index / field_width;

            let next_x = x as i8 + shift.0;
            let next_y = y as i8 + shift.1;

            if next_x >= 0
                && next_x < field_width as i8
                && next_y >= 0
                && next_y < field_height as i8
            {
                let next_index = next_y as usize * field_width + next_x as usize;

                if bit_vec.len() <= next_index || !bit_vec[next_index] {
                    res[next_index] = true;
                }
            }
        }
    }

    return res;
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case(
        4,
        4,
        vec![
            true, true, true, true,
            false, false, false, false,
            false, false, false, false,
            false, false, false, false,
        ],
        vec![
            false, false, false, false,
            false, false, false, false,
            false, false, false, false,
            false, false, false, false,
        ],
        (1, 0),
    )]
    #[case(
        4,
        4,
        vec![
            true, true, true, true,
            false, false, false, false,
            false, false, false, false,
            false, false, false, false,
        ],
        vec![
            false, false, false, false,
            false, false, false, false,
            false, false, false, false,
            false, false, false, false,
        ],
        (-1, 0),
    )]
    #[case(
        4,
        4,
        vec![
            true, true, true, true,
            false, false, false, false,
            false, false, false, false,
            false, false, false, false,
        ],
        vec![
            false, false, false, false,
            true, true, true, true,
            false, false, false, false,
            false, false, false, false,
        ],
        (0, 1),
    )]
    #[case(
        4,
        4,
        vec![
            true, true, true, true,
            false, false, false, false,
            false, false, false, false,
            false, false, false, false,
        ],
        vec![
            false, false, false, false,
            false, false, false, false,
            false, false, false, false,
            false, false, false, false,
        ],
        (0, -1),
    )]
    fn shift_signal_check(
        #[case] field_width: usize,
        #[case] field_height: usize,
        #[case] bit_vec: Vec<bool>,
        #[case] expected_res: Vec<bool>,
        #[case] shift: (i8, i8),
    ) {
        let res = shift_signal(
            &bit_vec,
            field_width * field_height,
            &LayerParams {
                field_height,
                field_width,
                layer_height: 1,
                layer_width: 1,
                partitions: None,
            },
            &shift,
        );

        assert_eq!(res, expected_res);
    }
}
