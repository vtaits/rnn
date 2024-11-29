use crate::LayerParams;

pub fn shift_signal(bit_vec: &[bool], field_size: usize, layer_params: &LayerParams, shift: &(i8, i8)) -> Vec<bool> {
  let mut res = vec![false; field_size];
  let field_width = layer_params.field_width;
  let field_height = layer_params.field_height;

  for (index, value) in bit_vec.iter().enumerate() {
    if *value {
      let y = index % field_width;
      let x = index / field_width;

      let next_x = x as i8 + shift.0;
      let next_y = y as i8 + shift.1;

      if next_x >= 0 && next_x < field_width as i8 && next_y >= 0 && next_y < field_height as i8 {
        let next_index = next_y as usize * field_width + next_x as usize;

        res[next_index] = true;
      }
    }
  }

  return res;
}
