use rnn_architecture::NeuronCoordinatesResolver;

pub struct RowColCoordinatesResolver {
    field_width: usize,
    field_height: usize,
    row_width: usize,
}

impl RowColCoordinatesResolver {
    pub fn new(layer_width: usize, field_width: usize, field_height: usize) -> Self {
        Self {
            field_width,
            field_height,
            row_width: field_width * layer_width,
        }
    }
}

impl NeuronCoordinatesResolver for RowColCoordinatesResolver {
    fn resolve(
        &self,
        field_x: usize,
        field_y: usize,
        neuron_in_field_x: usize,
        neuron_in_field_y: usize,
    ) -> usize {
        let x = self.field_width * field_x + neuron_in_field_x;
        let y = self.field_height * field_y + neuron_in_field_y;

        self.row_width * y + x
    }
}
