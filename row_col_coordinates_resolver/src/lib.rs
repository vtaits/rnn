use rnn_architecture::NeuronCoordinatesResolver;

pub struct RowColCoordinatesResolver {
    field_width: usize,
    field_size: usize,
    layer_width: usize,
}

impl RowColCoordinatesResolver {
    pub fn new(layer_width: usize, field_width: usize, field_height: usize) -> Self {
        Self {
            field_width,
            field_size: field_width * field_height,
            layer_width,
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
        let index_in_field = self.field_width * neuron_in_field_y + neuron_in_field_x;

        let field_offset = (self.layer_width * field_y + field_x) * self.field_size;

        field_offset + index_in_field
    }
}
