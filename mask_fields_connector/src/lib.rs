use rnn_architecture::{FieldsConnector, Memory, NeuronCoordinatesResolver};

pub struct MaskFieldsConnector<'a> {
    field_width: usize,
    field_height: usize,
    memory: &'a dyn Memory,
    neuron_coordinates_resolver: &'a dyn NeuronCoordinatesResolver,
}

impl<'a> MaskFieldsConnector<'a> {
    fn new(
        field_width: usize,
        field_height: usize,
        memory: &'a dyn Memory,
        neuron_coordinates_resolver: &'a dyn NeuronCoordinatesResolver,
    ) -> Self {
        Self {
            field_width,
            field_height,
            memory,
            neuron_coordinates_resolver,
        }
    }
}

impl<'a> FieldsConnector for MaskFieldsConnector<'a> {
    fn connect_1_to_2(
        &self,
        field_1_x: usize,
        field_1_y: usize,
        field_2_x: usize,
        field_2_y: usize,
    ) {
    }

    fn connect_2_to_1(
        &self,
        field_1_x: usize,
        field_1_y: usize,
        field_2_x: usize,
        field_2_y: usize,
    ) {
    }
}
