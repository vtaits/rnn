use rnn_architecture::{
    DistanceBetweenNeurons, FieldsConnector, Memory, NeuronCoordinatesResolver,
};

pub struct FullFieldsConnector {
    field_width: usize,
    field_height: usize,
    max_weight: f32,
    distance_between_neurons: Box<dyn DistanceBetweenNeurons>,
    neuron_coordinates_resolver: Box<dyn NeuronCoordinatesResolver>,
}

impl<'a> FullFieldsConnector {
    pub fn new(
        field_width: usize,
        field_height: usize,
        max_weight: f32,
        distance_between_neurons: Box<dyn DistanceBetweenNeurons>,
        neuron_coordinates_resolver: Box<dyn NeuronCoordinatesResolver>,
    ) -> Self {
        Self {
            field_width,
            field_height,
            max_weight,
            distance_between_neurons,
            neuron_coordinates_resolver,
        }
    }
}

impl FieldsConnector for FullFieldsConnector {
    fn connect_1_to_2(
        &mut self,
        memory: &mut dyn Memory,
        field_1_x: usize,
        field_1_y: usize,
        field_2_x: usize,
        field_2_y: usize,
        shift_x: i32,
        shift_y: i32,
    ) {
        let offset_1_x = field_1_x * self.field_width;
        let offset_1_y: usize = field_1_y * self.field_height;

        let offset_2_x = (field_2_x as i32 - shift_x) * self.field_width as i32;
        let offset_2_y = (field_2_y as i32 - shift_y) * self.field_height as i32;

        for neuron_in_field_1_x in 0..self.field_width {
            let neuron_1_x = (offset_1_x + neuron_in_field_1_x) as i32;

            for neuron_in_field_1_y in 0..self.field_height {
                let neuron_1_y = (offset_1_y + neuron_in_field_1_y) as i32;

                let neuron_1_index = self.neuron_coordinates_resolver.resolve(
                    field_1_x,
                    field_1_y,
                    neuron_in_field_1_x,
                    neuron_in_field_1_y,
                );

                for neuron_in_field_2_x in 0..self.field_width {
                    let neuron_2_x = offset_2_x + neuron_in_field_2_x as i32;

                    for neuron_in_field_2_y in 0..self.field_height {
                        let neuron_2_y = offset_2_y + neuron_in_field_2_y as i32;

                        let neuron_2_index = self.neuron_coordinates_resolver.resolve(
                            field_2_x,
                            field_2_y,
                            neuron_in_field_2_x,
                            neuron_in_field_2_y,
                        );

                        let dx = neuron_2_x - neuron_1_x;
                        let dy = neuron_2_y - neuron_1_y;

                        let distance = self.distance_between_neurons.get_distance(dx, dy);

                        memory.set_distance_1_to_2(neuron_1_index, neuron_2_index, distance);

                        if dx == 0 && dy == 0 {
                            memory.set_synapse_weight_1_to_2(
                                neuron_1_index,
                                neuron_2_index,
                                self.max_weight,
                            );
                        }
                    }
                }
            }
        }
    }

    fn connect_2_to_1(
        &mut self,
        memory: &mut dyn Memory,
        field_2_x: usize,
        field_2_y: usize,
        field_1_x: usize,
        field_1_y: usize,
        shift_x: i32,
        shift_y: i32,
    ) {
        let offset_2_x = field_2_x * self.field_width;
        let offset_2_y = field_2_y * self.field_height;

        let offset_1_x = (field_1_x as i32 - shift_x) * self.field_width as i32;
        let offset_1_y = (field_1_y as i32 - shift_y) * self.field_height as i32;

        for neuron_in_field_2_x in 0..self.field_width {
            let neuron_2_x = (offset_2_x + neuron_in_field_2_x) as i32;

            for neuron_in_field_2_y in 0..self.field_height {
                let neuron_2_y = (offset_2_y + neuron_in_field_2_y) as i32;

                let neuron_2_index = self.neuron_coordinates_resolver.resolve(
                    field_2_x,
                    field_2_y,
                    neuron_in_field_2_x,
                    neuron_in_field_2_y,
                );

                for neuron_in_field_1_x in 0..self.field_width {
                    let neuron_1_x = offset_1_x + neuron_in_field_1_x as i32;

                    for neuron_in_field_1_y in 0..self.field_height {
                        let neuron_1_y = offset_1_y + neuron_in_field_1_y as i32;

                        let neuron_1_index = self.neuron_coordinates_resolver.resolve(
                            field_1_x,
                            field_1_y,
                            neuron_in_field_1_x,
                            neuron_in_field_1_y,
                        );

                        let dx = neuron_1_x - neuron_2_x;
                        let dy = neuron_1_y - neuron_2_y;

                        let distance = self.distance_between_neurons.get_distance(dx, dy);

                        memory.set_distance_2_to_1(neuron_2_index, neuron_1_index, distance);

                        if dx == 0 && dy == 0 {
                            memory.set_synapse_weight_2_to_1(
                                neuron_2_index,
                                neuron_1_index,
                                self.max_weight,
                            );
                        }
                    }
                }
            }
        }
    }
}
