use rnn_architecture::{BlockSequence, Memory, Topology};

pub struct TopologySpiral {
    field_width: usize,
    field_height: usize,
    layer_width: usize,
    layer_height: usize,
    field_size: usize,
    field_count: usize,
    layer_size: usize,
    initial_strong_connection: f32,
}

impl TopologySpiral {
    fn new(
        field_width: usize,
        field_height: usize,
        layer_width: usize,
        layer_height: usize,
        initial_strong_connection: f32,
    ) -> Self {
        Self {
            field_width,
            field_height,
            layer_width,
            layer_height,
            field_size: field_width * field_height,
            field_count: layer_width * layer_height,
            layer_size: field_width * field_height * layer_width * layer_height,
            initial_strong_connection,
        }
    }
}

impl Topology for TopologySpiral {
    fn fill(&self, block_sequence: Box<dyn BlockSequence>, memory: Box<dyn Memory>) {
        let mut prev_x = 0;
        let mut prev_y = 0;
        let mut is_first = true;

        for (field_x, field_y) in block_sequence.as_ref().iterate_blocks() {
            if is_first {
                is_first = false;
            }

            prev_x = field_x;
            prev_y = field_y;
        }
    }
}
