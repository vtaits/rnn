use rnn_architecture::{BlockSequence, FieldsConnector, Memory, Topology};

pub struct TopologyRestoreFirst();

impl TopologyRestoreFirst {
    pub fn new() -> Self {
        Self {}
    }
}

impl Topology for TopologyRestoreFirst {
    fn fill(
        &self,
        block_sequence: Box<dyn BlockSequence>,
        fields_connector: &mut dyn FieldsConnector,
        memory: &mut dyn Memory,
    ) {
        let mut prev_x = 0;
        let mut prev_y = 0;
        let mut is_first = true;

        for (field_x, field_y) in block_sequence.as_ref().iterate_blocks() {
            fields_connector.connect_1_to_2(memory, field_x, field_y, field_x, field_y, 0, 0);

            if is_first {
                is_first = false;
            } else {
                let shift_x = field_x as i32 - prev_x as i32;
                let shift_y = field_y as i32 - prev_y as i32;

                fields_connector.connect_2_to_1(memory, prev_x, prev_y, 0, 0, shift_x, shift_y);
                fields_connector
                    .connect_2_to_1(memory, prev_x, prev_y, field_x, field_y, shift_x, shift_y);
            }

            prev_x = field_x;
            prev_y = field_y;
        }

        fields_connector.connect_2_to_1(memory, prev_x, prev_y, 0, 0, 0, 0);
    }
}
