use rnn_architecture::{BlockSequence, FieldsConnector, Topology};

pub struct TopologyRestoreFirst();

impl Topology for TopologyRestoreFirst {
    fn fill(
        &self,
        block_sequence: Box<dyn BlockSequence>,
        mut fields_connector: Box<dyn FieldsConnector>,
    ) {
        let fields_connector = fields_connector.as_mut();

        let mut prev_x = 0;
        let mut prev_y = 0;
        let mut is_first = true;

        for (field_x, field_y) in block_sequence.as_ref().iterate_blocks() {
            fields_connector.connect_1_to_2(field_x, field_y, field_x, field_y, 0, 0);
            fields_connector.connect_2_to_1(field_x, field_y, 0, 0, 0, 0);

            if is_first {
                is_first = false;
            } else {
                fields_connector.connect_2_to_1(
                    prev_y,
                    prev_x,
                    field_x,
                    field_y,
                    field_x as i32 - prev_x as i32,
                    field_y as i32 - prev_y as i32,
                );
            }

            prev_x = field_x;
            prev_y = field_y;
        }
    }
}
