pub struct TopologySpiral {
    field_width: usize,
    field_height: usize,
    layer_width: usize,
    layer_height: usize,
    field_size: usize,
}

impl TopologySpiral {
    fn new(
        field_width: usize,
        field_height: usize,
        layer_width: usize,
        layer_height: usize,
    ) -> Self {
        Self {
            field_width: usize,
            field_height: usize,
            layer_width: usize,
            layer_height: usize,
            field_size: field_width * field_height,
        }
    }

    fn get_next_field(&self, field_x: usize, field_y: usize) -> (usize, usize) {
        let layer_2_to_1_x = if field_y % 2 == 0 {
            // last field in row
            if field_x == self.layer_width - 1 {
                if field_y == self.layer_height - 1 {
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
            if field_x == self.layer_width - 1 {
                if field_y == self.layer_height - 1 {
                    0
                } else {
                    field_y + 1
                }
            } else {
                field_y
            }
        } else if field_x == 0 {
            if field_y == self.layer_height - 1 {
                0
            } else {
                field_y + 1
            }
        } else {
            field_y
        };

        (layer_2_to_1_x, layer_2_to_1_y)
    }

    fn get_last_field(&self) -> (usize, usize) {
        (
            if self.layer_height % 2 == 0 {
                0
            } else {
                self.layer_width - 1
            },
            self.layer_height - 1,
        )
    }
}

impl Topology for TopologySpiral {
    fn fill(&self, memory: Memory) {}
}
