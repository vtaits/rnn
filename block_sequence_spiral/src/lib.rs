use rnn_architecture::BlockSequence;

struct Coords {
    layer_width: usize,
    layer_height: usize,
    is_start: bool,
    last_x: usize,
    last_y: usize,
    x: usize,
    y: usize,
}

impl Coords {
    fn new(layer_width: usize, layer_height: usize) -> Self {
        Self {
            layer_width,
            layer_height,
            is_start: true,
            last_x: if layer_height % 2 == 0 {
                0
            } else {
                layer_width - 1
            },
            last_y: layer_height - 1,
            x: 0,
            y: 0,
        }
    }
}

impl Iterator for Coords {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.is_start {
            self.is_start = false;
            return Some((0, 0));
        }

        if self.x == self.last_x && self.y == self.last_y {
            return None;
        }

        let next_x = if self.y % 2 == 0 {
            // last field in row
            if self.x == self.layer_width - 1 {
                if self.y == self.layer_height - 1 {
                    0
                } else {
                    self.x
                }
            } else {
                self.x + 1
            }
        } else {
            // first field in row
            if self.x == 0 {
                self.x
            } else {
                self.x - 1
            }
        };

        let next_y = if self.y % 2 == 0 {
            if self.x == self.layer_width - 1 {
                if self.y == self.layer_height - 1 {
                    0
                } else {
                    self.y + 1
                }
            } else {
                self.y
            }
        } else if self.x == 0 {
            if self.y == self.layer_height - 1 {
                0
            } else {
                self.y + 1
            }
        } else {
            self.y
        };

        self.x = next_x;
        self.y = next_y;

        Some((self.x, self.y))
    }
}

pub struct BlockSequenceSpiral {
    layer_width: usize,
    layer_height: usize,
}

impl BlockSequenceSpiral {
    pub fn new(layer_width: usize, layer_height: usize) -> Self {
        Self {
            layer_width,
            layer_height,
        }
    }
}

impl BlockSequence for BlockSequenceSpiral {
    fn iterate_blocks(&self) -> Box<dyn Iterator<Item = (usize, usize)>> {
        let coords = Coords::new(self.layer_width, self.layer_height);

        Box::new(coords.into_iter())
    }
}
