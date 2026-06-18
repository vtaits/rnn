pub struct BlockSequenceSpiral {
    layer_width: usize,
    layer_height: usize,
}

impl BlockSequenceSpiral {
    fn new(layer_width: usize, layer_height: usize) -> Self {
        Self {
            layer_width,
            layer_height,
        }
    }
}

impl BlockSequence for BlockSequenceSpiral {
    fn fill(&self, memory: Memory) {}
}
