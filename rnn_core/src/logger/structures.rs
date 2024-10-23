pub enum LoggerEvent {
    // input a signal to the first field of the first layer
    Input(Vec<bool>),
    // layer index, increases, decreases
    ChangeLayerWeights(usize, u8, u8),
    // layer index, sum of accumulated weights
    LayerTotalWeight(usize, f32),
}
