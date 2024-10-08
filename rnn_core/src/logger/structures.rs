pub enum LoggerEvent {
    // layer index, increases, decreases
    ChangeLayerWeights(usize, u8, u8),
    // layer index, sum of accumulated weights
    LayerTotalWeight(usize, f32),
}
