pub fn excite_neurons_without_partitions(
    neurons: &mut Vec<u8>,
    signals: &Vec<f32>,
    threshold: f32,
) {
    for (index, value) in signals.iter().enumerate() {
        neurons[index] = if *value > threshold { 1 } else { 0 }
    }
}
