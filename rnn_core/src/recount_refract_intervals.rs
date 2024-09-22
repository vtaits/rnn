use ndarray::Array1;

pub fn recount_refract_intervals(
    neurons: &Array1<f32>,
    refract_intervals: &Array1<u8>,
    intial_refract_interval: &u8,
) -> Array1<u8> {
    neurons
        .iter()
        .zip(refract_intervals.iter())
        .map(|(neuron, refract_interval)| {
            if *neuron > 0.5 {
                return *intial_refract_interval;
            }

            if *refract_interval > 0 {
                return *refract_interval - 1;
            }

            0
        })
        .collect()
}
