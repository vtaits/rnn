pub fn recount_refract_intervals(
    neurons: &Vec<u8>,
    refract_intervals: &mut Vec<u8>,
    intial_refract_interval: &u8,
) {
    for (index, neuron) in neurons.iter().enumerate() {
        if *neuron > 0 {
            refract_intervals[index] = *intial_refract_interval;
            continue;
        }

        if refract_intervals[index] > 0 {
            refract_intervals[index] -= 1;
        }
    }
}
