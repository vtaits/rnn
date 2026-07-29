pub fn get_weight_coefficient(
    gamma_inc: &f32,
    gamma_dec: &f32,
    synapse_weight: &f32,
    g_0: &f32,
) -> f32 {
    let shifted_synapse_weight = synapse_weight - g_0;

    if shifted_synapse_weight < 0.0 {
        return (gamma_dec * shifted_synapse_weight).exp() - 1.0;
    }

    return 1.0 - (-gamma_inc * shifted_synapse_weight).exp();
}
