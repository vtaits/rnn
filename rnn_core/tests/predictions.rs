use rnn_core::{LayerParams, Network, SynapseParams};
use rstest::rstest;

#[rstest]
fn identical_prediction(
    #[values(
        vec![true, false, false, false],
        vec![false, true, false, false],
        vec![false, false, true, false],
        vec![false, false, false, true],
        vec![true, true, false, false],
        vec![false, true, true, false],
        vec![false, false, true, true],
        vec![false, true, false, true],
    )]
    bits: Vec<bool>,
    #[values(1, 2, 3, 4)] layer_width: usize,
    #[values(1, 2, 3, 4)] layer_height: usize,
) {
    let mut network = Network::new(
        LayerParams {
            field_width: 2,
            field_height: 2,
            layer_width,
            layer_height,
        },
        SynapseParams {
            alpha: 3.0,
            gamma: 0.5,
            g_dec: 0.0,
            g_inc: 0.0,
            g_0: 1.0,
            max_g: 10.0,
            initial_strong_g: 7.0,
            h: 3,
            refract_interval: 3,
            threshold: 0.9,
            signal_shift_interval: 2,
            signal_rest_shift_limit: Some(0),
        },
        None,
    );

    let result = network.predict(&bits);

    assert_eq!(result, bits);
}
