use rnn_core::{LayerParams, Network, SynapseParams};

#[test]
fn identical_prediction() {
    let mut network = Network::new(
        LayerParams {
            field_width: 2,
            field_height: 2,
            layer_width: 1,
            layer_height: 1,
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

    let result = network.predict(&[true, false, false, false]);

    assert_eq!(result, vec![true, false, false, false]);
}
