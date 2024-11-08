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

#[rstest]
#[case(
    vec![true, true, true, true, true, true],
    vec![true, false, true, false, true, true],
    3.0,
)]
#[case(
    vec![true, true, true, true, true, false],
    vec![true, false, true, false, true, false],
    2.0,
)]
fn restore_missed_bits(#[case] full: Vec<bool>, #[case] cut: Vec<bool>, #[case] alpha: f32) {
    let mut network = Network::new(
        LayerParams {
            field_width: 3,
            field_height: 2,
            layer_width: 2,
            layer_height: 2,
        },
        SynapseParams {
            alpha,
            gamma: 0.5,
            g_dec: 0.0,
            g_inc: 10.0,
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

    network.push_data_binary(&full);

    let result = network.predict(&cut);

    assert_eq!(result, full);
}

#[rstest]
#[case(
    vec![true, true, true, true, true, false],
    vec![true, false, true, false, true, false],
    3.0,
)]
#[case(
    vec![true, true, true, true, true, false],
    vec![false, false, false, false, true, true],
    1.0,
)]
fn not_restore_missed_bits(#[case] full: Vec<bool>, #[case] cut: Vec<bool>, #[case] alpha: f32) {
    let mut network = Network::new(
        LayerParams {
            field_width: 3,
            field_height: 2,
            layer_width: 2,
            layer_height: 2,
        },
        SynapseParams {
            alpha,
            gamma: 0.5,
            g_dec: 0.0,
            g_inc: 10.0,
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

    network.push_data_binary(&full);

    let result = network.predict(&cut);

    assert_eq!(result, cut);
}

#[rstest]
#[case(
    vec![
        (
            vec![
                true, true, true, true,
                false, false, false, false,
                false, false, false, false,
                false, false, false, false,
            ],
            vec![
                true, false, true, true,
                false, false, false, false,
                false, false, false, false,
                false, false, false, false,
            ],
        ),
        (
            vec![
                false, false, false, false,
                true, true, true, true,
                false, false, false, false,
                false, false, false, false,
            ],
            vec![
                false, false, false, false,
                true, false, true, true,
                false, false, false, false,
                false, false, false, false,
            ],
        ),
        (
            vec![
                false, false, false, false,
                false, false, false, false,
                true, true, true, true,
                false, false, false, false,
            ],
            vec![
                false, false, false, false,
                false, false, false, false,
                true, false, true, true,
                false, false, false, false,
            ],
        ),
        (
            vec![
                false, false, false, false,
                false, false, false, false,
                false, false, false, false,
                true, true, true, true,
            ],
            vec![
                false, false, false, false,
                false, false, false, false,
                false, false, false, false,
                true, false, true, true,
            ],
        ),
    ],
)]
fn restore_multiple_separated_sequences(#[case] sequences: Vec<(Vec<bool>, Vec<bool>)>) {
    let mut network = Network::new(
        LayerParams {
            field_width: 4,
            field_height: 4,
            layer_width: 3,
            layer_height: 3,
        },
        SynapseParams {
            alpha: 2.0,
            gamma: 0.5,
            g_dec: 0.0,
            g_inc: 5.0,
            g_0: 1.0,
            max_g: 10.0,
            initial_strong_g: 7.0,
            h: 3,
            refract_interval: 3,
            threshold: 0.8,
            signal_shift_interval: 2,
            signal_rest_shift_limit: Some(0),
        },
        None,
    );

    for sequence in sequences.iter() {
        network.push_data_binary(&sequence.0);
    }

    for sequence in sequences.iter() {
        network.push_data_binary(&sequence.0);
    }

    for sequence in sequences.iter() {
        println!("PREDICT");
        let result = network.predict(&sequence.1);

        assert_eq!(result, sequence.0);
    }
}

#[rstest]
#[case(
    vec![
        (
            vec![
                true, true, true, true,
                true, true, true, true,
                false, false, false, false,
                false, false, false, false,
            ],
            vec![
                true, false, true, true,
                false, true, true, true,
                false, false, false, false,
                false, false, false, false,
            ],
        ),
        (
            vec![
                false, false, false, false,
                true, true, true, true,
                true, true, true, true,
                false, false, false, false,
            ],
            vec![
                false, false, false, false,
                true, false, true, true,
                false, true, true, true,
                false, false, false, false,
            ],
        ),
        (
            vec![
                false, false, false, false,
                false, false, false, false,
                true, true, true, true,
                true, true, true, true,
            ],
            vec![
                false, false, false, false,
                false, false, false, false,
                true, false, true, true,
                false, true, true, true,
            ],
        ),
    ],
)]
fn restore_multiple_overlapping_sequences(#[case] sequences: Vec<(Vec<bool>, Vec<bool>)>) {
    let mut network = Network::new(
        LayerParams {
            field_width: 4,
            field_height: 4,
            layer_width: 3,
            layer_height: 3,
        },
        SynapseParams {
            alpha: 2.0,
            gamma: 0.5,
            g_dec: 0.0,
            g_inc: 5.0,
            g_0: 1.0,
            max_g: 10.0,
            initial_strong_g: 7.0,
            h: 3,
            refract_interval: 3,
            threshold: 0.8,
            signal_shift_interval: 2,
            signal_rest_shift_limit: Some(0),
        },
        None,
    );

    for sequence in sequences.iter() {
        network.push_data_binary(&sequence.0);
    }

    for sequence in sequences.iter() {
        network.push_data_binary(&sequence.0);
    }

    for sequence in sequences.iter() {
        let result = network.predict(&sequence.1);

        assert_eq!(result, sequence.0);
    }
}
