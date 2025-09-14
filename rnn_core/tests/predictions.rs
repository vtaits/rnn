use rnn_core::{LayerParams, Network, SynapseParams};
use rstest::rstest;

#[rstest]
fn identical_prediction_small_network(
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
    #[values(1, 2)] layer_width: usize,
    #[values(1, 2)] layer_height: usize,
) {
    let mut network = Network::new(
        LayerParams {
            field_width: 2,
            field_height: 4,
            layer_width,
            layer_height,
            partitions: None,
        },
        SynapseParams {
            alpha: 3.0,
            gamma_dec: 0.5,
            gamma_inc: 0.5,
            g_dec: 0.0,
            g_inc: 0.0,
            g_0: 1.0,
            min_g: -10.0,
            max_g: 10.0,
            h: 3.0,
            refract_interval: 2,
            threshold_train: 0.9,
            threshold_predict_min: 0.8,
            threshold_predict_max: 0.9,
            signal_shift_interval: 0,
            signal_rest_shift_limit: Some(1),
            signal_copy_shifts: None,
            excite_neuron_limit: 0.8,
        },
        None,
    );

    let result = network.predict(&bits, 0);

    assert_eq!(result, bits);
}

#[rstest]
fn identical_prediction_big_network(
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
    #[values(2, 3, 4)] layer_width: usize,
    #[values(2, 3, 4)] layer_height: usize,
) {
    let mut network = Network::new(
        LayerParams {
            field_width: 2,
            field_height: 4,
            layer_width,
            layer_height,
            partitions: None,
        },
        SynapseParams {
            alpha: 3.0,
            gamma_dec: 0.5,
            gamma_inc: 0.5,
            g_dec: 0.0,
            g_inc: 0.0,
            g_0: 1.0,
            min_g: -10.0,
            max_g: 10.0,
            h: 3.0,
            refract_interval: 2,
            threshold_train: 0.9,
            threshold_predict_min: 0.8,
            threshold_predict_max: 0.9,
            signal_shift_interval: 2,
            signal_rest_shift_limit: Some(1),
            signal_copy_shifts: Some(vec![(1, 0)]),
            excite_neuron_limit: 0.8,
        },
        None,
    );

    let result = network.predict(&bits, 0);

    assert_eq!(result, bits);
}

#[rstest]
#[case(
    vec![true, true, true, true, true, true],
    vec![true, false, true, false, true, true],
    2.0,
    1.2,
)]
#[case(
    vec![true, true, true, true, true, false],
    vec![true, false, true, false, true, false],
    1.4,
    1.1,
)]
fn restore_missed_bits(
    #[case] full: Vec<bool>,
    #[case] cut: Vec<bool>,
    #[case] alpha: f32,
    #[case] h: f32,
) {
    let mut network = Network::new(
        LayerParams {
            field_width: 3,
            field_height: 4,
            layer_width: 12,
            layer_height: 12,
            partitions: None,
        },
        SynapseParams {
            alpha,
            gamma_dec: 0.5,
            gamma_inc: 0.5,
            g_dec: 5.0,
            g_inc: 7.0,
            g_0: 1.0,
            min_g: -10.0,
            max_g: 10.0,
            h,
            refract_interval: 2,
            threshold_train: 0.8,
            threshold_predict_min: 0.8,
            threshold_predict_max: 0.9,
            signal_shift_interval: 3,
            signal_rest_shift_limit: Some(1),
            signal_copy_shifts: Some(vec![(1, 0)]),
            excite_neuron_limit: 0.8,
            // signal_copy_shifts: Some(vec![(1, 0), (0, 1), (-1, 0), (0, -1)]),
        },
        None,
    );

    network.push_data_and_apply(&full, 0);
    network.push_data_and_apply(&full, 0);

    let result = network.predict(&cut, 0);

    assert_eq!(result, full);
}

#[rstest]
#[case(
    vec![true, true, true, true, true, false],
    vec![true, false, true, false, true, false],
    10.0,
    2.0,
)]
#[case(
    vec![true, true, true, true, true, false],
    vec![false, false, false, false, true, true],
    1.0,
    3.0,
)]
fn not_restore_missed_bits(
    #[case] full: Vec<bool>,
    #[case] cut: Vec<bool>,
    #[case] alpha: f32,
    #[case] h: f32,
) {
    let mut network = Network::new(
        LayerParams {
            field_width: 3,
            field_height: 4,
            layer_width: 12,
            layer_height: 12,
            partitions: None,
        },
        SynapseParams {
            alpha,
            gamma_dec: 0.5,
            gamma_inc: 0.5,
            g_dec: 5.0,
            g_inc: 7.0,
            g_0: 1.0,
            min_g: -10.0,
            max_g: 10.0,
            h,
            refract_interval: 2,
            threshold_train: 0.9,
            threshold_predict_min: 0.8,
            threshold_predict_max: 0.9,
            signal_shift_interval: 3,
            signal_rest_shift_limit: Some(1),
            signal_copy_shifts: Some(vec![(1, 0)]),
            excite_neuron_limit: 0.8,
        },
        None,
    );

    network.push_data_and_apply(&full, 0);

    let result = network.predict(&cut, 0);

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
            field_height: 8,
            layer_width: 4,
            layer_height: 4,
            partitions: None,
        },
        SynapseParams {
            alpha: 1.5,
            gamma_dec: 0.5,
            gamma_inc: 0.5,
            g_dec: 5.0,
            g_inc: 7.0,
            g_0: 1.0,
            min_g: -10.0,
            max_g: 10.0,
            h: 1.2,
            refract_interval: 2,
            threshold_train: 0.8,
            threshold_predict_min: 0.75,
            threshold_predict_max: 0.82,
            signal_shift_interval: 3,
            signal_rest_shift_limit: Some(1),
            signal_copy_shifts: Some(vec![(0, 1)]),
            excite_neuron_limit: 0.8,
        },
        None,
    );

    for sequence in sequences.iter() {
        network.push_data_and_apply(&sequence.0, 0);
    }

    for sequence in sequences.iter() {
        let result = network.predict(&sequence.1, 0);

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
        (
            vec![
                false, false, false, false,
                false, false, false, false,
                false, false, false, false,
                false, false, false, false,
            ],
            vec![
                false, false, false, false,
                false, false, false, false,
                false, false, false, false,
                false, false, false, false,
            ],
        ),
    ],
)]
fn restore_multiple_overlapping_sequences(#[case] sequences: Vec<(Vec<bool>, Vec<bool>)>) {
    let mut network = Network::new(
        LayerParams {
            field_width: 4,
            field_height: 8,
            layer_width: 12,
            layer_height: 12,
            partitions: None,
        },
        SynapseParams {
            alpha: 2.0,
            gamma_dec: 0.5,
            gamma_inc: 0.5,
            g_dec: 5.0,
            g_inc: 7.0,
            g_0: 1.0,
            min_g: -10.0,
            max_g: 10.0,
            h: 1.5,
            refract_interval: 2,
            threshold_train: 0.8,
            threshold_predict_min: 0.8,
            threshold_predict_max: 0.9,
            signal_shift_interval: 3,
            signal_rest_shift_limit: Some(1),
            signal_copy_shifts: Some(vec![(1, 0)]),
            excite_neuron_limit: 0.8,
        },
        None,
    );

    for sequence in sequences.iter() {
        network.push_data_and_apply(&sequence.0, 0);
    }

    for sequence in sequences.iter() {
        network.push_data_and_apply(&sequence.0, 0);
    }

    for sequence in sequences.iter() {
        let result = network.predict(&sequence.1, 0);

        assert_eq!(result, sequence.0);
    }
}
