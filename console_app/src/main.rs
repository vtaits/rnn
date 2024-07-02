extern crate rnn_core;

use rnn_core::{Network, NetworkParams};

fn number_to_bytes(number: usize, capacity: usize) -> Vec<bool> {
    let mut result = Vec::new();

    let mut temp_number = number;

    for _ in 0..capacity {
        result.push(if temp_number % 2 == 1 { true } else { false });

        temp_number = temp_number / 2;
    }

    result.reverse();

    return result;
}

fn print_bytes(bytes: &Vec<bool>) {
    for byte in bytes.into_iter() {
        print!("{}", if *byte { "+" } else { "." });
    }
    print!("\n");
}

fn main() {
    let params = NetworkParams {
        field_width: 5,
        field_height: 4,
        layer_width: 10,
        layer_height: 10,
    };

    let capacity = params.field_width * params.field_height;

    let mut network = Network::new(&params);

    let numbers = vec![2, 3, 5, 7, 11, 13, 23, 31, 47, 97, 113];

    for number in numbers {
        let bytes = number_to_bytes(number, capacity);

        print!("{}\n", number);

        print_bytes(&bytes);

        network.tick(&bytes);

        network.print_states();
    }
}
