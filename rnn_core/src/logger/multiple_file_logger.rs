use std::io::Write;
use std::{fs::File, path::Path};

use super::{Logger, LoggerEvent};

pub struct MultipleFileLogger {
    weights_diff_file_1: Option<File>,
    weights_diff_file_2: Option<File>,
    sum_file_1: Option<File>,
    sum_file_2: Option<File>,
    count_file: Option<File>,
    count: Vec<usize>,
}

pub struct MultipleFileLoggerParams<
    P1: AsRef<Path>,
    P2: AsRef<Path>,
    P3: AsRef<Path>,
    P4: AsRef<Path>,
    P5: AsRef<Path>,
> {
    pub weights_diff_path_1: Option<P1>,
    pub weights_diff_path_2: Option<P2>,
    pub sum_path_1: Option<P3>,
    pub sum_path_2: Option<P4>,
    pub count_path: Option<P5>,
}

impl MultipleFileLogger {
    pub fn new<
        P1: AsRef<Path>,
        P2: AsRef<Path>,
        P3: AsRef<Path>,
        P4: AsRef<Path>,
        P5: AsRef<Path>,
    >(
        params: MultipleFileLoggerParams<P1, P2, P3, P4, P5>,
        capacity: usize,
    ) -> Self {
        let MultipleFileLoggerParams {
            weights_diff_path_1,
            weights_diff_path_2,
            sum_path_1,
            sum_path_2,
            count_path,
        } = params;

        let weights_diff_file_1 = weights_diff_path_1.map(|path| File::create(path).unwrap());
        let weights_diff_file_2 = weights_diff_path_2.map(|path| File::create(path).unwrap());
        let sum_file_1 = sum_path_1.map(|path| File::create(path).unwrap());
        let sum_file_2 = sum_path_2.map(|path| File::create(path).unwrap());
        let count_file = count_path.map(|path| File::create(path).unwrap());

        MultipleFileLogger {
            weights_diff_file_1,
            weights_diff_file_2,
            sum_file_1,
            sum_file_2,
            count_file,
            count: vec![0; capacity],
        }
    }
}

impl Logger for MultipleFileLogger {
    fn log_event(&mut self, logger_event: LoggerEvent) {
        match logger_event {
            LoggerEvent::ChangeLayerWeights(layer_index, inc, dec) => {
                let weights_diff_file = if layer_index == 1 {
                    &mut self.weights_diff_file_1
                } else {
                    &mut self.weights_diff_file_2
                };

                if let Some(weights_diff_file) = weights_diff_file {
                    let _ = writeln!(weights_diff_file, "{} {}", inc, dec);
                }
            }
            LoggerEvent::LayerTotalWeight(layer_index, value) => {
                let sum_file = if layer_index == 1 {
                    &mut self.sum_file_1
                } else {
                    &mut self.sum_file_2
                };

                if let Some(sum_file) = sum_file {
                    let _ = writeln!(sum_file, "{}", value);
                }
            }
            LoggerEvent::Input(signals) => {
                if let Some(count_file) = &mut self.count_file {
                    for (index, value) in signals.iter().enumerate() {
                        if *value {
                            self.count[index] += 1;
                        }
                    }

                    for value in self.count.iter() {
                        let _ = write!(count_file, "{} ", value);
                    }

                    let _ = writeln!(count_file);
                }
            }
        };
    }
}
