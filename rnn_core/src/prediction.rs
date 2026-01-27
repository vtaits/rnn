use crate::structures::InputPhase;

struct PredictionProcessingTickSplit {
    cur_index: usize,
    input_phase: InputPhase,
}

pub struct PredictionProcessingTick {
    results: Vec<Vec<bool>>,
    tick_splits: Vec<PredictionProcessingTickSplit>,
}

pub struct PredictionProcessing {
    prediction_depth: usize,
    split_count: usize,
    ticks: Vec<PredictionProcessingTick>,
    input_index: usize,
    collect_prediction_index: usize,
    collect_tick_index: Option<usize>,
    prediction_size: usize,
    output_field_index: usize,
}

impl PredictionProcessing {
    pub fn new(
        split_count: usize,
        field_size: usize,
        output_field_index: usize,
        prediction_depth: usize,
    ) -> Self {
        PredictionProcessing {
            prediction_depth,
            split_count,
            ticks: vec![],
            input_index: 0,
            collect_prediction_index: 0,
            collect_tick_index: Some(0),
            prediction_size: field_size / 2,
            output_field_index,
        }
    }

    pub fn add_tick(&mut self, index: usize) {
        self.input_index = index;

        if index < self.ticks.len() {
            self.ticks[index].tick_splits = vec![];
            return;
        }

        if self.ticks.len() == self.split_count {
            panic!(
                "Number of splits of prediction is {}, cannot add more",
                self.split_count
            );
        }

        self.ticks.push(PredictionProcessingTick {
            tick_splits: vec![],
            results: vec![vec![false; self.prediction_size]; self.prediction_depth + 1],
        });

        if self.collect_tick_index.is_none() {
            self.collect_tick_index = Some(index);
        }
    }

    pub fn shift(&mut self) {
        let output_field_index = self.output_field_index;

        for tick in self.ticks.iter_mut() {
            for tick_split in tick.tick_splits.iter_mut() {
                if output_field_index == tick_split.cur_index {
                    panic!("Signal have not been read");
                }

                tick_split.cur_index += 1;
            }
        }
    }

    pub fn add_tick_split(&mut self, input_phase: InputPhase) {
        self.shift();

        self.ticks[self.input_index]
            .tick_splits
            .push(PredictionProcessingTickSplit {
                cur_index: 0,
                input_phase,
            });
    }

    pub fn should_read(&self) -> bool {
        match self.collect_tick_index {
            Some(collect_tick_index) => {
                if self.ticks[collect_tick_index].tick_splits.len() == 0 {
                    return false;
                }

                self.ticks[collect_tick_index].tick_splits[0].cur_index == self.output_field_index
            }
            None => false,
        }
    }

    pub fn read(&mut self, last_field_state: &[u8]) {
        match self.collect_tick_index {
            Some(collect_tick_index) => {
                let input_phase = self.ticks[collect_tick_index].tick_splits[0].input_phase;

                for pos in 0..self.prediction_size {
                    let neuron_index = match input_phase {
                        InputPhase::Even => pos * 2,
                        InputPhase::Odd => pos * 2 + 1,
                    };

                    let value = last_field_state[neuron_index];

                    // uncomment to enable logging
                    // print!("{}", if value > 0 { "+" } else { "." });

                    if value > 0 {
                        self.ticks[collect_tick_index].results[self.collect_prediction_index]
                            [pos] = true;
                    }
                }

                // uncomment to enable logging
                // println!();

                self.ticks[collect_tick_index].tick_splits.remove(0);

                if self.ticks[collect_tick_index].tick_splits.len() == 0 {
                    let is_last_split = collect_tick_index == self.ticks.len() - 1;

                    if is_last_split {
                        self.collect_prediction_index += 1;
                        self.input_index = 0;

                        if self.collect_prediction_index > self.prediction_depth {
                            self.collect_tick_index = None;
                        } else {
                            self.collect_tick_index = Some(0);
                        }
                    } else {
                        self.collect_tick_index = Some(collect_tick_index + 1);
                    }
                }
            }
            None => {}
        }
    }

    pub fn is_finished(&self) -> bool {
        self.collect_tick_index.is_none()
    }

    pub fn get_prediction(&self) -> Vec<Vec<bool>> {
        let mut res = vec![];

        for prediction_index in 0..self.prediction_depth + 1 {
            let mut res_item = vec![];

            for tick in self.ticks.iter() {
                for byte in tick.results[prediction_index].iter() {
                    res_item.push(*byte);
                }
            }

            res.push(res_item);
        }

        res
    }

    pub fn is_all_ticks_added(&self) -> bool {
        self.ticks.len() == self.split_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_prediction_flow() {
        let mut prediction = PredictionProcessing::new(2, 10, 9, 0);

        prediction.add_tick(0); // []

        assert!(!prediction.should_read());
        assert!(!prediction.is_finished());
        assert!(!prediction.is_all_ticks_added());

        prediction.add_tick_split(InputPhase::Even); // [1]

        assert!(!prediction.should_read());
        assert!(!prediction.is_finished());
        assert!(!prediction.is_all_ticks_added());

        prediction.shift(); // [2]

        assert!(!prediction.should_read());
        assert!(!prediction.is_finished());
        assert!(!prediction.is_all_ticks_added());

        prediction.shift(); // [3]

        prediction.add_tick_split(InputPhase::Odd); // [4, 1]

        assert!(!prediction.should_read());
        assert!(!prediction.is_finished());
        assert!(!prediction.is_all_ticks_added());

        prediction.shift(); // [5, 2]

        assert!(!prediction.should_read());
        assert!(!prediction.is_finished());
        assert!(!prediction.is_all_ticks_added());

        prediction.shift(); // [6, 3]

        assert!(!prediction.should_read());
        assert!(!prediction.is_finished());
        assert!(!prediction.is_all_ticks_added());

        prediction.add_tick(1); // [6, 3] []

        assert!(!prediction.should_read());
        assert!(!prediction.is_finished());
        assert!(prediction.is_all_ticks_added());

        prediction.add_tick_split(InputPhase::Even); // [7, 4] [1]

        assert!(!prediction.should_read());
        assert!(!prediction.is_finished());
        assert!(prediction.is_all_ticks_added());

        prediction.shift(); // [8, 5] [2]

        assert!(!prediction.should_read());
        assert!(!prediction.is_finished());
        assert!(prediction.is_all_ticks_added());

        prediction.shift(); // [9, 6] [3]

        assert!(!prediction.should_read());
        assert!(!prediction.is_finished());
        assert!(prediction.is_all_ticks_added());

        prediction.shift(); // [10, 7] [4]

        assert!(prediction.should_read());
        assert!(!prediction.is_finished());
        assert!(prediction.is_all_ticks_added());

        prediction.read(&[1, 0, 0, 0, 1, 0, 0, 0, 0, 0]); // [7] [4]

        assert!(!prediction.should_read());
        assert!(!prediction.is_finished());
        assert!(prediction.is_all_ticks_added());

        prediction.shift(); // [8] [5]

        assert!(!prediction.should_read());
        assert!(!prediction.is_finished());
        assert!(prediction.is_all_ticks_added());

        prediction.shift(); // [9] [6]

        assert!(!prediction.should_read());
        assert!(!prediction.is_finished());
        assert!(prediction.is_all_ticks_added());

        prediction.shift(); // [10] [7]

        assert!(prediction.should_read());
        assert!(!prediction.is_finished());
        assert!(prediction.is_all_ticks_added());

        prediction.read(&[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]); // [] [7]

        assert!(!prediction.should_read());
        assert!(!prediction.is_finished());
        assert!(prediction.is_all_ticks_added());

        prediction.shift(); // [] [8]

        assert!(!prediction.should_read());
        assert!(!prediction.is_finished());
        assert!(prediction.is_all_ticks_added());

        prediction.shift(); // [] [9]

        assert!(!prediction.should_read());
        assert!(!prediction.is_finished());
        assert!(prediction.is_all_ticks_added());

        prediction.shift(); // [] [10]

        assert!(prediction.should_read());
        assert!(!prediction.is_finished());
        assert!(prediction.is_all_ticks_added());

        prediction.read(&[0, 0, 1, 0, 1, 0, 0, 0, 0, 0]); // [] []

        assert!(!prediction.should_read());
        assert!(prediction.is_finished());
        assert!(prediction.is_all_ticks_added());

        let result = prediction.get_prediction();

        assert_eq!(
            result,
            vec![vec![
                true, false, true, false, true, false, true, true, false, false
            ]],
            "30"
        );
    }

    #[test]
    fn multiple_prediction_flow() {
        let mut prediction = PredictionProcessing::new(2, 10, 9, 2);

        prediction.add_tick(0); // []
        prediction.add_tick_split(InputPhase::Even); // [1]
        prediction.shift(); // [2]
        prediction.shift(); // [3]
        prediction.add_tick_split(InputPhase::Odd); // [4, 1]
        prediction.shift(); // [5, 2]
        prediction.shift(); // [6, 3]
        prediction.add_tick(1); // [6, 3] []
        prediction.add_tick_split(InputPhase::Even); // [7, 4] [1]
        prediction.shift(); // [8, 5] [2]
        prediction.shift(); // [9, 6] [3]
        prediction.shift(); // [10, 7] [4]
        prediction.read(&[1, 0, 0, 0, 1, 0, 0, 0, 0, 0]); // [7] [4]
        prediction.shift(); // [8] [5]
        prediction.shift(); // [9] [6]
        prediction.shift(); // [10] [7]
        prediction.read(&[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]); // [] [7]
        prediction.shift(); // [] [8]
        prediction.shift(); // [] [9]
        prediction.shift(); // [] [10]
        prediction.read(&[0, 0, 1, 0, 1, 0, 0, 0, 0, 0]); // [] []

        assert!(!prediction.should_read());
        assert!(!prediction.is_finished());
        assert!(prediction.is_all_ticks_added());

        prediction.add_tick(0); // []
        prediction.add_tick_split(InputPhase::Even); // [1]
        prediction.shift(); // [2]
        prediction.shift(); // [3]
        prediction.add_tick_split(InputPhase::Odd); // [4, 1]
        prediction.shift(); // [5, 2]
        prediction.shift(); // [6, 3]
        prediction.add_tick(1); // []
        prediction.add_tick_split(InputPhase::Even); // [7, 4] [1]
        prediction.shift(); // [8, 5] [2]
        prediction.shift(); // [9, 6] [3]
        prediction.shift(); // [10, 7] [4]
        prediction.read(&[1, 0, 0, 0, 1, 0, 0, 0, 0, 0]); // [7] [4]
        prediction.shift(); // [8] [5]
        prediction.shift(); // [9] [6]
        prediction.shift(); // [10] [7]
        prediction.read(&[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]); // [] [7]
        prediction.shift(); // [] [8]
        prediction.shift(); // [] [9]
        prediction.shift(); // [] [10]
        prediction.read(&[0, 0, 1, 0, 1, 0, 0, 0, 0, 0]); // [] []

        assert!(!prediction.should_read());
        assert!(!prediction.is_finished());
        assert!(prediction.is_all_ticks_added());

        prediction.add_tick(0); // []
        prediction.add_tick_split(InputPhase::Even); // [1]
        prediction.shift(); // [2]
        prediction.shift(); // [3]
        prediction.add_tick_split(InputPhase::Odd); // [4, 1]
        prediction.shift(); // [5, 2]
        prediction.shift(); // [6, 3]
        prediction.add_tick(1); // []
        prediction.add_tick_split(InputPhase::Even); // [7, 4] [1]
        prediction.shift(); // [8, 5] [2]
        prediction.shift(); // [9, 6] [3]
        prediction.shift(); // [10, 7] [4]
        prediction.read(&[1, 0, 0, 0, 1, 0, 0, 0, 0, 0]); // [7] [4]
        prediction.shift(); // [8] [5]
        prediction.shift(); // [9] [6]
        prediction.shift(); // [10] [7]
        prediction.read(&[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]); // [] [7]
        prediction.shift(); // [] [8]
        prediction.shift(); // [] [9]
        prediction.shift(); // [] [10]
        prediction.read(&[0, 0, 1, 0, 1, 0, 0, 0, 0, 0]); // [] []

        assert!(!prediction.should_read());
        assert!(prediction.is_finished());
        assert!(prediction.is_all_ticks_added());

        let result = prediction.get_prediction();

        assert_eq!(
            result,
            vec![
                vec![true, false, true, false, true, false, true, true, false, false],
                vec![true, false, true, false, true, false, true, true, false, false],
                vec![true, false, true, false, true, false, true, true, false, false]
            ],
            "30 30 30"
        );
    }
}
