pub struct PredictionProcessingTick {
    result: Vec<bool>,
    tick_splits: Vec<usize>,
}

pub struct PredictionProcessing {
    split_count: usize,
    ticks: Vec<PredictionProcessingTick>,
    input_index: usize,
    collect_prediction_index: Option<usize>,
    field_size: usize,
    field_count: usize,
}

impl PredictionProcessing {
    pub fn new(split_count: usize, field_size: usize, field_count: usize) -> Self {
        PredictionProcessing {
            split_count,
            ticks: vec![],
            input_index: 0,
            collect_prediction_index: Some(0),
            field_size,
            field_count,
        }
    }

    pub fn add_tick(&mut self, index: usize) {
        self.input_index = index;

        if self.ticks.len() == self.split_count {
            panic!(
                "Number of splits of prediction is {}, cannot add more",
                self.split_count
            );
        }

        self.ticks.push(PredictionProcessingTick {
            tick_splits: vec![],
            result: vec![false; self.field_size],
        });

        if self.collect_prediction_index.is_none() {
            self.collect_prediction_index = Some(index);
        }
    }

    pub fn shift(&mut self) {
        let field_count = self.field_count;

        for tick in self.ticks.iter_mut() {
            for tick_split in tick.tick_splits.iter_mut() {
                if field_count == *tick_split {
                    panic!("Signal have not been read");
                }

                *tick_split += 1;
            }
        }
    }

    pub fn add_tick_split(&mut self) {
        self.shift();

        self.ticks[self.input_index].tick_splits.push(1);
    }

    pub fn should_read(&self) -> bool {
        match self.collect_prediction_index {
            Some(collect_prediction_index) => {
                if self.ticks[collect_prediction_index].tick_splits.len() == 0 {
                    return false;
                }

                self.ticks[collect_prediction_index].tick_splits[0] == self.field_count
            }
            None => false,
        }
    }

    pub fn read(&mut self, last_field_state: &[u8]) {
        match self.collect_prediction_index {
            Some(collect_prediction_index) => {
                for (index, value) in last_field_state.iter().enumerate() {
                    if *value > 0 {
                        self.ticks[collect_prediction_index].result[index] = true;
                    }
                }

                self.ticks[collect_prediction_index].tick_splits.remove(0);

                if self.ticks[collect_prediction_index].tick_splits.len() == 0 {
                    self.collect_prediction_index =
                        if collect_prediction_index == self.ticks.len() - 1 {
                            None
                        } else {
                            Some(collect_prediction_index + 1)
                        };
                }
            }
            None => {}
        }
    }

    pub fn is_finished(&self) -> bool {
        self.collect_prediction_index.is_none()
    }

    pub fn get_prediction(&self) -> Vec<bool> {
        let mut res = vec![];

        for tick in self.ticks.iter() {
            for byte in tick.result.iter() {
                res.push(*byte);
            }
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
    fn flow() {
        let mut prediction = PredictionProcessing::new(2, 5, 10);

        prediction.add_tick(0); // []

        assert!(!prediction.should_read());
        assert!(!prediction.is_finished());
        assert!(!prediction.is_all_ticks_added());

        prediction.add_tick_split(); // [1]

        assert!(!prediction.should_read());
        assert!(!prediction.is_finished());
        assert!(!prediction.is_all_ticks_added());

        prediction.shift(); // [2]

        assert!(!prediction.should_read());
        assert!(!prediction.is_finished());
        assert!(!prediction.is_all_ticks_added());

        prediction.shift(); // [3]

        prediction.add_tick_split(); // [4, 1]

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

        prediction.add_tick_split(); // [7, 4] [1]

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

        prediction.read(&[1, 0, 1, 0, 0]); // [7] [4]

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

        prediction.read(&[0, 0, 0, 0, 1]); // [] [7]

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

        prediction.read(&[0, 1, 1, 0, 0]); // [] []

        assert!(!prediction.should_read());
        assert!(prediction.is_finished());
        assert!(prediction.is_all_ticks_added());

        let result = prediction.get_prediction();

        assert_eq!(
            result,
            vec![true, false, true, false, true, false, true, true, false, false],
            "30"
        );
    }
}
