export type ITimelineItem =
	| {
			type: "Float";
			min_value: number;
			max_value: number;
			capacity: number;
	  }
	| {
			type: "Integer";
			min_value: number;
			max_value: number;
			capacity: number;
	  }
	| {
			type: "Enum";
			options: readonly string[];
			capacity: number;
	  }
	| {
			type: "Datetime";
			format?: string;
	  };

export type IConfig = {
	timelines: readonly ITimelineItem[];
};

export type ITimelineValue =
	| {
			Float: number;
	  }
	| {
			Integer: number;
	  }
	| {
			Enum: string;
	  }
	| {
			Datetime: string;
	  };

declare global {
	var __APP_CONFIG__: IConfig;
	var __TRAINING_SERVER__: string;
	var __PREDICTION_SERVER__: string;
}
