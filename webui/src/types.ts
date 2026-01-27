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
	  }
	| {
			type: "Weekday";
	  }
	| {
			type: "Time";
	  };

export type IConfig = {
	timelines: readonly ITimelineItem[];
	layer_params: {
		field_width: number;
		field_height: number;
		layer_width: number;
		layer_height: number;
	};
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
	  }
	| {
			Weekday: string;
	  }
	| {
			Time: string;
	  };

export type IPrediction = {
	raw: readonly (readonly boolean[])[];
	data: readonly (readonly ITimelineValue[])[];
};

declare global {
	var __APP_CONFIG__: IConfig;
	var __TRAINING_SERVER__: string;
	var __PREDICTION_SERVER__: string;
}
