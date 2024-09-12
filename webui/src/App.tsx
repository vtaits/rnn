import axios from "axios";
import { useCallback, useState } from "react";
import { Form } from "./Form";
import type { ITimelineValue } from "./types";

export function App() {
	const [lastPrediction, setLastPrediction] = useState<
		readonly ITimelineValue[] | null
	>(null);

	const onPredict = useCallback(async (values: readonly ITimelineValue[]) => {
		const response = await axios.post<ITimelineValue[]>(`${__PREDICTION_SERVER__}/predict`, values);
		setLastPrediction(response.data);
	}, []);

	const onTrain = useCallback(async (values: readonly ITimelineValue[]) => {
		await axios.post(
			`${__TRAINING_SERVER__}/push_data`,
			values,
		);
	}, []);

	return (
		<>
			<div
				style={{
					maxWidth: 600,
				}}
			>
				<Form onPredict={onPredict} onTrain={onTrain} />
			</div>

			{lastPrediction && (
				<div>
					<h3>Last prediction</h3>

					<pre>
						<code>{JSON.stringify(lastPrediction, null, 2)}</code>
					</pre>
				</div>
			)}
		</>
	);
}
