import { Tabs, TabsItem } from "@vkontakte/vkui";
import axios from "axios";
import { useCallback, useState } from "react";
import { BinaryField } from "./BinaryField";
import { BinaryForm } from "./BinaryForm";
import { Form } from "./Form";
import type { IPrediction, ITimelineValue } from "./types";

export function App() {
	const [isBinary, setIsBinary] = useState(false);

	const [lastPrediction, setLastPrediction] = useState<IPrediction | null>(
		null,
	);

	const onPredict = useCallback(async (values: readonly ITimelineValue[]) => {
		const response = await axios.post<IPrediction>(
			`${__PREDICTION_SERVER__}/predict`,
			values,
		);
		setLastPrediction(response.data);
	}, []);

	const onTrain = useCallback(async (values: readonly ITimelineValue[]) => {
		await axios.post(`${__TRAINING_SERVER__}/push_data`, values);
	}, []);

	const onPredictBinary = useCallback(async (values: readonly boolean[]) => {
		const response = await axios.post<IPrediction>(
			`${__PREDICTION_SERVER__}/predict_binary`,
			values,
		);
		setLastPrediction(response.data);
	}, []);

	const onTrainBinary = useCallback(async (values: readonly boolean[]) => {
		await axios.post(`${__TRAINING_SERVER__}/push_data_binary`, values);
	}, []);

	const onUpdate = useCallback(() => {
		axios.post(`${__TRAINING_SERVER__}/update_receivers`);
	}, []);

	return (
		<>
			<Tabs>
				<TabsItem
					selected={!isBinary}
					onClick={() => {
						setIsBinary(false);
					}}
				>
					Data
				</TabsItem>

				<TabsItem
					selected={isBinary}
					onClick={() => {
						setIsBinary(true);
					}}
				>
					Binary
				</TabsItem>
			</Tabs>

			<div
				style={{
					maxWidth: 600,
				}}
			>
				{isBinary ? (
					<BinaryForm
						onPredictBinary={onPredictBinary}
						onTrainBinary={onTrainBinary}
						onUpdate={onUpdate}
					/>
				) : (
					<Form onPredict={onPredict} onTrain={onTrain} onUpdate={onUpdate} />
				)}
			</div>

			{lastPrediction && (
				<div>
					<h3>Last prediction</h3>

					{isBinary ? (
						<BinaryField values={lastPrediction.raw[0]} />
					) : (
						<pre>
							<code>{JSON.stringify(lastPrediction.data, null, 2)}</code>
						</pre>
					)}
				</div>
			)}
		</>
	);
}
