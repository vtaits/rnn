import { useState } from "react";
import { Actions } from "./Actions";
import { BinaryField } from "./BinaryField";

type IBinaryFormProps = Readonly<{
	onPredictBinary: (bitVec: readonly boolean[]) => void;
	onTrainBinary: (bitVec: readonly boolean[]) => void;
	onUpdate: VoidFunction;
}>;

const FIELD_SIZE =
	__APP_CONFIG__.layer_params.field_height *
	__APP_CONFIG__.layer_params.field_width;

function processValues(values: readonly unknown[]): boolean[] {
	const res: boolean[] = [];

	for (let i = 0; i < FIELD_SIZE; ++i) {
		res[i] = !!values[i];
	}

	return res;
}

export function BinaryForm({
	onPredictBinary,
	onTrainBinary,
	onUpdate,
}: IBinaryFormProps) {
	const [values, setValues] = useState<boolean[]>([]);

	return (
		<>
			<BinaryField
				values={values}
				onChagne={(index) => {
					setValues((prevValues) => {
						const nextValues = [...prevValues];
						nextValues[index] = !nextValues[index];
						return nextValues;
					});
				}}
			/>

			<Actions
				onTrain={() => {
					onTrainBinary(processValues(values));
				}}
				onPredict={() => {
					onPredictBinary(processValues(values));
				}}
				onUpdate={onUpdate}
			/>
		</>
	);
}
