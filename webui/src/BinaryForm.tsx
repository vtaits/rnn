import { Checkbox } from "@vkontakte/vkui";
import { type ReactNode, useMemo, useState } from "react";
import { Actions } from "./Actions";

type IBinaryFormProps = Readonly<{
	onPredictBinary: (bitVec: readonly boolean[]) => void;
	onTrainBinary: (bitVec: readonly boolean[]) => void;
}>;

export function BinaryForm({
	onPredictBinary,
	onTrainBinary,
}: IBinaryFormProps) {
	const [values, setValues] = useState<boolean[]>([]);

	const rows = useMemo(() => {
		const res: ReactNode[] = [];

		let collectedIndex = 0;

		for (let y = 0; y < __APP_CONFIG__.layer_params.field_height; ++y) {
			const row: ReactNode[] = [];

			for (let x = 0; x < __APP_CONFIG__.layer_params.field_width; ++x) {
				row.push(
					<td key={x}>
						<Checkbox
							checked={Boolean(values[collectedIndex])}
							onChange={((index) => () => {
								setValues((prevValues) => {
									const nextValues = [...prevValues];
									nextValues[index] = !nextValues[index];
									return nextValues;
								});
							})(collectedIndex)}
						/>
					</td>,
				);

				++collectedIndex;
			}

			res.push(<tr key={y}>{row}</tr>);
		}

		return res;
	}, [values]);

	return (
		<>
			<table>
				<tbody>{rows}</tbody>
			</table>

			<Actions
				onTrain={() => {
					onTrainBinary(values);
				}}
				onPredict={() => {
					onPredictBinary(values);
				}}
			/>
		</>
	);
}
