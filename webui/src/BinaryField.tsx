import { Checkbox } from "@vkontakte/vkui";
import { type ReactNode, useMemo } from "react";

type IBinaryFieldProps = Readonly<{
	onChagne?: (inedx: number) => void;
	values: readonly boolean[];
}>;

export function BinaryField({ onChagne, values }: IBinaryFieldProps) {
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
							disabled={!onChagne}
							onChange={
								onChagne
									? ((index) => () => {
											onChagne(index);
										})(collectedIndex)
									: undefined
							}
						/>
					</td>,
				);

				++collectedIndex;
			}

			res.push(<tr key={y}>{row}</tr>);
		}

		return res;
	}, [onChagne, values]);

	return (
		<table>
			<tbody>{rows}</tbody>
		</table>
	);
}
