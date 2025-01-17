import { Button, Flex } from "@vkontakte/vkui";

type IActionsProps = Readonly<{
	onTrain: VoidFunction;
	onPredict: VoidFunction;
}>;

export function Actions({ onTrain, onPredict }: IActionsProps) {
	return (
		<Flex gap="s">
			<Button onClick={onTrain}>Train</Button>

			<Button onClick={onPredict}>Predict</Button>
		</Flex>
	);
}
