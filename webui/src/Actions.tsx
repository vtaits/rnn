import { Icon16DownloadOutline } from "@vkontakte/icons";
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

			<Button
				href={`${__TRAINING_SERVER__}/download_dump`}
				download
				after={<Icon16DownloadOutline />}
			>
				Download
			</Button>
		</Flex>
	);
}
