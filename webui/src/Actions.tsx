import { Icon16DownloadOutline } from "@vkontakte/icons";
import { Button, Flex } from "@vkontakte/vkui";

type IActionsProps = Readonly<{
	onTrain: VoidFunction;
	onPredict: VoidFunction;
	onUpdate: VoidFunction;
}>;

export function Actions({ onTrain, onPredict, onUpdate }: IActionsProps) {
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

			<Button onClick={onUpdate}>Update prediction server</Button>
		</Flex>
	);
}
