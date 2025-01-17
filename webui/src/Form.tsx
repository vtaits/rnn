import type { FieldSchemaBase } from "@vtaits/form-schema";
import { VKUIProvider } from "@vtaits/react-form-schema-ui-vkui";
import {
	Form as BaseForm,
	type DefaultFieldSchema,
} from "@vtaits/react-hook-form-schema/form";
import { Fragment, useCallback, useRef } from "react";
import { Actions } from "./Actions";
import type { ITimelineValue } from "./types";

const schemas = __APP_CONFIG__.timelines.reduce<
	Record<string, DefaultFieldSchema<FieldSchemaBase>>
>((res, timelineItem, index) => {
	switch (timelineItem.type) {
		case "Datetime":
			res[index] = {
				type: "datetime",
				label: `#${index + 1} datetime`,
				required: true,
			};
			break;

		case "Integer":
			res[index] = {
				type: "input",
				isNumber: true,
				label: `#${index + 1} integer`,
				hint: `${timelineItem.min_value} - ${timelineItem.max_value} ; ${timelineItem.capacity} bits`,
				required: true,
			};
			break;

		case "Float":
			res[index] = {
				type: "input",
				isNumber: true,
				label: `#${index + 1} float`,
				hint: `${timelineItem.min_value} - ${timelineItem.max_value} ; ${timelineItem.capacity} bits`,
				required: true,
			};
			break;

		case "Enum":
			res[index] = {
				type: "select",
				label: `#${index + 1} enum`,
				required: true,
				options: timelineItem.options,
				getOptionLabel: (option) => option as string,
				getOptionValue: (option) => option as string,
			};
			break;

		default:
			throw new Error("Unknown timeline item type");
	}

	return res;
}, {});

const names = __APP_CONFIG__.timelines.map((_, index) => String(index));

type IFormProps = Readonly<{
	onPredict: (values: readonly ITimelineValue[]) => Promise<void>;
	onTrain: (values: readonly ITimelineValue[]) => Promise<void>;
}>;

export function Form({ onPredict, onTrain }: IFormProps) {
	const submitTypeRef = useRef<"train" | "predict">("train");

	const handleSubmit = useCallback(
		async (values: Record<string, unknown>) => {
			const timelineValues = __APP_CONFIG__.timelines.map<ITimelineValue>(
				({ type }, index) => {
					switch (type) {
						case "Datetime":
							return {
								Datetime: values[index] as string,
							};

						case "Integer":
							return {
								Integer: values[index] as number,
							};

						case "Float":
							return {
								Float: values[index] as number,
							};

						case "Enum":
							return {
								Enum: values[index] as string,
							};

						default:
							throw new Error("Unknown timeline item type");
					}
				},
			);

			switch (submitTypeRef.current) {
				case "predict":
					await onPredict(timelineValues);
					break;

				case "train":
					await onTrain(timelineValues);
					break;

				default:
					throw new Error("Unknown submit type");
			}
		},
		[onPredict, onTrain],
	);

	return (
		<VKUIProvider>
			<BaseForm
				onSubmit={handleSubmit}
				schemas={schemas}
				renderFields={({ renderField }) => (
					<>
						{names.map((name) => (
							<Fragment key={name}>{renderField(name)}</Fragment>
						))}
					</>
				)}
				renderActions={({ onSubmit }) => (
					<Actions
						onTrain={() => {
							submitTypeRef.current = "train";
							onSubmit();
						}}
						onPredict={() => {
							submitTypeRef.current = "predict";
							onSubmit();
						}}
					/>
				)}
			/>
		</VKUIProvider>
	);
}
