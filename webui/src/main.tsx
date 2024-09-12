import { unwrap } from "krustykrab";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App.tsx";

createRoot(unwrap(document.getElementById("root"))).render(
	<StrictMode>
		<App />
	</StrictMode>,
);
