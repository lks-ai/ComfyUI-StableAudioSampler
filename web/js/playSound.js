/*
	shouts to: pygoss, Fill, and Joviex!
*/
import { app } from "../../../scripts/app.js";

console.log("StableAudioSampler")

app.registerExtension({
	name: "lks-ai.StableAudioSampler",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "StableAudioSampler") {
			console.log(app);
			console.log(nodeData);
			const onExecuted = nodeType.prototype.onExecuted;
			//console.log(onExecuted);
			nodeType.prototype.onExecuted = async function (message) {
				onExecuted?.apply(this, arguments);
				// console.log(this.widgets);
				// console.log(app.ui.lastQueueSize);
				// console.log(message)

				// TODO can check this.widgets[] for specific controls
				let file = message.paths[0];
				if (!file) {
					file = "temp/stableaudiosampler.wav";
				}

				const url = new URL(`http://localhost:8188/view?filename=${encodeURIComponent(file)}&subfolder=&type=output&format=audio%2Fwav`);
				console.log(import.meta.url)
				console.log(url)
				const audio = new Audio(url);
				audio.volume = 1.0; //this.widgets[1].value;
				audio.play();
			};
		}
	},
});