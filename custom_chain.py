from langchain.chains.llm import LLMChain
from langchain.input import get_colored_text
from typing import Any, Dict, List, Optional, Tuple
from langchain.schema import PromptValue
from logger import logger


class CustomLoggingChain(LLMChain):
    async def aprep_prompts(
            self, input_list: List[Dict[str, Any]]
    ) -> Tuple[List[PromptValue], Optional[List[str]]]:
        """Prepare prompts from inputs."""
        stop = None
        if "stop" in input_list[0]:
            stop = input_list[0]["stop"]
        prompts = []
        for inputs in input_list:
            selected_inputs = {k: inputs[k] for k in self.prompt.input_variables}
            prompt = self.prompt.format_prompt(**selected_inputs)
            _colored_text = get_colored_text(prompt.to_string(), "green")
            _text = "Prompt after formatting:\n" + _colored_text
            logger.debug(_text)

            if self.callback_manager.is_async:
                await self.callback_manager.on_text(
                    _text, end="\n", verbose=self.verbose
                )
            else:
                self.callback_manager.on_text(_text, end="\n", verbose=self.verbose)
            if "stop" in inputs and inputs["stop"] != stop:
                raise ValueError(
                    "If `stop` is present in any inputs, should be present in all."
                )
            prompts.append(prompt)
        return prompts, stop