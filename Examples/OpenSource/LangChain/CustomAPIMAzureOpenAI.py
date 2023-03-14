from __future__ import annotations

from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain
from langchain.llms.openai import *

class CustomAPIMAzureOpenAI(AzureOpenAI):
    """Azure specific OpenAI class that uses deployment name."""

    deployment_name: str = ""
    """Deployment name to use."""
    subscription_key: str = ""

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            **{"deployment_name": self.deployment_name, "subscription_key":self.subscription_key},
            **super()._identifying_params,
        }

    @property
    def _invocation_params(self) -> Dict[str, Any]:
        return {**{"engine": self.deployment_name}, **super()._invocation_params}

    def _generate(
        self, prompts: List[str], stop: Optional[List[str]] = None
    ) -> LLMResult:
        """Call out to OpenAI's endpoint with k unique prompts.
        Args:
            prompts: The prompts to pass into the model.
            stop: Optional list of stop words to use when generating.
        Returns:
            The full LLM output.
        Example:
            .. code-block:: python
                response = openai.generate(["Tell me a joke."])
        """
        # TODO: write a unit test for this
        params = self._invocation_params
        sub_prompts = self.get_sub_prompts(params, prompts, stop)
        choices = []
        token_usage: Dict[str, int] = {}
        # Get the token usage from the response.
        # Includes prompt, completion, and total tokens used.
        _keys = {"completion_tokens", "prompt_tokens", "total_tokens"}
        for _prompts in sub_prompts:
            if self.streaming:
                if len(_prompts) > 1:
                    raise ValueError("Cannot stream results with multiple prompts.")
                params["stream"] = True
                response = _streaming_response_template()
                for stream_resp in completion_with_retry(
                    self, prompt=_prompts, headers={'Ocp-Apim-Subscription-Key':self.subscription_key},  **params
                ):
                    self.callback_manager.on_llm_new_token(
                        stream_resp["choices"][0]["text"],
                        verbose=self.verbose,
                        logprobs=stream_resp["choices"][0]["logprobs"],
                    )
                    _update_response(response, stream_resp)
                choices.extend(response["choices"])
            else:
                response = completion_with_retry(self, prompt=_prompts, headers={'Ocp-Apim-Subscription-Key':self.subscription_key},  **params)
                choices.extend(response["choices"])
            if not self.streaming:
                # Can't update token usage if streaming
                update_token_usage(_keys, response, token_usage)
        return self.create_llm_result(choices, prompts, token_usage)
    
    async def _agenerate(self, prompts: List[str], stop: Optional[List[str]] = None
                        ) -> LLMResult:
        """Call out to OpenAI's endpoint async with k unique prompts."""
        params = self._invocation_params
        sub_prompts = self.get_sub_prompts(params, prompts, stop)
        choices = []
        token_usage: Dict[str, int] = {}
        # Get the token usage from the response.
        # Includes prompt, completion, and total tokens used.
        _keys = {"completion_tokens", "prompt_tokens", "total_tokens"}
        for _prompts in sub_prompts:
            if self.streaming:
                if len(_prompts) > 1:
                    raise ValueError("Cannot stream results with multiple prompts.")
                params["stream"] = True
                response = _streaming_response_template()
                async for stream_resp in await acompletion_with_retry(
                    self, prompt=_prompts, headers={'Ocp-Apim-Subscription-Key':self.subscription_key}, **params
                ):
                    if self.callback_manager.is_async:
                        await self.callback_manager.on_llm_new_token(
                            stream_resp["choices"][0]["text"],
                            verbose=self.verbose,
                            logprobs=stream_resp["choices"][0]["logprobs"],
                        )
                    else:
                        self.callback_manager.on_llm_new_token(
                            stream_resp["choices"][0]["text"],
                            verbose=self.verbose,
                            logprobs=stream_resp["choices"][0]["logprobs"],
                        )
                    _update_response(response, stream_resp)
                choices.extend(response["choices"])
            else:
                response = await acompletion_with_retry(self, prompt=_prompts, headers={'Ocp-Apim-Subscription-Key':self.subscription_key},  **params)
                choices.extend(response["choices"])
            if not self.streaming:
                # Can't update token usage if streaming
                update_token_usage(_keys, response, token_usage)
        return self.create_llm_result(choices, prompts, token_usage)
    
    