from llama_index.core import PromptTemplate
from llama_index.core.llms import LLM

from llm_agents import LlamaIndexBaseAgent
from llm_agents.helper_mixins import UnifiedHelperMixin
from llm_agents.mape_k_component_mixins import StreamingMixin, BaseAgentInitMixin
from llm_agents.models import SinglePromptResultModel, ExecuteResult
from llama_index.core.workflow.retry_policy import ConstantDelayRetryPolicy
from llama_index.core.workflow import Context, StartEvent, StopEvent, step, Workflow
from llm_agents.agent_utils import OPENAI_REASONING_MODEL_NAME
from llm_agents.prompt_mixins import UnifiedPrompt
from llm_agents.utils.postprocess_message import replace_plot_placeholders


class UnifiedMixin(UnifiedHelperMixin, StreamingMixin):
    @step(retry_policy=ConstantDelayRetryPolicy(delay=5, maximum_attempts=0))
    async def unified_mape_k(self, ctx: Context, ev: StartEvent) -> StopEvent:
        user_message = ev.input
        await ctx.set("user_message", user_message)

        # use SinglePromptPrompt for unified call
        sp_pm = UnifiedPrompt()
        template = sp_pm.get_prompts()["default"].get_template()
        prompt_str = template.format(
            domain_description=self.domain_description,
            feature_context=self.get_formatted_feature_context(),
            instance=self.instance,
            predicted_class_name=self.predicted_class_name,
            understanding_displays=self.understanding_displays.as_text(),
            modes_of_engagement=self.modes_of_engagement.as_text(),
            chat_history=self.get_chat_history_as_xml(),
            user_message=user_message,
            user_model=self.user_model.get_state_summary(as_dict=False),
            explanation_collection=self.user_model.get_complete_explanation_collection(as_dict=False),
            explanation_plan=self.format_predefined_plan_for_prompt(),
            last_shown_explanations=self.get_formatted_last_shown_explanations(),
        )

        # Wrap the prompt string in a PromptTemplate for structured prediction
        unified_prompt = PromptTemplate(prompt_str)
        # Use unified prediction (streaming or not) with logging
        result: SinglePromptResultModel = await self._predict(SinglePromptResultModel, unified_prompt,
                                                              "SinglePromptResult")

        # Process the unified result using helper method
        # This handles all MAPE-K phases in one go
        self.process_unified_result(user_message, result)

        # Prepare final result for workflow
        final = ExecuteResult(
            reasoning=result.reasoning,
            response=replace_plot_placeholders(result.response, self.visual_explanations_dict),
        )
        return StopEvent(result=final)


class MapeKUnifiedBaseAgent(Workflow, LlamaIndexBaseAgent, UnifiedMixin, StreamingMixin, BaseAgentInitMixin):
    """
    Unified MAPE-K agent: performs all MAPE-K steps in a single LLM call with streaming support.
    """

    def __init__(self, llm: LLM = None, structured_output: bool = True, timeout: float = 100.0, **kwargs):
        # Initialize LlamaIndexBaseAgent with all the base parameters
        LlamaIndexBaseAgent.__init__(self, **kwargs)

        # Initialize with special reasoning model
        self._init_agent_components(
            llm=llm,
            structured_output=structured_output,
            timeout=timeout,
            special_model=OPENAI_REASONING_MODEL_NAME,
            **kwargs
        )