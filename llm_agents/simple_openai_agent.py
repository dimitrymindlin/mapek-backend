"""
Simple OpenAI Agent for direct question answering.

This agent provides direct responses to user questions using the available context
and explanations without going through the MAPE-K workflow or monitoring phases.
"""
import os
import json
import logging
import datetime
from typing import Any, Dict, Optional, Callable, AsyncGenerator

from agents import Agent, Runner
from dotenv import load_dotenv

from create_experiment_data.instance_datapoint import InstanceDatapoint
from llm_agents.agent_utils import timed, append_new_log_row, update_last_log_row, OPENAI_MODEL_NAME, OPENAI_MINI_MODEL_NAME
from llm_agents.openai_base_agent import OpenAIAgent
from llm_agents.models import ChosenExplanationModel
from llm_agents.explanation_state import ExplanationState
from llm_agents.utils.postprocess_message import replace_plot_placeholders

# Configure logger
logger = logging.getLogger(__name__)

# Reduce verbosity for OpenAI and HTTP libraries
logging.getLogger("openai._base_client").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv()

# Configure OpenAI
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
LLM_MODEL = os.getenv('OPENAI_MODEL_NAME')


def create_simple_system_prompt(domain_description: str, feature_context: str, instance: Dict[str, Any], 
                               predicted_class_name: str, explanation_collection: str) -> str:
    """
    Create a simple system prompt for direct question answering.
    
    Args:
        domain_description: Description of the domain/dataset
        feature_context: Information about features
        instance: The current instance being explained
        predicted_class_name: The predicted class
        explanation_collection: Available explanations
        
    Returns:
        The formatted system prompt
    """
    return f"""You are an AI explanation assistant that helps users understand machine learning model predictions.

CONTEXT:
Domain: {domain_description}

Features: {feature_context}

Current Instance: {instance}

Model Prediction: {predicted_class_name}

AVAILABLE EXPLANATIONS:
{explanation_collection}

YOUR ROLE:
- Answer user questions directly and clearly
- Use the available explanations to support your responses
- Explain the model's prediction in an understandable way
- Be conversational and helpful
- Focus on what the user is asking about
- Reference specific feature values and explanations when relevant

GUIDELINES:
- Keep responses concise but informative
- Use plain language, avoid technical jargon unless necessary
- When discussing feature importance, reference actual values from the instance
- If you don't have specific information to answer a question, say so clearly
- Stay focused on the current prediction and available explanations
"""


def create_simple_user_prompt(chat_history: str, user_message: str) -> str:
    """
    Create a simple user prompt with chat history and current message.
    
    Args:
        chat_history: The conversation history
        user_message: The current user message
        
    Returns:
        The formatted user prompt
    """
    return f"""CONVERSATION HISTORY:
{chat_history}

CURRENT USER QUESTION:
{user_message}

Please provide a clear, helpful response to the user's question using the available context and explanations."""


class SimpleOpenAIAgent(OpenAIAgent):
    """
    Simple OpenAI agent that provides direct answers without MAPE-K workflow.
    
    This agent focuses on straightforward question answering using the available
    context and explanations, without the complexity of monitoring, analyzing,
    planning, or executing separate phases.
    """
    
    def __init__(
            self,
            experiment_id: str,
            feature_names: str = "",
            feature_units: str = "",
            feature_tooltips: str = "",
            domain_description: str = "",
            user_ml_knowledge: str = "",
    ):
        super().__init__(
            experiment_id=experiment_id,
            feature_names=feature_names,
            feature_units=feature_units,
            feature_tooltips=feature_tooltips,
            domain_description=domain_description,
            user_ml_knowledge=user_ml_knowledge
        )
        self.simple_agent = None
        
    def initialize_new_datapoint(
            self,
            instance: InstanceDatapoint,
            xai_explanations,
            xai_visual_explanations,
            predicted_class_name: str,
            opposite_class_name: str,
            datapoint_count: int
    ):
        """Initialize the agent with a new datapoint."""
        # Call parent initialization
        super().initialize_new_datapoint(
            instance,
            xai_explanations,
            xai_visual_explanations,
            predicted_class_name,
            opposite_class_name,
            datapoint_count
        )
        
        # Create the simple agent with system prompt
        self.simple_agent = Agent(
            name="simple_agent",
            model=LLM_MODEL,
            instructions=create_simple_system_prompt(
                domain_description=self.domain_description,
                feature_context=self.get_formatted_feature_context(),
                instance=self.instance,
                predicted_class_name=self.predicted_class_name,
                explanation_collection=str(xai_explanations),
            ),
        )
    
    @timed
    async def answer_user_question(self, user_question: str) -> Dict[str, Any]:
        """
        Answer user question directly using the simple agent.
        
        Args:
            user_question: The user's question
            
        Returns:
            Dictionary containing the response and metadata
        """
        if self.simple_agent is None:
            raise ValueError("Agent not initialized. Call initialize_new_datapoint first.")
        
        # Create user prompt with history and current question
        user_prompt = create_simple_user_prompt(
            chat_history=self.chat_history,
            user_message=user_question
        )
        
        # Get response from the agent
        response = await Runner.run(self.simple_agent, user_prompt)
        response = response.final_output
        
        # Log the prompt
        self.log_component_input_output("simple", user_prompt, response)
        
        # Extract the response text
        response_text = str(response)
        
        # Update chat history
        self.append_to_history("user", user_question)
        self.append_to_history("agent", response_text)
        
        # Prepare result
        result = {
            "response": response_text,
            "reasoning": "Direct response without MAPE-K workflow",
            "timestamp": datetime.now().isoformat(),
        }
        
        # Log to CSV
        row = {
            "timestamp": datetime.now().strftime("%d.%m.%Y_%H:%M"),
            "experiment_id": self.logging_experiment_id,
            "datapoint_count": self.datapoint_count,
            "user_message": user_question,
            "monitor": "N/A - Direct Response",
            "analyze": "N/A - Direct Response", 
            "plan": "N/A - Direct Response",
            "execute": result,
            "user_model": "N/A - No User Model",
        }
        append_new_log_row(row, self.log_file)
        update_last_log_row(row, self.log_file)
        
        return result
