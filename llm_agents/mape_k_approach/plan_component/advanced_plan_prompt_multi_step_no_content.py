from typing import List

from pydantic import BaseModel, Field

class ExplanationStepModel(BaseModel):
    """
    Data model for an explanation step in the explanation plan.
    """
    step_name: str = Field(..., description="The name of the explanation step.")
    description: str = Field(..., description="Description of the explanation step.")
    dependencies: list = Field(..., description="List of dependencies for the explanation step.")


class NewExplanationModel(BaseModel):
    """
    Data model for a new explanation concept to be added to the explanation plan.
    """
    explanation_name: str = Field(..., description="The name of the new explanation concept.")
    description: str = Field(..., description="Description of the new explanation concept.")
    explanation_steps: List[ExplanationStepModel] = Field(...,
                                                          description="List of steps for the new explanation concept. Each step is a dict with "
                                                                      "a 'step_name', 'description' and 'dependencies' keys.")


class ChosenExplanationModel(BaseModel):
    """
    Data model for a chosen explanation concept to be added to the explanation plan.
    """
    explanation_name: str = Field(..., description="The name of the explanation concept.")
    step: str = Field(..., description="The name or label of the step of the explanation.")


class PlanResultModel(BaseModel):
    """
    Data model for the result of the explanation plan generation.
    """
    reasoning: str = Field(...,
                           description="The reasoning behind the decision for new explanations and which explanations to include in the next steps.")
    new_explanations: List[NewExplanationModel] = Field(...,
                                                        description="List of new explanations to be added to the explanation plan. Each new explanation is a dict with an explanation_name, a description, and a list of steps called 'explanations'. Each step is a dict with a 'step_name', 'description' and 'dependencies' keys.")
    explanation_plan: List[ChosenExplanationModel] = Field(...,
                                                           description="Mandatory List of explanations or scaffolding with dicts with keys `(explanation_name, step)`, indicating the next steps to explain to the user. Cannot be empty list, at least contains the next explanations.")

def get_plan_prompt_template():
    return """
You are an explainer specialist who plans the explanation strategy in a dialogue with a user. The user is curious about an AI models prediction and is presented with explanations vie explainable AI tools. Your primary responsibilities include:

1. **Define New Explanations**
2. **Maintain the General Explanation Plan**
3. **Generate the Next Explanation to show the user**
    
<<Context>>:
- Domain Description: {domain_description}
- Model Features: {feature_names}
- Current Explained Instance:: {instance}
- Predicted Class by AI Model: {predicted_class_name}\n

<<User Model>>:
{user_model}\n

<<Explanation Collection>>:
{explanation_collection}
In the explanation collection above, the scaffolding strategies can be used to better grasp the user's understanding. When planning to use ScaffoldingStrategy, this should be the only step in the next explanation step.

<<Chat History>>:
{chat_history}\n

<<User Message>>:
"{user_message}".\n

<<Explanation Plan>>:
{previous_plan}\n
Note: The Explanation Plan serves as a high-level roadmap and should only be updated when shifts in user understanding occur, for example the user cannot understand an explanation because he lacks some more general concept knowledge, or the user explicitely wants to explore other concepts or explanation directions.

<<Last Explanation>>:
{last_explanation}\n
    
<<Task>>:
You have three primary tasks:
1. **Define New Explanations**:
    - Identify if new explanation concepts need to be introduced based on the user's latest input. If the user does not explicitly request a new concept or definition, use scaffolding strategies to address gaps in understanding.
    - If a new concept is required, define it and integrate it into the explanation_plan.
2. **Maintain the General Explanation Plan**:
    - Continuously assess whether the high-level explanation_plan remains relevant.
    - Update the explanation_plan only if substantial gaps or shifts in user understanding are identified.

**Guidelines**:
1. **Creating an Explanation Plan or deciding when to update the Plan if an one is given**:
    - Consider that the user might only ask one or maximally three questions in a row, so the explanation plan should be concise and focused.
    - **Initial Plan**: If no explanation_plan is given, generate a new one based on the user's latest input.
    - **Significant Changes**: Update the explanation_plan if the user demonstrates a major misunderstanding, requests a new overarching concept, or if their queries indicate a need for restructuring the explanation flow.
    - **Minor Adjustments**: Do not modify the explanation_plan for minor misunderstandings or clarifications. Instead, handle these through communication_goals. Delete already explained and understood concepts from the explanation_plan if this can be justified by the UserModel and the user's latest input.
    - If no plan is given, generate a new explanation_plan based on the user's latest input, planning ahead which order of explanations would be most beneficial for the user.

2. **Generating Communication Steps**:
    - Consider that the user might only ask one or maximally three questions in a row, so the communication steps should be concise and focused, awakening the users cusriousity and interest in the topic.
    - **Assess User Understanding**: If appropriate, begin with a step that assesses the user's familiarity with key concepts related to the next_explanation if the user's Machine Learning knowledge is not hight. The user might hear about machine learning for the first time and if you do not have enough information on the user yet, try to elicit the user's knowledge before diving into explanations.
    - **Adaptive Content**: Depending on the user's response, adapt the subsequent communication_goals to either delve deeper into the concept or simplify the explanation.
    - **Avoid Redundancy**: Do not repeat explanations unless the user explicitly requests clarification.
    - **Adapt to the user**: It is more important to react to a user's clarification request or question rather than folowing the general plan.

3. **Integration of New Explanations**:
    - When introducing new explanations, ensure they logically fit within the existing explanation_plan.
    - Provide clear connections between new and existing concepts to maintain a coherent learning path.

4. **Output Structure**:
    - **If updating the explanation_plan**:
        - Provide the updated explanation_plan in the `new_explanations` section.
        - Adjust the `chosen_explanation_plan` accordingly.
    - **Always provide the next_explanation with communication_goals** tailored to the latest user input.

**Example Workflow**:

1. **User Interaction**:
    - User asks, "Can you explain what overfitting is?"

2. **System Response**:
    - **Check Explanation Plan**: Determine if overfitting has already been covered or needs to be added.
    - **Update General Plan if Necessary**: If overfitting is a significant new concept, add it to the explanation_plan. Since it is a general concept of ML and not related to any of the provided explanations, add it as a step to "PossibleClarifications".
    - **Generate Communication Steps**:
        - Step 1: "Are you familiar how Machine Learning models learn from datasets and terms such as classification or regression?"
        - Step 2 (if user is familiar): "Overfitting occurs when a model learns the training data too well, including its noise and outliers, which negatively impacts its performance on new data."

Think step by step and provide a reasoning for each decision based on the user's latest input, the conversation history, and the current explanation plan.
"""
