import os
from typing import List, TypedDict
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

class State(TypedDict):
    text: str
    classification: str
    entities: List[str]
    summary: str
    
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def classification_node(state: State):
    """
    Classify the text into one of predefined categories.
    
    Parameters:
        state (State): The current state dictionary containing the text to classify
        
    Returns:
        dict: A dictionary with the "classification" key containing the category result
        
    Categories:
        - News: Factual reporting of current events
        - Blog: Personal or informal web writing
        - Research: Academic or scientific content
        - Other: Content that doesn't fit the above categories
    """
    prompt = PromptTemplate(
        input_variables = ["text"],
        template = "Classify the following text into one of the following categories: News, Blog, Research, Other\n\nText: {text}\n\nCategory:"
    )

    message = HumanMessage(content=prompt.format(text=state["text"]))

    classification = llm.invoke([message]).content.strip()

    return {"classification": classification}

def entity_extraction_node(state: State):
    # Function to identify and extract named entities from text
    # Organized by category (Person, Organization, Location)

    # Create template for entity extraction prompt
    # Specifies what entities to look for and format (comma-separated)
    prompt = PromptTemplate(
        input_variables = ["text"],
        template="Extract all the entities (Person, Organization, Location) from the following text. Provide the result as a comma-separated list.\n\nText:{text}\n\nEntities:"
    )

    message = HumanMessage(content=prompt.format(text=state["text"]))
    entities = llm.invoke([message]).content.strip().split(",")
    return {"entities": entities}

def summary_node(state: State):
    # Create a template for the summarization prompt
    # This tells the model to summarize the input text in one sentence
    summarization_prompt = PromptTemplate.from_template(
        """Summarize the following text in one short sentence.
        
        Text: {input}
        
        Summary:"""
    )

    chain = summarization_prompt | llm

    response  = chain.invoke({"input": state["text"]})
    return {"summary": response.content}

workflow = StateGraph(State)

# Add nodes to the  graph
workflow.add_node("classification_node", classification_node)
workflow.add_node("entity_extraction_node", entity_extraction_node)
workflow.add_node("summary_node", summary_node)

# Add edges to the graph
workflow.set_entry_point("classification_node")
workflow.add_edge("classification_node", "entity_extraction_node")
workflow.add_edge("entity_extraction_node", "summary_node")
workflow.add_edge("summary_node", END)

app = workflow.compile()

sample_text = """
Anthropic's MCP (Model Context Protocol) is an open-source powerhouse that lets your applications interact effortlessly with APIs across various systems.
"""

# Create the initial state with our sample text
state_input = {"text": sample_text}

# Run the agent's full workflow on our sample text
result = app.invoke(state_input)

# Print each component of the result:
# - The classification category (News, Blog, Research, or Other)
print("Classification:", result["classification"])

# - The extracted entities (People, Organizations, Locations)
print("\nEntities:", result["entities"])

# - The generated summary of the text
print("\nSummary:", result["summary"])