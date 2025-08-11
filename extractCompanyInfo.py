from dotenv import load_dotenv
import os
import json
import pandas as pd
import re
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import initialize_agent, Tool
from langchain_core.messages import SystemMessage, HumanMessage

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize the Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
    """
    You are an intelligent agent tasked with extracting company details from a paragraph. For each paragraph, identify:
    - S.No. (int)
    - Company Name (string)
    - Founding Date (format as YYYY-MM-DD; if only year is provided, use January 1st; if year and month, use 1st of the month)
    - Founders (list of names as a comma-separated string)

    Input paragraph: {paragraph}

    Return ONLY the extracted information as a valid JSON string:
    {{"S.No.": "<int>", "company_name": "<string>", "founding_date": "<YYYY-MM-DD>", "founders": "<comma-separated string>"}}
    """
)

# LCEL chain
chain = prompt | llm | StrOutputParser()

# Validate and fix dates
def validate_and_fix_date(date_str):
    parts = date_str.strip().split("-")
    if len(parts) == 1:  # Year only
        return f"{parts[0]}-01-01"
    elif len(parts) == 2:  # Year + month
        return f"{parts[0]}-{parts[1].zfill(2)}-01"
    elif len(parts) == 3:  # Full date
        return f"{parts[0]}-{parts[1].zfill(2)}-{parts[2].zfill(2)}"
    else:
        raise ValueError(f"Invalid date format: {date_str}")

# Write CSV safely (append mode)
def write_to_csv(data, filename="company_info.csv"):
    df = pd.DataFrame(data)
    # Append if file exists
    if os.path.exists(filename):
        df.to_csv(filename, mode="a", index=False, header=False)
    else:
        df.to_csv(filename, index=False)
    print(f"CSV file '{filename}' updated successfully")

# Process essay text
async def process_essay(essay_text):
    paragraphs = [p.strip() for p in essay_text.split("\n") if p.strip()]
    results = []
    for paragraph in paragraphs:
        try:
            response = await chain.ainvoke({"paragraph": paragraph})
            # Clean JSON if model outputs extra text
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if not match:
                raise ValueError("No valid JSON found in LLM output")
            parsed = json.loads(match.group())
            parsed["founding_date"] = validate_and_fix_date(parsed["founding_date"])
            results.append(parsed)
        except Exception as e:
            print(f"Error processing paragraph: {paragraph}\nError: {e}")
    return results

# CSV-writing tool for agent
def csv_tool_func(input_data):
    try:
        if isinstance(input_data, str):
            match = re.search(r"\{.*\}", input_data, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
            else:
                raise ValueError("No JSON found in input string")
        else:
            parsed = input_data
        write_to_csv([parsed], "company_info_agent.csv")
    except Exception as e:
        return f"Error writing CSV: {e}"
    return "Data saved to CSV"

csv_tool = Tool(
    name="CSV_Writer",
    description="Writes extracted company data to a CSV file. Input should be a JSON string with fields: company_name, founding_date, founders.",
    func=csv_tool_func
)

# Initialize agent (default prompt)
agent = initialize_agent(
    tools=[csv_tool],
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

# Process with agent
async def process_with_agent(paragraph):
    try:
        # First, extract JSON with LCEL chain
        response = await chain.ainvoke({"paragraph": paragraph})
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if not match:
            raise ValueError("No JSON found from extraction")
        data = json.loads(match.group())
        # Pass data to the agent tool
        return await agent.arun(f"Save this data to CSV: {json.dumps(data)}")
    except Exception as e:
        print(f"Agent error: {e}")
        return None

# Main function
async def main():
    essay = """
    Google LLC was founded on September 4, 1998, by Larry Page and Sergey Brin while they were Ph.D. students at Stanford University. It evolved from a research project into a global leader in search technology.
    Microsoft was established in 1975 by Bill Gates and Paul Allen. The company played a pivotal role in the personal computing revolution with its development of the MS-DOS operating system.
    Amazon, founded by Jeff Bezos in July 1994, began as an online bookstore and has since expanded into a vast e-commerce and cloud computing empire.
    """

    # Process using LCEL chain
    extracted_data = await process_essay(essay)
    write_to_csv(extracted_data)

    # Optionally process with agent (appends separately)
    for paragraph in essay.split("\n"):
        if paragraph.strip():
            await process_with_agent(paragraph)

# Run script
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
