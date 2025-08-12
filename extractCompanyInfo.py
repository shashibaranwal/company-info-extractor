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
    # df.insert("S.No.", range(1, len(data) + 1))
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
    In the ever-evolving landscape of global commerce, the origin stories of major corporations are not merely tales of personal ambition and entrepreneurial spirit but also reflections of broader socio-economic trends and technological revolutions that have reshaped industries. These narratives, which often begin with modest ambitions, unfold into chronicles of innovation and strategic foresight that define industries and set benchmarks for future enterprises.

    Early Foundations: Pioneers of Industry
    One of the earliest examples is The Coca-Cola Company, founded on May 8, 1886, by Dr. John Stith Pemberton in Atlanta, Georgia. Initially sold at Jacob's Pharmacy as a medicinal beverage, Coca-Cola would become one of the most recognized brands worldwide, revolutionizing the beverage industry.
    Similarly, Sony Corporation was established on May 7, 1946, by Masaru Ibuka and Akio Morita in Tokyo, Japan. Starting with repairing and building electrical equipment in post-war Japan, Sony would grow to pioneer electronics, entertainment, and technology.
    As the mid-20th century progressed, McDonald's Corporation emerged as a game-changer in the fast-food industry. Founded on April 15, 1955, in Des Plaines, Illinois, by Ray Kroc, McDonald's built upon the original concept of Richard and Maurice McDonald to standardize and scale fast-food service globally. Around the same period, Intel Corporation was established on July 18, 1968, by Robert Noyce and Gordon Moore in Mountain View, California

    driving advancements in semiconductors and microprocessors that became the backbone of modern computing.

    The Rise of Technology Titans
    Samsung Electronics Co., Ltd., founded on January 13, 1969, by Lee Byung-chul in Su-dong, South Korea, initially focused on producing electrical appliances like televisions and refrigerators. As Samsung expanded into semiconductors, telecommunications, and digital media, it
    grew into a global technology leader. Similarly, Microsoft Corporation was founded on April 4, 1975, by Bill Gates and Paul Allen in Albuquerque, New Mexico, with the vision of placing a computer on every desk and in every home.
    In Cupertino, California, Apple Inc. was born on April 1, 1976, founded by Steve Jobs, Steve Wozniak, and Ronald Wayne. Their mission to make personal computing accessible and elegant revolutionized technology and design. A few years later, Oracle Corporation was established on June 16, 1977, by Larry Ellison, Bob Miner, and Ed Oates in Santa Clara, California.
    Specializing in relational databases, Oracle would become a cornerstone of enterprise software and cloud computing.
    NVIDIA Corporation, founded on April 5, 1993, by Jensen Huang, Chris Malachowsky, and Curtis Priem in Santa Clara, California, began with a focus on graphics processing units (GPUs) for gaming. Today, NVIDIA is a leader in artificial intelligence, deep learning, and autonomous systems, showcasing the power of continuous innovation.

    E-Commerce and the Internet Revolution
    The 1990s witnessed a dramatic shift toward e-commerce and internet technologies. Amazon.com Inc. was founded on July 5, 1994, by Jeff Bezos in a garage in Bellevue, Washington, with the vision of becoming the world's largest online bookstore. This vision rapidly expanded to encompass
    e-commerce, cloud computing, and digital streaming. Similarly, Google LLC was founded on September 4, 1998, by Larry Page and Sergey Brin, PhD students at Stanford University, in a garage in Menlo Park, California.
    Google's mission to "organize the world's information" transformed how we search, learn, and connect.
    In Asia, Alibaba Group Holding Limited was founded on June 28, 1999, by Jack Ma and 18 colleagues in Hangzhou, China. Originally an e-commerce platform connecting manufacturers with buyers, Alibaba expanded into cloud

    computing, digital entertainment, and financial technology, becoming a global powerhouse.
    In Europe, SAP SE was founded on April 1, 1972, by Dietmar Hopp,
    Hans-Werner Hector, Hasso Plattner, Klaus Tschira, and Claus Wellenreuther in Weinheim, Germany. Specializing in enterprise resource planning (ERP) software, SAP revolutionized how businesses manage operations and data.

    Social Media and Digital Platforms
    The 2000s brought a wave of social media and digital platforms that reshaped communication and commerce. LinkedIn Corporation was founded on December 28, 2002, by Reid Hoffman and a team from PayPal and Socialnet.com in Mountain View, California, focusing on professional networking.
    Facebook, Inc. (now Meta Platforms, Inc.) was launched on February 4, 2004, by Mark Zuckerberg and his college roommates in Cambridge, Massachusetts, evolving into a global social networking behemoth.
    Another transformative platform, Twitter, Inc., was founded on March 21, 2006, by Jack Dorsey, Biz Stone, and Evan Williams in San Francisco, California. Starting as a microblogging service, Twitter became a critical tool for communication and social commentary. Spotify AB, founded on April 23, 2006, by Daniel Ek and Martin Lorentzon in Stockholm, Sweden, leveraged streaming technology to democratize music consumption, fundamentally altering the music industry.
    In the realm of video-sharing, YouTube LLC was founded on February 14, 2005, by Steve Chen, Chad Hurley, and Jawed Karim in San Mateo, California. YouTube became the leading platform for user-generated video content, influencing global culture and media consumption.

    Innovators in Modern Technology
    Tesla, Inc., founded on July 1, 2003, by a group including Elon Musk, Martin Eberhard, Marc Tarpenning, JB Straubel, and Ian Wright, in San Carlos, California, championed the transition to sustainable energy with its electric vehicles and energy solutions. Airbnb, Inc., founded in August 2008 by Brian Chesky, Joe Gebbia, and Nathan Blecharczyk in San Francisco, California, disrupted traditional hospitality with its peer-to-peer lodging platform.
    In the realm of fintech, PayPal Holdings, Inc. was established in December 1998 by Peter Thiel, Max Levchin, Luke Nosek, and Ken Howery in Palo Alto,

    California. Originally a cryptography company, PayPal became a global leader in online payments. Stripe, Inc., founded in 2010 by Patrick and John Collison in Palo Alto, California, followed suit, simplifying online payments and enabling digital commerce.
    Square, Inc. (now Block, Inc.), founded on February 20, 2009, by Jack Dorsey and Jim McKelvey in San Francisco, California, revolutionized mobile payment systems with its simple and accessible card readers.

    Recent Disruptors
    Zoom Video Communications, Inc. was founded on April 21, 2011, by Eric Yuan in San Jose, California. Initially designed for video conferencing, Zoom became essential during the COVID-19 pandemic, transforming remote work and communication. Slack Technologies, LLC, founded in 2009 by Stewart Butterfield, Eric Costello, Cal Henderson, and Serguei Mourachov in Vancouver, Canada, redefined workplace communication with its innovative messaging platform.
    Rivian Automotive, Inc., founded on June 23, 2009, by RJ Scaringe in Plymouth, Michigan, entered the electric vehicle market with a focus on adventure and sustainability. SpaceX, established on March 14, 2002, by Elon Musk in Hawthorne, California, revolutionized aerospace with reusable rockets and ambitious plans for Mars exploration.
    TikTok, developed by ByteDance and launched in September 2016 by Zhang Yiming in Beijing, China, revolutionized short-form video content, becoming a cultural phenomenon worldwide.

    Conclusion
    These corporations, with their diverse beginnings and visionary founders, exemplify the interplay of innovation, timing, and strategic foresight that shapes industries and transforms markets. From repairing electronics in post-war Japan to building global e-commerce empires and redefining space exploration, their stories are milestones in the narrative of global economic transformation. Each reflects not only the aspirations of their founders but also the technological advancements and socio-economic trends of their time, serving as inspirations for future innovators.
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
