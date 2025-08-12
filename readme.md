# Company Info Extractor

This project is an intelligent system that extracts company details (name, founding date, and founders) from textual paragraphs using the LangChain framework with the Gemini API. The extracted data is organized and exported into a CSV file with a serial number for each entry.

## Project Overview

The system leverages the LangChain Expression Language (LCEL) to process essays or paragraphs, identify key company information, and save it to a `company_info.csv` file.

## Features

- Extracts company name, founding date, and founders from paragraphs.
- Formats dates as `YYYY-MM-DD` with default handling for incomplete data.
- Uses the Gemini API via LangChain for natural language processing.

## Requirements

- Python 3.x
- Virtual environment (`venv`)
- Required packages:
  - `langchain`
  - `langchain-google-genai`
  - `python-dotenv`
  - `pandas`

