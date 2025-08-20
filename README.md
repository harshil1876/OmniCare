# OmniCare AI: Autonomous Operating System for Enterprise Intelligence

OmniCare AI is an enterprise-grade platform that converges AI, ML, Big Data, and Automation to create a living, adapting system for enterprise intelligence. It is designed to reduce costs, optimize decisions, and enable faster scaling by providing a unified AI brain for real-time insights and automation.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Workflow](#workflow)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

OmniCare AI aims to solve the problem of fragmented data and slow decision-making in enterprises that use dozens of SaaS tools. By unifying all tools and data into a single platform with an AI-powered core, OmniCare provides real-time insights, predictive analytics, and autonomous workflow automation across various departments.

## Features

- **Unified Command Center:** A Streamlit-based dashboard for executives to interact with company data, run simulations, and generate reports.
- **Domain-Specific AI Agents:** Specialized agents for Finance, HR, Marketing, and Operations that can perform tasks and provide insights.
- **Big Data Ingestion:** Ingests data from various sources like CRMs, HR systems, and marketing platforms in real-time.
- **Predictive Analytics:** Utilizes models like Prophet for sales forecasting, churn prediction, and demand forecasting.
- **Knowledge Management:** A searchable knowledge base powered by ChromaDB and LangChain for document retrieval and semantic search.
- **Autonomous Workflows:** Prefect-based workflows that can be triggered by data events to automate tasks and send alerts.

## Architecture

The OmniCare AI platform is built on a modular architecture with the following layers:

1.  **Data Aggregation Layer:** Handles batch and real-time data ingestion from various sources.
2.  **Processing & AI Intelligence Layer:** Cleans and preprocesses data, runs predictive models, and powers the AI agents.
3.  **User Interaction Layer:** The Streamlit-based frontend that provides the user interface for interacting with the platform.
4.  **Security & Compliance Layer:** Manages role-based access control, audit logging, and compliance checking.

## Project Structure

```
.
├── .gitignore
├── app.py
├── data
│   └── sales_data.csv
├── pages
│   ├── 1_Analytics.py
│   ├── 2_Agents.py
│   ├── 3_Knowledge_Base.py
│   └── 4_Settings.py
├── README.md
└── requirements.txt
```

## Getting Started

### Prerequisites

-   Conda or Miniconda installed
-   Python 3.11

### Installation

1.  **Create a clean conda environment:**
    ```bash
    conda create -n omnienv python=3.11 -y
    conda activate omnienv
    ```

2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the application, execute the following command in your terminal from the root of the project directory:

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`.

## Workflow

1.  **Data Ingestion:** Data is ingested from various sources and stored in the `data` directory.
2.  **Analytics:** The Analytics page displays visualizations and insights from the ingested data.
3.  **Knowledge Base:** Users can upload documents to the Knowledge Base and perform semantic searches.
4.  **Agents:** AI agents can be tasked with specific functions, such as generating reports or analyzing data.
5.  **Workflows:** Autonomous workflows can be set up to trigger actions based on data events.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License.
