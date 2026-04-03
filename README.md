# Chart Generator Agent

[![Fetch.ai](https://img.shields.io/badge/Fetch.ai-uAgents-purple)](https://fetch.ai)
[![ASI:One](https://img.shields.io/badge/ASI:One-LLM-blue)](https://asi1.ai)
[![Cloudinary](https://img.shields.io/badge/Cloudinary-Storage-3448C5)](https://cloudinary.com)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
![innovationlab](https://img.shields.io/badge/innovationlab-3D8BD3)
![tag:uagents](https://img.shields.io/badge/tag-uagents-blue)
![spotlight](https://img.shields.io/badge/spotlight-0EA5E9)

An AI-powered chart generation agent that transforms natural language descriptions into beautiful data visualizations. Uses ASI:One LLM to generate Python matplotlib code, executes it securely, and shares the charts via Cloudinary URLs — all through the uAgents chat protocol.

## Features

- **AI-Powered Code Generation** — Uses ASI:One LLM to generate optimized matplotlib code from natural language
- **Multiple Chart Types** — Line charts, bar charts, pie charts, scatter plots, box plots, heatmaps, word clouds, and more
- **Secure Execution** — Charts rendered locally using matplotlib's Agg backend (non-interactive)
- **Cloud Storage** — Automatic upload to Cloudinary for permanent, shareable URLs
- **Chat Integration** — Built on uAgents chat protocol for seamless agent-to-agent communication
- **Zero Local Storage** — Charts captured to memory and uploaded directly — no temp files
- **Smart Code Extraction** — Handles various code block formats from LLM responses

## Example

**Query:**
```
Create a pie chart of product categories with Electronics 35%, Clothing 25%, Food 20%, Books 15%, Other 5%
```

**Output:**

![Chart Example](https://res.cloudinary.com/doesqlfyi/image/upload/v1762378904/charts/chart_1762378903_1.png)

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Agent Framework | Fetch.ai uAgents + Chat Protocol |
| AI Model | ASI:One (asi1-mini) |
| Visualization | matplotlib, numpy, pandas, seaborn |
| Cloud Storage | Cloudinary |
| Extended Charts | wordcloud, networkx, squarify, matplotlib-venn |

## Getting Started

### Prerequisites

- Python 3.10+
- ASI:One API key ([get one here](https://asi1.ai))
- Cloudinary account (free tier works)

### Installation

```bash
git clone https://github.com/gautammanak1/Chart-Generator-Agent.git
cd Chart-Generator-Agent
pip install -r requirements.txt
```

### Configuration

Create a `.env` file:

```env
ASI_ONE_API_KEY=your_asi1_api_key
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_cloudinary_key
CLOUDINARY_API_SECRET=your_cloudinary_secret
```

### Run

```bash
python agent.py
```

The agent starts and publishes its manifest to Agentverse. Interact with it via the uAgents chat protocol.

## Project Structure

```
├── agent.py           # Agent setup, LLM integration, chart execution, Cloudinary upload
├── requirements.txt   # Python dependencies
├── .env.example       # Environment variable template
└── README.md
```

## Supported Visualizations

| Type | Library |
|------|---------|
| Line / Bar / Pie / Scatter / Box | matplotlib |
| Heatmaps / Correlation | seaborn |
| Word Clouds | wordcloud |
| Network Graphs | networkx |
| Treemaps | squarify |
| Venn Diagrams | matplotlib-venn |

## License

MIT
