# Rio Kutty Activities Engine 🚀

Rio Kutty Activities Engine is an automated service designed to generate fun and educational activities for kids based on their favorite stories. Using advanced AI agents and structured workflows, it creates multiple types of engaging content like quizzes, art projects, and science experiments.

---

## 🌟 How It Works

The engine follows a simple but powerful process:

1.  **Request**: The service receives a request containing a `story_id` and the child's `age`.
2.  **Fetch**: It retrieves the full story text from the database (Firestore).
3.  **Generate**: Multiple AI "Agents" work in parallel to create different activities:
    -   **MCQ Agent**: Creates fun trivia questions.
    -   **Art Agent**: Suggests creative drawing or craft ideas.
    -   **Moral Agent**: Helps find the lesson or values in the story.
    -   **Science Agent**: Explains the "how" and "why" behind story events.
    -   AI calls run with retry + exponential backoff, circuit breaker protection, and rate limiting.
    -   Text generation uses a primary low-cost Gemini model with automatic fallback model support.
4.  **Validate**: A validator agent checks the content to ensure it's age-appropriate and high-quality.
5.  **Save**: The activities are saved back to the database for the app to display.

---

## 📁 Project Structure

Here is how the project is organized:

```text
rio-kutty-activities-agents/
├── src/
│   ├── agents/            # 🤖 AI Agents specialized for different tasks
│   │   ├── mcq_agent.py      # Generates multiple choice questions
│   │   ├── art_agent.py      # Generates art & craft activities
│   │   ├── moral_agent.py    # Generates moral & creative activities
│   │   ├── science_agent.py  # Generates science-related activities
│   │   └── validators/       # Ensures generated content is correct
│   ├── workflows/         # 🧠 Logic that connects the agents (using LangGraph)
│   │   └── activity_workflow.py
│   ├── services/          # 🛠️ External integrations
│   │   ├── ai_service.py     # Connects to Google Gemini AI
│   │   └── database/         # Handles Firestore DB operations
│   ├── utils/             # ⚙️ Configuration, Logging, and Helpers
│   └── main.py            # 🚀 Entry point - The FastAPI web server
├── tests/                 # ✅ Unit and integration tests
├── Dockerfile             # 🐳 For deploying the app easily
└── pyproject.toml         # 📦 Project dependencies and metadata
```

---

## 🛠️ Technology Stack

-   **Python**: The core programming language.
-   **FastAPI**: A modern, fast web framework for building APIs.
-   **LangGraph**: A library for building stateful, multi-agent applications.
-   **Google Gemini AI**: The "brain" behind the content generation.
-   **Google Cloud Firestore**: The database for storing stories and activities.
-   **Pub/Sub**: Handles background messages for activity generation.

---

## 🚀 Getting Started

### 1. Prerequisites
-   Python 3.11+
-   A Google Cloud project with GenAI and Firestore enabled.

### 2. Setup Environment
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_gemini_api_key
GOOGLE_CLOUD_PROJECT=your_gcp_project_id
GOOGLE_CLOUD_BUCKET=your_gcs_bucket_name
FIRESTORE_DATABASE=(default)

GEMINI_MODEL=gemini-2.0-flash-lite
GEMINI_FALLBACK_MODEL=gemini-2.0-flash
MULTIMODAL_MODEL=gemini-2.5-flash-image
HF_TOKEN=your_huggingface_token

MAX_RETRIES=3
RETRY_DELAY_SECONDS=2
CIRCUIT_BREAKER_FAILURE_THRESHOLD=5
RATE_LIMIT_TOKENS_PER_SECOND=3
RATE_LIMIT_BURST_CAPACITY=6
CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS=60

# Prompt versioning
MCQ_PROMPT_VERSION=latest
ART_PROMPT_VERSION=latest
MORAL_PROMPT_VERSION=latest
SCIENCE_PROMPT_VERSION=latest
```

### 3. Install Dependencies
```bash
pip install -e .
```

### 4. Run the Application
```bash
python -m src.main
```
The server will start at `http://localhost:8080`.

---

## 📡 API Endpoints

-   `POST /generate-activities`: Directly triggers activity generation for a story (returns immediate accepted response; processing continues in background).
-   `POST /pubsub-handler`: Listens for messages from Google Cloud Pub/Sub (returns `202` on valid message).
-   `GET /health`: Simple health check.

---

## 🧪 DeepEval (Activity LLM Quality)

Install test extras:
```bash
pip install -e .[test]
```

Run DeepEval activity checks (opt-in):
```bash
RUN_DEEPEVAL=true OPENAI_API_KEY=your_openai_key GOOGLE_API_KEY=your_gemini_api_key pytest tests/integration/test_deepeval_activity_llms.py -v
```

Notes:
- DeepEval tests are skipped by default.
- These tests call real LLMs and validate output quality for MCQ, Art, Moral, and Science generation.

---

## 🤝 Contributing

We welcome contributions! Feel free to open issues or submit pull requests to help make learning more fun for kids everywhere!
