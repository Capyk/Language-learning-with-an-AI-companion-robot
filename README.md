# German Vocabulary AI Tutor Experiment

This project is a **research platform** designed to measure the effectiveness of **Adaptive AI Feedback** versus **Static Feedback** in learning German vocabulary.  
It features a **3-phase experiment**:

1. **Pre-test**
2. **Learning Session**
3. **Post-test**

---

## 🚀 Getting Started

Follow these instructions to get the project up and running on your local machine for testing.

---

## 📦 Prerequisites

- **Python 3.10+**
- **Node.js v18+** and **npm**
- **Google Gemini API Key**  
  (Get one at https://aistudio.google.com/)

---

## 🛠️ Backend Setup (FastAPI)

The backend manages:

- Experiment logic
- Session state
- Integration with **Google Gemini** for adaptive hints

### 1️⃣ Navigate to the backend folder

```bash
cd backend
```

### 2️⃣ Create and activate a virtual environment

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Mac / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3️⃣ Install dependencies

```bash
pip install fastapi uvicorn pandas google-genai python-dotenv
```

### 4️⃣ Configure Environment Variables

Create a `.env` file in the `backend/` directory:

```env
GEMINI_API_KEY=your_actual_api_key_here
```

### 5️⃣ Prepare Data

- Ensure `vocab.csv` is located in the `backend/` root directory
- Place image assets in:

```
backend/data/images/
```

### 6️⃣ Run the server

```bash
uvicorn app.main:app --reload
```

The backend will be available at:  
**http://127.0.0.1:8000**

---

## 💻 Frontend Setup (React + Vite)

The frontend:

- Provides the experiment UI
- Handles character selection
- Performs input validation

### 1️⃣ Navigate to the frontend folder

```bash
cd frontend
```

### 2️⃣ Install dependencies

```bash
npm install
```

### 3️⃣ Proxy Configuration (Important)

Ensure `vite.config.ts` includes the following proxy setup to avoid CORS issues:

```ts
server: {
  proxy: {
    '/experiment': 'http://127.0.0.1:8000',
    '/images': 'http://127.0.0.1:8000'
  }
}
```

### 4️⃣ Run the development server

```bash
npm run dev
```

The frontend will be available at:  
**http://localhost:5173**

---

## 🧪 Testing the Experiment

1. Open **http://localhost:5173** in your browser
2. Select a **Condition**

### 🅰️ Condition A — Static Feedback

- Tests basic recall
- Immediate **Correct / Incorrect** feedback
- Full answer shown after an incorrect attempt

### 🅱️ Condition B — Adaptive Feedback (AI Companion)

- Uses **Gemini-powered hints**
- Feedback progression:
  - **Subtle hint**
  - **Strong hint**
  - Full answer revealed on the **3rd attempt**

---

## ✅ Key Features to Test

### Noun Capitalization

In Condition B (Learning phase), submit a noun in lowercase.

Example:

```
apfel → ❌
Apfel → ✅
```

The system should prompt grammatical correction.

### Article Selector

Ensure **der / die / das** is selected before submitting a noun.

### Phase Transitions

Completing the required number of tasks should trigger:

- Pre-test → Learning
- Learning → Post-test

---

## 📊 Data Export

To retrieve experiment logs for analysis after a session, use the export endpoint:

```http
GET http://127.0.0.1:8000/experiment/export/{session_id}
```

This returns structured data suitable for statistical analysis and reporting.

---

## 📌 Notes

- Designed for **controlled experiments**
- Supports **between-subject comparisons**
- Logs all attempts, hints, and responses for research evaluation
