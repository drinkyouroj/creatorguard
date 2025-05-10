# CreatorGuard 🛡️
*A private, emotionally-intelligent comment moderation and content insight system for creators.*

CreatorGuard is a lightweight, local-first tool designed to help creators process, moderate, and respond to comment streams — especially on YouTube — without sacrificing emotional well-being. It can also track personal reflections and content development workflows, making it a full emotional OS for creators.

---

## 🎯 Goals

- Protect creators' mental health by filtering harmful or overwhelming comments
- Interpret comment sentiment using GPT-4-turbo
- Suggest emotionally-safe responses
- Track emotional state and creator reflections alongside comment engagement
- Log all moderation decisions and emotional insights in a searchable SQLite database

---

## 🔧 Features (MVP)

- 🧠 GPT-powered comment classification (`supportive`, `toxic`, `off-topic`, etc.)
- 🚫 Moderation tagging (`hide`, `respond`, `flag`, `ignore`)
- 💬 Suggested replies in your tone
- 📊 Summary stats on audience tone
- 📓 Optional journaling/emotional log integration

---

## 🗃️ Folder Structure

```
creatorguard/
├── db/
│   ├── schema.sql           # SQLite schema definition
│   └── init_db.py           # DB initializer script
├── comments/                # Raw or cleaned comment dumps
├── gpt/                     # GPT logic for interpretation & reply generation
├── personal/                # Optional: personal mood/journal logs
├── creatorguard.db          # SQLite database (auto-created)
├── .env.example             # Template for API keys
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the Repo
```bash
git clone https://github.com/yourusername/creatorguard.git
cd creatorguard
```

### 2. Create a Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Your Environment
Copy `.env.example` to `.env` and fill in your API keys:
```bash
cp .env.example .env
```

You’ll need:
- OpenAI API key
- (Coming soon) YouTube Data API key

### 5. Initialize the Database
```bash
python db/init_db.py
```

---

## 📍 Upcoming Features

- ✅ GPT-powered comment moderation
- 🧠 Personalized tone engine using your own writing
- 💬 Live chatbot mod (Phase 2)
- ✍️ Personal journal and mood tracking (Phase 2)
- 📚 Content idea log and reflection system

---

## 🔐 Privacy First

This system is built to run locally, respecting your emotional boundaries and keeping your data private unless explicitly shared or deployed.

---

## 🤝 License

MIT License — open to contribution and adaptation for other creators.

---

## 🙌 Acknowledgements

Inspired by the need to stay emotionally grounded in an overwhelming digital world.  
Built using [OpenAI](https://openai.com/) and [SQLite](https://www.sqlite.org/).
