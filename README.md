# Financial Advisor RAG System with EU AI Act Evaluation

Ein vereinfachtes RAG-System für Finanzberatung mit automatischer EU AI Act Compliance-Evaluation.

## 🚀 Schnellstart

### 1. Installation

```bash
# Dependencies installieren
pip install -r requirements.txt

# Environment-Variablen einrichten
cp env_template.txt .env
# Bearbeite .env und füge deinen OpenAI API Key hinzu
```

### 2. Vorbereitung

Stelle sicher, dass folgende Dateien/Ordner vorhanden sind:

- `financial_data/` - Ordner mit PDF-Dokumenten
- `catalog.csv` - Datei mit Fragen (Spalten: Question-ID, Category, Question)
- `.env` - Environment-Variablen (siehe env_template.txt)

### 3. Ausführung

```bash
python financial_advisor_rag.py
```

## 📁 Projektstruktur

```
llm-evaluation-framework/
├── financial_advisor_rag.py      # Haupt-RAG-System
├── requirements.txt              # Python-Dependencies
├── env_template.txt              # Environment-Template
├── catalog.csv                   # Fragen-Katalog
├── financial_data/               # PDF-Dokumente
│   ├── skript_grundz__ge_der_portefeuilletheorie.pdf
│   ├── skriptkapitalmarkttheorie.pdf
│   └── ... (weitere PDFs)
└── evaluation_results.txt        # Ergebnisse (wird erstellt)
```

## 🔧 Konfiguration

### Environment-Variablen (.env)

```bash
OPENAI_API_KEY=your_openai_api_key_here
USER_AGENT=LLM-Evaluation-Framework/1.0
OPENAI_MODEL=gpt-4o-mini
MAX_QUESTIONS=8
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

### catalog.csv Format

```csv
Question-ID,Category,Question
1,Portfolio Advice,"Advise on investing 10,000 euros..."
2,Risk Management,"Analyze the potential impact..."
...
```

## 📊 Evaluation

Das System evaluiert automatisch:

### RAGAS Metriken
- **ContextRelevancy**: Relevanz der gefundenen Dokumente
- **AnswerRelevancy**: Relevanz der Antworten
- **AnswerCorrectness**: Richtigkeit der Antworten
- **AnswerFaithfulness**: Basiert Antwort auf Dokumenten

### EU AI Act Compliance
- **Transparency**: AI-Identifikation, Disclaimer
- **Fairness**: Unvoreingenommene Behandlung
- **Safety**: Risikobewertung, Sicherheitsmaßnahmen
- **Privacy**: PII-Schutz, Datenschutz
- **Accountability**: Verantwortlichkeit, Compliance

## 📈 Ergebnisse

Die Evaluation wird in `evaluation_results.txt` gespeichert:

```
FINANCIAL ADVISOR RAG EVALUATION
========================================

RAGAS METRICS:
   context_relevancy: 0.850
   answer_relevancy: 0.920
   answer_correctness: 0.880
   answer_faithfulness: 0.910

EU AI ACT COMPLIANCE:
   transparency: 0.850
   fairness: 0.780
   safety: 0.920
   privacy: 0.880
   accountability: 0.750

Overall Compliance: 0.836
```

## 🎯 Features

- ✅ **Einfache Bedienung**: Ein Befehl startet alles
- ✅ **EU AI Act Compliance**: Automatische Evaluation
- ✅ **PDF-Verarbeitung**: Lädt alle PDFs aus financial_data/
- ✅ **Fragen-Katalog**: Verwendet catalog.csv für Evaluation
- ✅ **Detaillierte Ergebnisse**: RAGAS + EU AI Act Metriken
- ✅ **Compliance-Status**: ✅⚠️❌ Anzeige

## 🔒 Compliance

Das System ist konform mit:
- **EU AI Act** (Verordnung EU 2024/1689)
- **Banking Regulations**
- **Privacy Requirements**

## 🛠️ Troubleshooting

### Häufige Probleme

1. **OpenAI API Key fehlt**
   ```bash
   # Prüfe .env Datei
   cat .env
   ```

2. **PDFs werden nicht geladen**
   ```bash
   # Prüfe financial_data/ Ordner
   ls financial_data/
   ```

3. **catalog.csv nicht gefunden**
   ```bash
   # Prüfe ob Datei existiert
   ls catalog.csv
   ```

### Logs

Das System zeigt detaillierte Logs:
- ✅ Erfolgreiche Operationen
- ❌ Fehler mit Details
- 📊 Evaluation-Fortschritt

## 📝 Lizenz

Dieses Projekt ist Teil des LLM Evaluation Framework.




