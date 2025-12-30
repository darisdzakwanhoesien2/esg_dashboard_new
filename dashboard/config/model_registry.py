MODEL_REGISTRY = {
    "ClimateBERT": {
        "econbert": {
            "id": "climatebert/econbert",
            "requires_auth": True
        },
        "netzero": {
            "id": "climatebert/netzero-reduction",
            "requires_auth": True
        },
        "transition_physical": {
            "id": "climatebert/transition-physical",
            "requires_auth": True
        }
    },
    "ClimateBERT (DistilRoBERTa)": {
        "detector": {
            "id": "climatebert/distilroberta-base-climate-detector",
            "requires_auth": True
        },
        "commitment": {
            "id": "climatebert/distilroberta-base-climate-commitment",
            "requires_auth": True
        },
        "sentiment": {
            "id": "climatebert/distilroberta-base-climate-sentiment",
            "requires_auth": True
        }
    },
    "ESG Scoring": {
        "gri": {
            "id": "nlp-esg-scoring/bert-base-finetuned-esg-gri-clean",
            "requires_auth": True
        },
        "tcfd": {
            "id": "nlp-esg-scoring/bert-base-finetuned-esg-TCFD-clean",
            "requires_auth": True
        }
    },
    "ESGBERT": {
        "environmental": {
            "id": "ESGBERT/EnvironmentalBERT-base",
            "requires_auth": True
        },
        "social": {
            "id": "ESGBERT/SocialBERT-base",
            "requires_auth": True
        },
        "governance": {
            "id": "ESGBERT/GovernanceBERT-base",
            "requires_auth": True
        }
    }
}
