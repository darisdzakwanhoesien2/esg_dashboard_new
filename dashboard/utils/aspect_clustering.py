ASPECT_CLUSTER_JSON = {
    "Governance": [
        "governance",
        "corporate governance",
        "board oversight",
        "management oversight"
    ],
    "Emissions": [
        "emissions",
        "carbon",
        "carbon footprint",
        "ghg",
        "greenhouse gas",
        "scope 1",
        "scope 2",
        "scope 3"
    ],
    "Financial Reporting": [
        "financial reporting",
        "financial performance",
        "financial disclosure",
        "financial"
    ],
    "Climate & Energy": [
        "energy",
        "renewable energy",
        "energy efficiency",
        "energy transition"
    ],
    "Community & Social Impact": [
        "stakeholder engagement",
        "community",
        "social impact"
    ],
    "Tax & Compliance": [
        "tax",
        "taxation",
        "compliance"
    ]
}

def cluster_aspect(aspect: str) -> str:
    if not isinstance(aspect, str):
        return "Unclustered"

    a = aspect.lower()
    for cluster, keywords in ASPECT_CLUSTER_JSON.items():
        for kw in keywords:
            if kw in a:
                return cluster
    return "Unclustered"
