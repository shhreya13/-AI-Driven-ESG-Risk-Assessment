
risk_terms = {
    "Environmental": [
        "emissions", "carbon", "pollution", "waste", 
        "water scarcity", "deforestation", "climate change",
        "greenhouse gases", "methane", "oil spill", 
        "hazardous chemicals", "renewable energy gap"
    ],
    "Social": [
        "labor", "child labor", "forced labor", "diversity", 
        "community", "human rights", "workplace safety", 
        "harassment", "pay gap", "gender inequality", "supply chain abuse"
    ],
    "Governance": [
        "fine", "lawsuit", "non-compliance", "data breach", 
        "fraud", "executive pay", "cybersecurity", 
        "corruption", "bribery", "tax evasion", 
        "insider trading", "board independence"
    ]
}

# Assign weights (impact factor)
weight_map = {
    # Environmental
    "emissions": 2, "carbon": 2, "pollution": 1, "waste": 1,
    "water scarcity": 3, "climate change": 3, "deforestation": 3,
    "greenhouse gases": 2, "methane": 2, "oil spill": 3, 
    "hazardous chemicals": 3, "renewable energy gap": 2,

    # Social
    "labor": 1, "child labor": 4, "forced labor": 4, "diversity": 2,
    "community": 1, "human rights": 3, "workplace safety": 2,
    "harassment": 3, "pay gap": 2, "gender inequality": 2,
    "supply chain abuse": 3,

    # Governance
    "lawsuit": 3, "fine": 2, "non-compliance": 2, "data breach": 3,
    "fraud": 3, "executive pay": 1, "cybersecurity": 3,
    "corruption": 4, "bribery": 4, "tax evasion": 4,
    "insider trading": 4, "board independence": 2
}
