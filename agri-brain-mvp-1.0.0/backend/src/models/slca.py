def slca_score(carbon_kg, fairness=0.82,resilience=0.78,transparency=0.9,weights=None):
    w=weights or {'carbon':0.35,'fairness':0.25,'resilience':0.20,'transparency':0.20}
    carbon_score=max(0.0,1.0-carbon_kg/80.0)
    return float(max(0.0,min(1.0,w['carbon']*carbon_score+w['fairness']*fairness+w['resilience']*resilience+w['transparency']*transparency)))
