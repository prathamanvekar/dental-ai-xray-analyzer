class ClinicalRuleEngine:
    """
    Module 5: Rule-Based Clinical Decision Engine.
    Maps classification results to clinical recommendations.
    """

    def __init__(self):
        # Define treatment rules
        self.rules = {
            "Caries": {
                "general": "Restorative therapy required.",
                "Incisor": "Composite resin restoration recommended for aesthetics.",
                "Molar": "Amalgam or composite restoration. Check for deep decay.",
                "severity_high": "Possible root canal treatment if pulp involved."
            },
            "Periodontal_Bone_Loss": {
                "general": "Periodontal evaluation needed.",
                "severity_low": "Scaling and root planing (SRP).",
                "severity_high": "Surgical intervention or extraction may be necessary."
            },
            "Periapical_Lesion": {
                "general": "Endodontic evaluation required.",
                "treatment": "Root canal treatment (RCT) or apicoectomy."
            },
            "Cyst": {
                "general": "Surgical evaluation required.",
                "treatment": "Enucleation or marsupialization depending on size."
            },
            "Root_Canal_Treated": {
                "general": "Monitor periapical status.",
                "action": "Ensure coronal seal is intact. Crown placement recommended."
            },
            "Impacted_Tooth": {
                "general": "Evaluation for extraction.",
                "Molar": "Third molar (wisdom tooth) surgical extraction if symptomatic.",
                "Canine": "Orthodontic exposure and bonding if salvageable."
            },
            "Healthy": {
                "general": "No active pathology detected.",
                "action": "Maintain regular hygiene and recall appointments."
            }
        }

    def generate_recommendation(self, finding):
        """
        Generate a structured recommendation based on findings.
        Input: finding dict {disease, tooth_type, jaw, confidence}
        """
        disease = finding['disease']
        tooth = finding['tooth_type']
        conf = finding['confidence']

        rec = {
            "diagnosis": disease,
            "treatment": "Evaluation needed",
            "notes": [],
            "urgency": "Routine"
        }

        # Get rules for the disease
        rule = self.rules.get(disease, self.rules["Healthy"])

        # 1. Base Treatment
        if "treatment" in rule:
            rec["treatment"] = rule["treatment"]
        elif "general" in rule:
            rec["treatment"] = rule["general"]

        # 2. Tooth-Specific Rules
        if tooth in rule:
            rec["notes"].append(f"Specific to {tooth}: {rule[tooth]}")

        # 3. Urgency / Severity Logic based on Confidence
        if disease == "Healthy":
            rec["urgency"] = "Routine"
        elif conf > 0.85:
            rec["notes"].append("High confidence detection - Priority evaluation.")
            rec["urgency"] = "High"
            if "severity_high" in rule:
                rec["notes"].append(rule["severity_high"])
        else:
            rec["urgency"] = "Moderate"
            if "severity_low" in rule:
                rec["notes"].append(rule["severity_low"])

        # 4. Jaw Specific Notes
        rec["notes"].append(f"Location: {finding['jaw']} arch.")

        return rec
