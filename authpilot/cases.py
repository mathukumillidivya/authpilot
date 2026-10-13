"""Generate prior-authorization test cases with known correct answers.

Each case is a request from a provider plus the patient's chart as simplified FHIR R4
resources. The generator decides the truth first (for every coverage criterion: met, not met,
or not documented) and then writes a chart that supports exactly that truth. The correct
decision follows mechanically:

    any criterion NOT_MET            -> DENY
    otherwise any criterion MISSING  -> NEED_INFO (ask the provider for documentation)
    otherwise                        -> APPROVE

Charts are made realistic on purpose: unrelated history, routine visits, older measurements
that point the other way, evidence written only in a free-text note, values right at a
threshold, and provider summaries that overstate the case.

Usage: python -m authpilot.cases --n-per-policy 60 --seed 7
Writes data/cases.jsonl (one case per line) and data/split.json.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

MET, NOT_MET, MISSING, NA = "met", "not_met", "missing", "not_applicable"

POLICIES = {
    "cpap": {"service": "CPAP device for obstructive sleep apnea", "code": "E0601",
             "criteria": ["cpap_eval", "cpap_test", "cpap_symptoms", "cpap_education"]},
    "oxygen": {"service": "Home oxygen therapy", "code": "E1390",
               "criteria": ["o2_test", "o2_group2", "o2_timing", "o2_tester"]},
    "bariatric": {"service": "Bariatric surgery", "code": None,
                  "criteria": ["bar_bmi", "bar_comorbid", "bar_prior", "bar_procedure"]},
    "icd": {"service": "Implantable cardioverter defibrillator", "code": "33249",
            "criteria": ["icd_indication", "icd_gdmt", "icd_sdm", "icd_exclusions"]},
    "cgm": {"service": "Continuous glucose monitor", "code": "E2103",
            "criteria": ["cgm_dx", "cgm_need", "cgm_visit"]},
}


def decide(statuses: dict) -> str:
    vals = [v for v in statuses.values() if v != NA]
    if NOT_MET in vals:
        return "DENY"
    if MISSING in vals:
        return "NEED_INFO"
    return "APPROVE"


# --------------------------------------------------------------------------------------------
# chart builder

FIRST = ["Mary", "James", "Patricia", "Robert", "Linda", "Michael", "Barbara", "William", "Susan",
         "Richard", "Dorothy", "Joseph", "Carol", "Thomas", "Nancy", "Charles", "Helen", "Daniel",
         "Ruth", "Kenneth", "Gloria", "Raymond", "Rosa", "Walter", "Mei", "Arjun", "Fatima", "Luis"]
LAST = ["Johnson", "Okafor", "Nguyen", "Patel", "Garcia", "Kowalski", "Brown", "Haddad", "Silva",
        "Murphy", "Chen", "Rossi", "Williams", "Kim", "Martinez", "Novak", "Davis", "Ibrahim",
        "Fischer", "Lopez", "Thompson", "Yamamoto", "Clark", "Mensah"]
CLINICIANS = ["Dr. A. Rivera", "Dr. S. Mehta", "Dr. K. O'Neill", "Dr. L. Zhang", "Dr. P. Adeyemi",
              "Dr. R. Castillo", "Dr. J. Holm"]

BACKGROUND = {  # condition text, ICD-10, tags used to keep the background from changing the truth
    "Hyperlipidemia": ("E78.5", {"obesity_comorbid"}),
    "Essential hypertension": ("I10", {"obesity_comorbid", "cpap_comorbid"}),
    "Osteoarthritis of knee": ("M17.11", {"obesity_comorbid"}),
    "Gastroesophageal reflux disease": ("K21.9", set()),
    "Hypothyroidism": ("E03.9", set()),
    "Benign prostatic hyperplasia": ("N40.0", set()),
    "Chronic kidney disease, stage 2": ("N18.2", set()),
    "Allergic rhinitis": ("J30.9", set()),
    "Vitamin D deficiency": ("E55.9", set()),
    "Glaucoma": ("H40.9", set()),
    "Type 2 diabetes mellitus without complications": ("E11.9", {"obesity_comorbid", "diabetes"}),
    "Coronary artery disease": ("I25.10", {"cpap_comorbid", "cad"}),
    "Major depressive disorder, recurrent, mild": ("F33.0", {"cpap_comorbid"}),
}
ROUTINE_MEDS = ["atorvastatin 20 mg daily", "omeprazole 20 mg daily", "levothyroxine 50 mcg daily",
                "vitamin D3 2000 IU daily", "tamsulosin 0.4 mg nightly", "fluticasone nasal spray daily",
                "acetaminophen 500 mg as needed", "latanoprost eye drops nightly"]


ROUTINE_VISITS = [
    ("Primary care follow-up", "Routine follow-up. Feels well overall. BP {bp}/{dia}, HR {hr}. Reviewed medications, no "
                               "changes. Encouraged regular walking. Return in 6 months."),
    ("Primary care follow-up", "Seen for seasonal allergy symptoms, improved with nasal spray. BP {bp}/{dia}. Immunizations "
                               "reviewed and influenza vaccine given."),
    ("Primary care follow-up", "Follow-up of chronic conditions. Labs reviewed and stable. Weight {wt} lb. Discussed diet and "
                               "fall prevention. No new complaints."),
    ("Urgent care visit", "Presented with 3 days of cough and sore throat, no fever. Lungs clear. Viral upper respiratory "
                          "infection; supportive care advised."),
    ("Dermatology visit", "Annual skin check. Two seborrheic keratoses on back, benign appearing. No lesions of concern."),
    ("Ophthalmology visit", "Glaucoma follow-up. Intraocular pressures controlled on current drops. Return in 12 months."),
    ("Podiatry visit", "Nail care and callus debridement. Pedal pulses palpable, sensation intact."),
    ("Annual wellness visit", "Medicare annual wellness visit. Depression screen negative, cognitive screen normal, fall "
                              "risk low. Advance directive discussed. BP {bp}/{dia}."),
    ("Telephone follow-up", "Called to review lab results, all within expected range. No medication changes."),
]
LABS = [("Hemoglobin", 11.8, 15.5, "g/dL", "718-7"), ("White blood cell count", 4.2, 9.8, "10*3/uL", "6690-2"),
        ("Sodium", 136, 144, "mmol/L", "2951-2"), ("Potassium", 3.6, 4.9, "mmol/L", "2823-3"),
        ("eGFR", 58, 92, "mL/min/1.73m2", "33914-3"), ("LDL cholesterol", 62, 128, "mg/dL", "13457-7")]


@dataclass
class Chart:
    rng: random.Random
    order_date: date
    patient_id: str
    resources: list = field(default_factory=list)
    n: int = 0

    def _id(self, prefix: str) -> str:
        self.n += 1
        return f"{prefix}-{self.n:03d}"

    def iso(self, d: date) -> str:
        return d.isoformat()

    def condition(self, text, icd, onset: date, status="active") -> str:
        rid = self._id("cond")
        self.resources.append({"resourceType": "Condition", "id": rid, "subject": self.patient_id,
                               "code": {"text": text, "coding": [{"system": "ICD-10-CM", "code": icd}]},
                               "clinicalStatus": status, "onsetDateTime": self.iso(onset)})
        return rid

    def obs(self, text, value, unit, when: date, loinc=None, performer=None, note=None, method=None) -> str:
        rid = self._id("obs")
        r = {"resourceType": "Observation", "id": rid, "subject": self.patient_id, "status": "final",
             "code": {"text": text, **({"coding": [{"system": "LOINC", "code": loinc}]} if loinc else {})},
             "valueQuantity": {"value": value, "unit": unit}, "effectiveDateTime": self.iso(when)}
        if performer:
            r["performer"] = [{"display": performer}]
        if method:
            r["method"] = {"text": method}
        if note:
            r["note"] = [{"text": note}]
        self.resources.append(r)
        return rid

    def med(self, text, start: date, status="active") -> str:
        rid = self._id("med")
        self.resources.append({"resourceType": "MedicationRequest", "id": rid, "subject": self.patient_id,
                               "medicationCodeableConcept": {"text": text}, "authoredOn": self.iso(start),
                               "status": status})
        return rid

    def procedure(self, text, when: date, cpt=None) -> str:
        rid = self._id("proc")
        self.resources.append({"resourceType": "Procedure", "id": rid, "subject": self.patient_id,
                               "code": {"text": text, **({"coding": [{"system": "CPT", "code": cpt}]} if cpt else {})},
                               "status": "completed", "performedDateTime": self.iso(when)})
        return rid

    def encounter(self, text, when: date, mode="in person", clinician=None) -> str:
        rid = self._id("enc")
        self.resources.append({"resourceType": "Encounter", "id": rid, "subject": self.patient_id, "status": "finished",
                               "class": {"code": {"in person": "AMB", "telehealth": "VR"}[mode], "display": mode},
                               "type": [{"text": text}], "period": {"start": self.iso(when)},
                               "participant": [{"display": clinician or self.rng.choice(CLINICIANS)}]})
        return rid

    def order(self, text, code, when: date, requester) -> str:
        rid = self._id("order")
        self.resources.append({"resourceType": "ServiceRequest", "id": rid, "subject": self.patient_id, "status": "active",
                               "intent": "order", "code": {"text": text, "coding": [{"code": code}]},
                               "authoredOn": self.iso(when), "requester": {"display": requester}})
        return rid

    def note(self, title, when: date, text, author=None) -> str:
        rid = self._id("doc")
        self.resources.append({"resourceType": "DocumentReference", "id": rid, "subject": self.patient_id,
                               "status": "current", "type": {"text": title}, "date": self.iso(when),
                               "author": [{"display": author or self.rng.choice(CLINICIANS)}],
                               "content": [{"attachment": {"contentType": "text/plain", "data": text}}]})
        return rid


def days(n: int) -> timedelta:
    return timedelta(days=n)


def background(c: Chart, exclude_tags: set, bmi_range=(22.0, 32.0), hct_range=(36.0, 47.0), with_bmi=True,
               min_visit_ago=45):
    """Unrelated but realistic history: conditions, medicines, vitals, labs, routine visits."""
    rng, D = c.rng, c.order_date
    pool = [k for k, (_, tags) in BACKGROUND.items() if not (tags & exclude_tags)]
    for name in rng.sample(pool, rng.randint(2, 4)):
        c.condition(name, BACKGROUND[name][0], D - days(rng.randint(400, 4000)))
    for m in rng.sample(ROUTINE_MEDS, rng.randint(2, 4)):
        c.med(m, D - days(rng.randint(120, 2000)))
    height = rng.uniform(1.55, 1.85)
    bmi = rng.uniform(*bmi_range)
    for k in range(rng.randint(2, 4)):
        when = D - days(rng.randint(20, 700))
        c.obs("Blood pressure systolic", rng.randint(118, 138), "mm[Hg]", when, loinc="8480-6")
        if with_bmi:
            c.obs("Body mass index", round(bmi + rng.uniform(-0.8, 0.8), 1), "kg/m2", when, loinc="39156-5")
    c.obs("Hematocrit", round(rng.uniform(*hct_range), 1), "%", D - days(rng.randint(30, 300)), loinc="4544-3")
    c.obs("Creatinine", round(rng.uniform(0.7, 1.2), 2), "mg/dL", D - days(rng.randint(30, 300)), loinc="2160-0")
    visits = sorted(rng.sample(range(min_visit_ago, 1100), rng.randint(5, 11)))
    for k, ago in enumerate(visits):
        when = D - days(ago)
        kind, text = rng.choice(ROUTINE_VISITS)
        c.encounter(kind, when, mode=rng.choice(["in person"] * 4 + ["telehealth"]))
        c.note(f"{kind} note", when, text.format(bp=rng.randint(118, 138), dia=rng.randint(70, 86),
                                                 hr=rng.randint(60, 84), wt=rng.randint(140, 210)))
        c.obs("Heart rate", rng.randint(60, 84), "/min", when, loinc="8867-4")
        if k % 2 == 0:
            for name, lo, hi, unit, loinc in LABS:
                c.obs(name, round(rng.uniform(lo, hi), 1), unit, when - days(rng.randint(0, 6)), loinc=loinc)


# --------------------------------------------------------------------------------------------
# per-policy evidence


def cpap(c: Chart, st: dict, traps: set) -> dict:
    rng, D = c.rng, c.order_date
    ev, facts = {}, {}
    test_day = D - days(rng.randint(10, 50))
    # sleep test
    path_b = st["cpap_symptoms"] != NA
    if st["cpap_test"] == MET:
        if "boundary" in traps:
            ahi, events = (15.0, 31) if not path_b else (5.0, 11)
        else:
            ahi = round(rng.uniform(16, 48), 1) if not path_b else round(rng.uniform(6, 14), 1)
            events = int(ahi * rng.uniform(5.5, 7.0))
    elif st["cpap_test"] == NOT_MET:
        ahi = 4.9 if "boundary" in traps else round(rng.uniform(1.5, 4.5), 1)
        events = int(ahi * 6.5)
    if st["cpap_test"] == MISSING:
        ev["cpap_test"] = [c.note("Sleep medicine note", test_day + days(3),
                                  "Home sleep test kit returned. Data upload failed; study report not yet available. "
                                  "Will repeat or obtain report.")]
    else:
        report = (f"Home sleep apnea test (Type III device), {test_day.isoformat()}. Total recording time 6.8 h. "
                  f"Apnea-hypopnea index {ahi} events/hour ({events} total events). Lowest SpO2 {rng.randint(78, 89)}%.")
        if "note_only" in traps:
            ev["cpap_test"] = [c.note("Sleep study report", test_day, report, author="Regional Sleep Center")]
        else:
            ev["cpap_test"] = [c.obs("Apnea-hypopnea index (home sleep test)", ahi, "events/h", test_day,
                                     performer="Regional Sleep Center", note=f"{events} total respiratory events"),
                               c.note("Sleep study report", test_day, report, author="Regional Sleep Center")]
        facts["ahi"] = ahi
    # evaluation before the test
    if st["cpap_eval"] == MET:
        when = test_day - days(rng.randint(7, 40))
        ev["cpap_eval"] = [c.encounter("Sleep medicine consultation", when),
                           c.note("Sleep medicine consultation", when,
                                  "Seen in clinic for loud snoring and witnessed pauses in breathing reported by spouse. "
                                  "Neck circumference 17 in, Mallampati III. Plan: home sleep apnea test.")]
    elif st["cpap_eval"] == NOT_MET:
        when = test_day + days(rng.randint(2, 8))
        ev["cpap_eval"] = [c.encounter("Sleep medicine consultation", when),
                           c.note("Sleep medicine consultation", when,
                                  "First visit with sleep medicine to review home sleep test that was ordered by "
                                  "urgent care. Snoring and witnessed apneas.")]
    # symptoms / comorbidity (only for AHI 5-14)
    if st["cpap_symptoms"] == MET:
        when = test_day - days(rng.randint(5, 30))
        if rng.random() < 0.5 and "note_only" not in traps:
            ev["cpap_symptoms"] = [c.condition("Essential hypertension", "I10", D - days(rng.randint(400, 3000)))]
        else:
            ev["cpap_symptoms"] = [c.note("Primary care progress note", when,
                                          f"Reports excessive daytime sleepiness, dozing while reading and watching TV. "
                                          f"Epworth Sleepiness Scale {rng.randint(12, 18)}.")]
    elif st["cpap_symptoms"] == NOT_MET:
        when = test_day - days(rng.randint(5, 30))
        ev["cpap_symptoms"] = [c.note("Sleep medicine consultation", when,
                                      "Denies daytime sleepiness (Epworth 4), no insomnia, no cognitive or mood "
                                      "complaints. No history of hypertension, coronary disease or stroke.")]
    # supplier education
    if st["cpap_education"] == MET:
        ev["cpap_education"] = [c.note("DME supplier delivery note", D - days(rng.randint(0, 3)),
                                       "CPAP set up in home. Patient and spouse instructed on proper use, mask fitting, "
                                       "cleaning and care of the equipment. Questions answered.", author="RestWell Medical Supply")]
    facts["path"] = "5-14" if path_b else ">=15"
    return {"evidence": ev, "facts": facts, "exclude_tags": {"cpap_comorbid"} if st["cpap_symptoms"] in (NOT_MET, MISSING) else set(),
            "summary": f"Obstructive sleep apnea confirmed on home sleep test"
                       + (f", AHI {facts['ahi']}" if "ahi" in facts else "") + ". Requesting CPAP.",
            "diagnosis": ("G47.33", "Obstructive sleep apnea")}


def oxygen(c: Chart, st: dict, traps: set) -> dict:
    rng, D = c.rng, c.order_date
    ev, facts = {}, {}
    group2 = st["o2_group2"] != NA
    test_day = D - days(rng.randint(2, 25)) if st["o2_timing"] != NOT_MET else D - days(rng.randint(40, 120))
    performer = "Pulmonary function laboratory, St. Luke's Hospital" if st["o2_tester"] != NOT_MET else "BreatheEasy Home Medical (DME supplier)"
    if st["o2_test"] == MET:
        spo2 = (88 if "boundary" in traps else rng.randint(83, 87)) if not group2 else 89
    else:
        spo2 = 89 if "boundary" in traps else rng.randint(90, 94)
    if st["o2_test"] != MISSING:
        text = f"Resting pulse oximetry on room air: SpO2 {spo2}%. Patient seated, stable, at rest for 10 minutes."
        if "note_only" in traps:
            ev["o2_test"] = [c.note("Oximetry report", test_day, text, author=performer)]
        else:
            ev["o2_test"] = [c.obs("Oxygen saturation at rest, room air (pulse oximetry)", spo2, "%", test_day,
                                   loinc="59408-5", performer=performer, method="room air, at rest")]
        ev["o2_timing"] = list(ev["o2_test"])
        ev["o2_tester"] = list(ev["o2_test"])
        facts["spo2"] = spo2
    else:
        ev["o2_test"] = [c.note("Pulmonology progress note", D - days(2),
                                "Dyspnea on exertion. Oximetry ordered; result pending.")]
    if st["o2_test"] == NOT_MET and facts.get("spo2") == 89:
        # a Group II value: make it unambiguous that no Group II condition is present
        st_note = c.note("Cardiology progress note", D - days(rng.randint(5, 25)),
                         "Echocardiogram: normal right ventricular size and function, estimated pulmonary artery "
                         "pressure normal. No peripheral edema on exam.")
        ev["o2_test"] = ev["o2_test"] + [st_note]
    if st["o2_group2"] == MET:
        kind = rng.choice(["ph", "hct", "edema"])
        if kind == "ph":
            ev["o2_group2"] = [c.condition("Pulmonary hypertension", "I27.20", D - days(rng.randint(200, 1500)))]
        elif kind == "hct":
            ev["o2_group2"] = [c.obs("Hematocrit", round(rng.uniform(57, 60), 1), "%", D - days(rng.randint(5, 20)), loinc="4544-3")]
        else:
            ev["o2_group2"] = [c.note("Cardiology progress note", D - days(rng.randint(5, 25)),
                                      "Exam: 2+ bilateral dependent pitting edema of the lower legs, consistent with "
                                      "congestive heart failure.")]
    elif st["o2_group2"] == NOT_MET:
        ev["o2_group2"] = [c.note("Cardiology progress note", D - days(rng.randint(5, 25)),
                                  "Echocardiogram: normal right ventricular size and function, estimated pulmonary artery "
                                  "pressure normal. No peripheral edema on exam.")]
    c.condition("Chronic obstructive pulmonary disease", "J44.9", D - days(rng.randint(700, 4000)))
    return {"evidence": ev, "facts": facts, "exclude_tags": set(),
            "hct_range": (36.0, 47.0),
            "summary": "COPD with hypoxemia" + (f", resting SpO2 {facts['spo2']}%" if "spo2" in facts else "") + ". Requesting home oxygen.",
            "diagnosis": ("J44.9", "Chronic obstructive pulmonary disease")}


BAR_PROCS = {MET: [("Laparoscopic Roux-en-Y gastric bypass", "43644"), ("Laparoscopic sleeve gastrectomy", "43775"),
                   ("Laparoscopic adjustable gastric banding", "43770")],
             NOT_MET: [("Intragastric balloon placement", "43290"), ("Open sleeve gastrectomy", "43843")]}


def bariatric(c: Chart, st: dict, traps: set) -> dict:
    rng, D = c.rng, c.order_date
    ev, facts = {}, {}
    latest = round(rng.uniform(36, 48), 1) if st["bar_bmi"] == MET else round(rng.uniform(33.0, 34.8), 1)
    if "boundary" in traps:
        latest = 35.0 if st["bar_bmi"] == MET else 34.9
    older = round(rng.uniform(36.5, 42), 1) if st["bar_bmi"] == NOT_MET else round(latest + rng.uniform(-1.5, 1.5), 1)
    c.obs("Body mass index", older, "kg/m2", D - days(rng.randint(250, 500)), loinc="39156-5")
    when = D - days(rng.randint(5, 40))
    if "note_only" in traps:
        ev["bar_bmi"] = [c.note("Bariatric surgery consultation", when,
                                f"Weight and height measured today; BMI {latest} kg/m2.")]
    else:
        ev["bar_bmi"] = [c.obs("Body mass index", latest, "kg/m2", when, loinc="39156-5")]
    facts["bmi"] = latest
    if st["bar_comorbid"] == MET:
        name = rng.choice(["Type 2 diabetes mellitus without complications", "Essential hypertension", "Obstructive sleep apnea"])
        icd = {"Obstructive sleep apnea": "G47.33"}.get(name, BACKGROUND.get(name, ("", 0))[0])
        if "note_only" in traps:
            ev["bar_comorbid"] = [c.note("Primary care progress note", D - days(rng.randint(30, 200)),
                                         f"Assessment includes {name.lower()}, followed in clinic.")]
        else:
            ev["bar_comorbid"] = [c.condition(name, icd, D - days(rng.randint(300, 3000)))]
    elif st["bar_comorbid"] == NOT_MET:
        ev["bar_comorbid"] = [c.note("Bariatric surgery consultation", D - days(rng.randint(5, 40)),
                                     "No obesity-related comorbidities identified: blood pressure normal, HbA1c 5.3%, "
                                     "sleep study negative for apnea, lipids normal, no joint disease.")]
    if st["bar_prior"] == MET:
        ev["bar_prior"] = [c.note("Medical weight management summary", D - days(rng.randint(60, 300)),
                                  "Completed a 6-month physician-supervised weight management program with dietitian "
                                  "visits and a trial of phentermine. Net weight loss 2 kg, regained after stopping. "
                                  "Considered unsuccessful.")]
    proc = rng.choice(BAR_PROCS[st["bar_procedure"]])
    facts["procedure"] = proc
    return {"evidence": ev, "facts": facts, "service": proc[0], "code": proc[1],
            "exclude_tags": {"obesity_comorbid"} if st["bar_comorbid"] != MET else set(),
            "summary": f"Morbid obesity, BMI {facts['bmi']}. Requesting {proc[0].lower()}.",
            "diagnosis": ("E66.01", "Morbid (severe) obesity due to excess calories")}


def icd(c: Chart, st: dict, traps: set) -> dict:
    rng, D = c.rng, c.order_date
    ev, facts = {}, {}
    nonisch = st["icd_gdmt"] != NA
    kind = "Non-ischemic dilated cardiomyopathy" if nonisch else "Ischemic cardiomyopathy"
    cond = c.condition(kind, "I42.0" if nonisch else "I25.5", D - days(rng.randint(150, 1500)))
    if st["icd_indication"] == MET:
        lvef = 35 if "boundary" in traps else rng.randint(20, 33)
        nyha = rng.choice(["II", "III"])
        older = rng.randint(38, 50)
    else:
        lvef = 36 if "boundary" in traps else rng.randint(38, 48)
        nyha = rng.choice(["II", "III"])
        older = rng.randint(22, 32)       # an older, qualifying value that no longer applies
    c.obs("Left ventricular ejection fraction (echocardiogram)", older, "%", D - days(rng.randint(200, 500)), loinc="10230-1")
    echo_day = D - days(rng.randint(10, 60))
    nyha_note = c.note("Cardiology progress note", D - days(rng.randint(3, 20)),
                       f"Heart failure, NYHA class {nyha}. Euvolemic on exam. Continue current regimen.")
    if "note_only" in traps:
        ev["icd_indication"] = [cond, c.note("Echocardiogram report", echo_day, f"Left ventricular ejection fraction {lvef}% by biplane Simpson."), nyha_note]
    else:
        ev["icd_indication"] = [cond, c.obs("Left ventricular ejection fraction (echocardiogram)", lvef, "%", echo_day, loinc="10230-1"), nyha_note]
    facts["lvef"], facts["nyha"] = lvef, nyha
    if nonisch:
        start = D - days(rng.randint(100, 400)) if st["icd_gdmt"] == MET else D - days(rng.randint(30, 75))
        ev["icd_gdmt"] = [c.med("sacubitril/valsartan 49/51 mg twice daily", start),
                          c.med("carvedilol 12.5 mg twice daily", start + days(rng.randint(0, 10))),
                          c.med("spironolactone 25 mg daily", start + days(rng.randint(0, 10)))]
    if st["icd_sdm"] == MET:
        when = D - days(rng.randint(3, 30))
        ev["icd_sdm"] = [c.encounter("Shared decision-making visit for ICD", when),
                         c.note("Shared decision-making note", when,
                                "Shared decision-making encounter for primary prevention ICD using the CardioSmart ICD "
                                "decision aid. Risks, benefits and alternatives discussed; patient elects implantation.")]
    if st["icd_exclusions"] == MET:
        ev["icd_exclusions"] = [nyha_note]
    else:
        kind_ex = rng.choice(["mi", "pci", "cabg"])
        if kind_ex == "mi":
            ev["icd_exclusions"] = [c.condition("Acute non-ST elevation myocardial infarction", "I21.4", D - days(rng.randint(10, 38)))]
        elif kind_ex == "pci":
            ev["icd_exclusions"] = [c.procedure("Percutaneous coronary intervention with drug-eluting stent", D - days(rng.randint(20, 85)), "92928")]
        else:
            ev["icd_exclusions"] = [c.procedure("Coronary artery bypass graft, 3 vessels", D - days(rng.randint(20, 85)), "33533")]
    return {"evidence": ev, "facts": facts, "exclude_tags": {"cad"} if nonisch else set(),
            "summary": f"{kind}, NYHA class {nyha}, LVEF {facts['lvef'] if 'claim' not in traps else 30}%. "
                       "Requesting primary prevention ICD.",
            "diagnosis": ("I42.0" if nonisch else "I25.5", kind)}


def cgm(c: Chart, st: dict, traps: set) -> dict:
    rng, D = c.rng, c.order_date
    ev, facts = {}, {}
    if st["cgm_dx"] == MET:
        name, code = rng.choice([("Type 2 diabetes mellitus with hyperglycemia", "E11.65"), ("Type 1 diabetes mellitus", "E10.9")])
        ev["cgm_dx"] = [c.condition(name, code, D - days(rng.randint(500, 5000)))]
    else:
        ev["cgm_dx"] = [c.condition("Prediabetes", "R73.03", D - days(rng.randint(300, 2000))),
                        c.obs("Hemoglobin A1c", round(rng.uniform(5.8, 6.3), 1), "%", D - days(rng.randint(20, 90)), loinc="4548-4")]
    if st["cgm_need"] == MET:
        if rng.random() < 0.6:
            ev["cgm_need"] = [c.med("insulin glargine 22 units at bedtime", D - days(rng.randint(100, 900)))]
            facts["need"] = "insulin"
        else:
            notes = []
            for k in range(3):
                notes.append(c.obs("Glucose, capillary", rng.randint(41, 52), "mg/dL", D - days(rng.randint(5, 80)), loinc="2339-0"))
            notes.append(c.note("Endocrinology progress note", D - days(rng.randint(20, 60)),
                                "Recurrent hypoglycemia below 54 mg/dL despite two adjustments: glipizide dose reduced in "
                                "March, then stopped and replaced with sitagliptin in May. Lows persist."))
            ev["cgm_need"] = notes
            facts["need"] = "hypoglycemia"
            c.med("metformin 1000 mg twice daily", D - days(rng.randint(300, 1500)))
    else:
        c.med("metformin 1000 mg twice daily", D - days(rng.randint(300, 1500)))
        lows = [c.obs("Glucose, capillary", rng.randint(55, 62) if "boundary" in traps else rng.randint(85, 140), "mg/dL",
                      D - days(rng.randint(5, 80)), loinc="2339-0") for _ in range(2)]
        ev["cgm_need"] = lows + [c.note("Primary care progress note", D - days(rng.randint(20, 60)),
                                        "Diabetes managed with metformin alone, no insulin. "
                                        + ("Occasional low readings in the mid 50s to low 60s, asymptomatic."
                                           if "boundary" in traps else "No hypoglycemic episodes."))]
        facts["need"] = "none"
    if st["cgm_visit"] == MET:
        when = D - days(rng.randint(5, 170))
        mode = rng.choice(["in person", "telehealth"])
        ev["cgm_visit"] = [c.encounter("Diabetes follow-up visit", when, mode=mode),
                           c.note("Diabetes follow-up note", when, "Reviewed glucose log and HbA1c, assessed diabetes control and "
                                  "hypoglycemia risk. Discussed continuous glucose monitoring.")]
    elif st["cgm_visit"] == NOT_MET:
        when = D - days(rng.randint(200, 330))
        ev["cgm_visit"] = [c.encounter("Diabetes follow-up visit", when),
                           c.note("Diabetes follow-up note", when, "Reviewed glucose log and HbA1c. Stable.")]
    return {"evidence": ev, "facts": facts, "exclude_tags": {"diabetes"},
            "summary": "Diabetes" + (" treated with insulin" if facts.get("need") == "insulin" or "claim" in traps else "")
                       + ". Requesting a continuous glucose monitor.",
            "diagnosis": ("E11.65", "Type 2 diabetes mellitus with hyperglycemia") if st["cgm_dx"] == MET else ("R73.03", "Prediabetes")}


BUILDERS = {"cpap": cpap, "oxygen": oxygen, "bariatric": bariatric, "icd": icd, "cgm": cgm}

# Scenario menus: which criteria can be NOT_MET or MISSING for each policy, and conditional criteria.
CAN_FAIL = {
    "cpap": {"cpap_eval": [NOT_MET, MISSING], "cpap_test": [NOT_MET, MISSING], "cpap_symptoms": [NOT_MET, MISSING], "cpap_education": [MISSING]},
    "oxygen": {"o2_test": [NOT_MET, MISSING], "o2_group2": [NOT_MET], "o2_timing": [NOT_MET], "o2_tester": [NOT_MET]},
    "bariatric": {"bar_bmi": [NOT_MET], "bar_comorbid": [NOT_MET], "bar_prior": [MISSING], "bar_procedure": [NOT_MET]},
    "icd": {"icd_indication": [NOT_MET], "icd_gdmt": [NOT_MET], "icd_sdm": [MISSING], "icd_exclusions": [NOT_MET]},
    "cgm": {"cgm_dx": [NOT_MET], "cgm_need": [NOT_MET], "cgm_visit": [NOT_MET, MISSING]},
}


def statuses_for(policy: str, rng: random.Random, target: str) -> dict:
    crit = POLICIES[policy]["criteria"]
    st = {k: MET for k in crit}
    # conditional criteria
    if policy == "cpap" and rng.random() < 0.55:
        st["cpap_symptoms"] = NA
    if policy == "oxygen" and rng.random() < 0.6:
        st["o2_group2"] = NA
    if policy == "icd" and rng.random() < 0.5:
        st["icd_gdmt"] = NA
    options = [(k, v) for k, vs in CAN_FAIL[policy].items() for v in vs
               if st[k] != NA and (target == "DENY" and v == NOT_MET or target == "NEED_INFO" and v == MISSING)]
    if target != "APPROVE":
        if not options:
            return statuses_for(policy, rng, "DENY")
        k, v = rng.choice(options)
        st[k] = v
        if policy == "cpap" and k == "cpap_test" and v != MET:
            st["cpap_symptoms"] = NA if v == MISSING else st["cpap_symptoms"]
        if policy == "oxygen" and k == "o2_test" and v == MISSING:
            st["o2_timing"] = st["o2_tester"] = NA
            st["o2_group2"] = NA
        if policy == "oxygen" and k == "o2_test" and v == NOT_MET:
            st["o2_group2"] = NA
        if policy == "oxygen" and k == "o2_group2":
            pass
    return st


def make_case(idx: int, policy: str, target: str, rng: random.Random) -> dict:
    st = statuses_for(policy, rng, target)
    traps = set()
    if rng.random() < 0.3:
        traps.add("note_only")
    if rng.random() < 0.25:
        traps.add("boundary")
    if decide(st) == "DENY" and rng.random() < 0.35:
        traps.add("claim")
    order_date = date(2025, 3, 1) + days(rng.randint(0, 120))
    pid = f"pt-{idx:04d}"
    c = Chart(rng=rng, order_date=order_date, patient_id=pid)
    built = BUILDERS[policy](c, st, traps)
    background(c, built.get("exclude_tags", set()), bmi_range=built.get("bmi_range", (22.0, 32.0)),
               hct_range=built.get("hct_range", (36.0, 47.0)), with_bmi=policy != "bariatric",
               min_visit_ago=200 if policy == "cgm" and st.get("cgm_visit") != MET else 45)
    requester = rng.choice(CLINICIANS)
    service = built.get("service", POLICIES[policy]["service"])
    code = built.get("code") or POLICIES[policy]["code"]
    order_id = c.order(service, code, order_date, requester)
    if policy == "bariatric":
        built["evidence"]["bar_procedure"] = [order_id]
    rng.shuffle(c.resources)
    name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    birth = date(rng.randint(1938, 1959), rng.randint(1, 12), rng.randint(1, 28))
    patient = {"resourceType": "Patient", "id": pid, "name": [{"text": name}], "birthDate": birth.isoformat(),
               "gender": rng.choice(["female", "male"])}
    summary = built["summary"]
    if "claim" in traps:
        summary += " Patient meets all coverage criteria."
    request = {
        "request_id": f"PA-{20250000 + idx}", "patient_id": pid, "patient_name": name, "birth_date": birth.isoformat(),
        "service": service, "code": code,
        "diagnosis": {"code": built["diagnosis"][0], "text": built["diagnosis"][1]},
        "order_date": order_date.isoformat(), "ordering_provider": requester,
        "clinical_summary": summary,
    }
    truth = {"decision": decide(st), "policy": policy, "statuses": st,
             "evidence": {k: v for k, v in built["evidence"].items() if st.get(k) not in (NA,)},
             "traps": sorted(traps)}
    return {"case_id": f"case-{idx:04d}", "request": request, "chart": [patient] + c.resources, "truth": truth}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-policy", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dev-per-policy", type=int, default=20)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    cases, split = [], {"dev": [], "test": []}
    idx = 0
    for policy in POLICIES:
        targets = (["APPROVE"] * 40 + ["DENY"] * 35 + ["NEED_INFO"] * 25)
        for k in range(args.n_per_policy):
            idx += 1
            case = make_case(idx, policy, rng.choice(targets), rng)
            cases.append(case)
            split["dev" if k < args.dev_per_policy else "test"].append(case["case_id"])
    out = Path("data")
    out.mkdir(exist_ok=True)
    with open(out / "cases.jsonl", "w") as f:
        for case in cases:
            f.write(json.dumps(case) + "\n")
    (out / "split.json").write_text(json.dumps(split, indent=1))
    from collections import Counter
    print(len(cases), "cases;", Counter(c["truth"]["decision"] for c in cases),
          Counter((c["truth"]["policy"], c["truth"]["decision"]) for c in cases))
    print("traps:", Counter(t for c in cases for t in c["truth"]["traps"]))
    print("mean resources per chart:", sum(len(c["chart"]) for c in cases) / len(cases))


if __name__ == "__main__":
    main()
