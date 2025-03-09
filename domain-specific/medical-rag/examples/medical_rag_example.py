"""
Medical RAG Example

This script demonstrates how to use the Medical RAG implementation for different healthcare scenarios:
1. Medical literature search
2. Clinical decision support
3. Patient education
4. Medical fact checking
"""

import os
import sys

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from medical_rag_pipeline import MedicalRAG


def create_sample_medical_document(filename, content):
    """
    Create a sample medical document for demonstration.
    
    Args:
        filename: The name of the file to create.
        content: The content to write to the file.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        f.write(content)
    print(f"Created sample file: {filename}")


def run_medical_rag_example():
    """
    Run an example of the Medical RAG pipeline.
    """
    # Create a directory for sample medical documents
    sample_dir = "../data/raw/samples"
    os.makedirs(sample_dir, exist_ok=True)
    
    # Create sample medical documents
    create_sample_medical_document(f"{sample_dir}/diabetes_overview.txt", """
# Type 2 Diabetes Mellitus

## Abstract
Type 2 diabetes mellitus (T2DM) is a chronic metabolic disorder characterized by hyperglycemia resulting from insulin resistance and relative insulin deficiency. This review provides an overview of the epidemiology, pathophysiology, clinical presentation, diagnosis, and management of T2DM.

## Introduction
Diabetes mellitus affects over 463 million people worldwide, with T2DM accounting for approximately 90% of all cases. The prevalence of T2DM continues to increase globally, driven by rising rates of obesity, sedentary lifestyles, and population aging.

## Pathophysiology
T2DM is characterized by two primary defects: insulin resistance in peripheral tissues (primarily muscle, liver, and adipose tissue) and progressive beta-cell dysfunction leading to impaired insulin secretion. Additional pathophysiological mechanisms include increased hepatic glucose production, incretin deficiency/resistance, and abnormal fat metabolism.

## Clinical Presentation
Many patients with T2DM are asymptomatic, particularly in the early stages. When present, symptoms may include polyuria, polydipsia, polyphagia, fatigue, blurred vision, slow wound healing, and recurrent infections. T2DM is often diagnosed during routine screening or when patients present with complications.

## Diagnosis
Diagnosis of T2DM is based on one of the following criteria:
- Fasting plasma glucose ≥ 126 mg/dL (7.0 mmol/L)
- 2-hour plasma glucose ≥ 200 mg/dL (11.1 mmol/L) during an oral glucose tolerance test
- HbA1c ≥ 6.5% (48 mmol/mol)
- Random plasma glucose ≥ 200 mg/dL (11.1 mmol/L) in patients with classic symptoms of hyperglycemia

## Management
Management of T2DM involves a comprehensive approach including:

### Lifestyle Modifications
- Medical nutrition therapy
- Regular physical activity
- Weight management
- Smoking cessation
- Limiting alcohol consumption

### Pharmacological Therapy
- Metformin is generally the first-line medication
- Other options include sulfonylureas, thiazolidinediones, DPP-4 inhibitors, SGLT-2 inhibitors, GLP-1 receptor agonists, and insulin

### Monitoring
- Self-monitoring of blood glucose
- Regular HbA1c testing (target generally < 7.0%)
- Regular screening for complications

## Complications
T2DM is associated with various microvascular and macrovascular complications:

### Microvascular
- Diabetic nephropathy
- Diabetic retinopathy
- Diabetic neuropathy

### Macrovascular
- Cardiovascular disease
- Cerebrovascular disease
- Peripheral arterial disease

## Prevention
Prevention strategies for T2DM focus on modifiable risk factors:
- Maintaining a healthy weight
- Regular physical activity
- Healthy diet
- Avoiding tobacco use
- Moderate alcohol consumption

## Conclusion
T2DM is a complex metabolic disorder requiring a multifaceted approach to management. Early diagnosis, comprehensive treatment, and regular monitoring are essential to prevent or delay complications and improve quality of life.

## References
1. American Diabetes Association. Standards of Medical Care in Diabetes-2022. Diabetes Care. 2022;45(Supplement 1).
2. International Diabetes Federation. IDF Diabetes Atlas, 10th edition. 2021.
3. DeFronzo RA, et al. Type 2 diabetes mellitus. Nature Reviews Disease Primers. 2015;1:15019.
""")

    create_sample_medical_document(f"{sample_dir}/hypertension_guidelines.txt", """
# Clinical Practice Guideline for Hypertension Management

## Abstract
This guideline provides evidence-based recommendations for the prevention, detection, evaluation, and management of high blood pressure (BP) in adults.

## Introduction
Hypertension remains a leading cause of cardiovascular morbidity and mortality worldwide. This guideline aims to provide updated, comprehensive recommendations for the management of hypertension based on the latest available evidence.

## Definition and Classification
Hypertension is defined as a systolic BP ≥ 130 mmHg or a diastolic BP ≥ 80 mmHg. BP is categorized as:

- Normal: <120/<80 mmHg
- Elevated: 120-129/<80 mmHg
- Hypertension Stage 1: 130-139/80-89 mmHg
- Hypertension Stage 2: ≥140/≥90 mmHg
- Hypertensive Crisis: >180/>120 mmHg

## BP Measurement
Proper BP measurement techniques are essential for accurate diagnosis and management. Key recommendations include:

- Use validated devices with appropriate cuff size
- Position patient seated, back supported, arm at heart level
- No caffeine, exercise, or smoking within 30 minutes before measurement
- No talking during measurement
- Take at least 2 readings, 1-2 minutes apart
- Use home BP monitoring and ambulatory BP monitoring when appropriate

## Evaluation
Initial evaluation of patients with hypertension should include:

- Complete history and physical examination
- Basic laboratory tests (urinalysis, blood glucose, serum sodium, potassium, calcium, creatinine, lipid profile)
- 12-lead ECG
- Assessment of cardiovascular risk factors and target organ damage
- Screening for secondary causes in patients with clinical suspicions

## Non-pharmacological Interventions
Non-pharmacological interventions are recommended for all patients with hypertension:

- Weight reduction for overweight/obese patients
- DASH-style diet rich in fruits, vegetables, and low-fat dairy products
- Dietary sodium reduction (<2,300 mg/day)
- Physical activity (150 min/week moderate-intensity aerobic exercise)
- Moderation of alcohol consumption
- Adequate potassium intake (3,500-5,000 mg/day)

## Pharmacological Treatment
- Initial therapy should include thiazide diuretics, calcium channel blockers (CCBs), angiotensin-converting enzyme inhibitors (ACEIs), or angiotensin receptor blockers (ARBs)
- Two first-line agents from different classes are recommended for stage 2 hypertension
- Medication selection should consider comorbidities, contraindications, and patient preferences
- Specific recommendations are provided for compelling indications (e.g., coronary artery disease, heart failure, chronic kidney disease, diabetes)

## BP Goals
- General population: <130/80 mmHg
- Older adults (≥65 years): <130/80 mmHg (if tolerated)
- Patients with coronary artery disease: <130/80 mmHg
- Patients with diabetes: <130/80 mmHg
- Patients with chronic kidney disease: <130/80 mmHg
- Patients with heart failure: <130/80 mmHg
- Patients with stroke/TIA: <130/80 mmHg

## Special Populations
Specific considerations for:
- Older adults
- Pregnancy
- Children and adolescents
- Racial/ethnic minorities
- Resistant hypertension
- Secondary hypertension
- Hypertension in women
- Hypertensive urgencies and emergencies

## Follow-up and Monitoring
- Monthly follow-up until BP goal is reached
- Every 3-6 months thereafter
- Monitor for treatment adherence, side effects, and adjust medications as needed
- Periodic laboratory tests to monitor for adverse effects
- Regular assessment of target organ damage

## Implementation Strategies
- Team-based care
- Home BP monitoring
- Telehealth interventions
- Clinical decision support systems
- Performance measures and quality improvement initiatives

## References
1. Whelton PK, et al. 2017 ACC/AHA/AAPA/ABC/ACPM/AGS/APhA/ASH/ASPC/NMA/PCNA Guideline for the Prevention, Detection, Evaluation, and Management of High Blood Pressure in Adults. J Am Coll Cardiol. 2018;71:e127-e248.
2. Williams B, et al. 2018 ESC/ESH Guidelines for the management of arterial hypertension. Eur Heart J. 2018;39:3021-3104.
3. Unger T, et al. 2020 International Society of Hypertension global hypertension practice guidelines. J Hypertens. 2020;38:982-1004.
""")

    create_sample_medical_document(f"{sample_dir}/patient_education_asthma.txt", """
# Asthma: A Patient Education Guide

## What is Asthma?
Asthma is a chronic (long-term) lung disease that inflames and narrows the airways. It causes recurring periods of wheezing, chest tightness, shortness of breath, and coughing. The coughing often occurs at night or early in the morning.

## Who Gets Asthma?
Asthma affects people of all ages, but it most often starts during childhood. In the United States, more than 25 million people are known to have asthma. About 7 million of these people are children.

## What Causes Asthma?
The exact cause of asthma isn't known. Researchers think some genetic and environmental factors interact to cause asthma, most often early in life. These factors include:

- An inherited tendency to develop allergies
- Parents who have asthma
- Certain respiratory infections during childhood
- Contact with airborne allergens or exposure to viral infections in infancy or early childhood when the immune system is developing

## Asthma Triggers
Common asthma triggers include:
- Allergens (such as pollen, pet dander, dust mites, mold)
- Irritants in the air (such as smoke, air pollution, chemical fumes)
- Respiratory infections (such as colds, flu)
- Physical activity (exercise-induced asthma)
- Cold air
- Stress and strong emotions
- Medications (such as aspirin, NSAIDs, beta-blockers)
- Certain food additives (such as sulfites)
- Gastroesophageal reflux disease (GERD)

## Signs and Symptoms
Common signs and symptoms of asthma include:
- Wheezing (a whistling sound when breathing, especially when exhaling)
- Shortness of breath
- Chest tightness or pain
- Coughing, especially at night, during exercise, or when laughing

Not all people with asthma have these symptoms, and having these symptoms doesn't always mean you have asthma. A doctor should diagnose asthma.

## Asthma Diagnosis
To diagnose asthma, your doctor will:
- Ask about your medical history
- Perform a physical exam
- Conduct lung function tests (spirometry)
- May perform additional tests such as allergy testing, chest X-ray, or tests to rule out other conditions

## Asthma Management
Asthma can be effectively managed by:

### Working with Your Healthcare Provider
Develop an asthma action plan with your healthcare provider that includes:
- When and how to take medications
- How to identify and avoid triggers
- How to recognize and handle worsening symptoms
- When to seek emergency care

### Taking Medications as Prescribed
Asthma medications typically include:
- **Quick-relief medications** (rescue medications) to rapidly relieve airway constriction and breathing difficulty during an asthma attack
- **Long-term control medications** (controller medications) to reduce airway inflammation and prevent symptoms

### Monitoring Your Asthma
- Use a peak flow meter to measure how well air moves out of your lungs
- Keep track of your symptoms and triggers
- Have regular check-ups with your healthcare provider

### Avoiding Triggers
- Identify and minimize exposure to your asthma triggers
- Keep your home clean and free from dust, mold, and pet dander
- Avoid smoking and secondhand smoke
- Get vaccinated against influenza and pneumonia

## When to Seek Emergency Care
Seek emergency care if you experience:
- Severe shortness of breath or difficulty breathing
- No improvement after using a rescue inhaler
- Shortness of breath while walking or talking
- Lips or fingernails turning blue
- Severe anxiety due to breathing difficulty

## Living with Asthma
With proper management, most people with asthma can:
- Live normal, active lives
- Participate in exercise and sports (with proper precautions)
- Sleep through the night without symptoms
- Avoid severe asthma attacks
- Have normal or near-normal lung function

Remember, asthma can change over time, so it's important to work with your healthcare provider to adjust your treatment as needed.

## Resources for More Information
- American Lung Association: www.lung.org
- Asthma and Allergy Foundation of America: www.aafa.org
- Centers for Disease Control and Prevention: www.cdc.gov/asthma
""")

    # Initialize the Medical RAG pipeline
    vector_store_path = "../data/processed/medical_vector_store"
    print("Initializing Medical RAG pipeline...")
    medical_rag = MedicalRAG(
        vector_store_path=vector_store_path if os.path.exists(vector_store_path) else None,
        verify_facts=True,
        entity_recognition=True
    )
    
    # Ingest the sample medical documents
    print("\nIngesting sample medical documents...")
    medical_rag.ingest_medical_directory(sample_dir)
    
    # Save the vector store
    print("\nSaving the vector store...")
    medical_rag.save_vector_store(vector_store_path)
    
    # Example scenarios
    scenarios = [
        {
            "title": "Medical Literature Search",
            "query": "What are the current diagnostic criteria for type 2 diabetes?"
        },
        {
            "title": "Clinical Decision Support",
            "query": "What are the recommended first-line medications for treating hypertension?"
        },
        {
            "title": "Patient Education",
            "query": "What should I tell a patient about managing their asthma triggers at home?"
        },
        {
            "title": "Medical Fact Checking",
            "query": "Is it true that all patients with hypertension should have a blood pressure goal of less than 130/80 mmHg?"
        },
        {
            "title": "Complex Medical Query",
            "query": "What is the relationship between diabetes and hypertension, and how does this affect treatment approaches?"
        }
    ]
    
    # Run queries for each scenario
    print("\n=== Running Example Medical Queries ===\n")
    for scenario in scenarios:
        print(f"\n{'='*80}")
        print(f"SCENARIO: {scenario['title']}")
        print(f"{'='*80}")
        
        print(f"Query: {scenario['query']}")
        
        # Run the query
        result = medical_rag.query(scenario['query'])
        
        # Print entity recognition results
        if result["identified_entities"]:
            print("\nIdentified Medical Entities:")
            for entity in result["identified_entities"]:
                print(f"- {entity['text']} (Type: {entity['type']}" + 
                      (f", Ontology ID: {entity['ontology']})" if entity.get('ontology') else ")"))
        
        # Print retrieved documents
        print("\nRetrieved Medical Documents:")
        for i, doc in enumerate(result["retrieved_documents"][:2]):  # Show only the first 2 for brevity
            print(f"\nDocument {i+1} (from {doc.metadata.get('source', 'Unknown')}):")
            if 'section_name' in doc.metadata:
                print(f"Section: {doc.metadata['section_name']}")
            
            # Print a short preview
            preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
            print(f"\n{preview}")
        
        if len(result["retrieved_documents"]) > 2:
            print(f"\n... and {len(result['retrieved_documents']) - 2} more documents")
        
        # Print the response
        print("\nMedical RAG Response:")
        print(result["response"])
        
        # Print citations if available
        if result["citations"]:
            print("\nCitations:")
            for citation in result["citations"]:
                print(f"- {citation}")
        
        print(f"\nConfidence Score: {result['confidence_score']:.2f}")
        print("\n" + "="*80)
    
    print("\nMedical RAG example completed successfully!")


if __name__ == "__main__":
    run_medical_rag_example() 