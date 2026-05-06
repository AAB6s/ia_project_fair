# Model Card

Pretrained transfer-learning document pipeline for extracting structured context from legal or evidence documents.

## Models
- cnn1_content: labels=injury_photo, scene_damage, screenshot, typed_document, correspondence, handwritten, medical_record, data_chart, signature_stamp, irrelevant; test_macro_f1=0.93751; ece=0.00693; auc=0.99633; decision=logit_bias
- cnn2_evidence: labels=primary_evidence, secondary_evidence; test_macro_f1=0.94814; ece=0.00782; auc=0.99042; decision=binary_threshold
- cnn3_quality: labels=clear, acceptable, degraded; test_macro_f1=0.72497; ece=0.01909; auc=0.87338; decision=logit_bias
- cnn4_tamper: labels=authentic, tampered; test_macro_f1=0.93843; ece=0.01016; auc=0.98382; decision=argmax

## XAI
Grad-CAM, perturbation occlusion, LIME-style segment perturbation, calibration curves, thresholds, confusion matrices, and local examples are saved under results/xai.

## Integration
The exported JSON is intended for backend/RAG/LLM use. The CV models extract evidence structure; the legal RAG system should retrieve law/context and the LLM should generate feedback from the retrieved context plus this JSON.

## Limits
Quality labels are subjective, proxy evidence labels are model-derived, and legal recommendations require RAG grounding and human review.