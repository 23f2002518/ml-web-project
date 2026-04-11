# Honest Risk Handling

This file gives you exact wording for the uncomfortable questions.

## 1. Custom-CNN Metrics Gap

### Safe Answer

The custom CNN implementation is fully present in the notebook and repo, and it is the from-scratch model in my project. However, in the local archived workspace, the executed metric history was not preserved as cleanly as the EfficientNet and AST runs, so the archival W&B comparison marks it as needing rerun. I prefer to state that honestly instead of inventing numbers.

### If They Push Further

The important point is that the code path and architecture are real, and I can explain the full model, training logic, and why it was included. The archival evidence gap is specifically about preserved local metrics, not about the existence of the model itself.

### Never Say

- `It was not trained at all`
- `I just forgot everything about it`
- any made-up F1 number

## 2. Heavy AI Assistance

### Safe Answer

I used AI heavily as an assistant during implementation and debugging, but I have since reviewed the code and prepared to explain the data flow, architecture choices, training logic, metrics, and deployment clearly.

### Better Follow-Up

I am not hiding the AI assistance. What I want to demonstrate in the viva is that I now understand the pipeline end to end and can defend the major technical choices.

### Never Say

- `AI built everything and I do not know anything`
- `I copied the whole project without understanding it`

## 3. If You Forget An Exact Detail

### Safe Answer

I do not recall the exact line or formula right now, but the idea is...

### Why This Works

It shows honesty and keeps you in control. Many proctors care more about conceptual understanding than perfect recall of every token.

## 4. If Asked Why Only EfficientNet Was Deployed

### Safe Answer

The public deployment is meant to be stable and CPU-friendly, so I used the strongest lightweight single model, which is EfficientNet-B0. The full offline ensemble is better suited to Kaggle scoring than to a simple live Space demo.

## 5. If Asked What You Would Improve Next

### Safe Answer

I would rerun the scratch model with fully preserved tracking, deepen the error analysis with confusion matrices and failure examples, and continue tuning augmentation and ensemble calibration.

## 6. General Tone Rule

- Be honest
- Be calm
- Never volunteer a weakness in a dramatic way
- Frame limitations as `this is what is preserved`, `this is what is implemented`, and `this is what I would improve next`
