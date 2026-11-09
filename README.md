# Cat vs Dog Classifier (Streamlit)

## Folder layout
```
app.py
model.py
predictor.py
requirements.txt
.streamlit/config.toml
models/
  best_cnn_model.pth        <- put your trained weights here
  training_history.json     <- included (extracted from checkpoint.pth)
```

## Run
```
pip install -r requirements.txt
streamlit run app.py
```

## Features
- Single image (upload or camera) with confidence level and probability bars
- Grad-CAM heatmap showing where the model looked
- Batch mode with summary, gallery and CSV download
- Model insights: test metrics, training curves, architecture
- Optional test-time augmentation (sidebar)
