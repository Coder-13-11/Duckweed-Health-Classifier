# Duckweed Copper Analyzer

A computer vision biosensor that estimates copper contamination in water using *Lemna minor* (duckweed) as a bioindicator.

---

## Overview

This project provides a low-cost alternative to traditional spectrophotometry by analyzing duckweed color and coverage changes caused by copper stress.

**Traditional Method**
- $10,000+ spectrophotometer  
- Laboratory required  
- 10–15 minute analysis  

**This System**
- Smartphone camera  
- Field-deployable  
- User-friendly  
- Instant results (<5 seconds)  

---

## How It Works

1. Crop image to petri dish region  
2. Detect dark green pixels using RGB thresholds  
3. Calculate frond coverage percentage  
4. Map coverage to copper concentration using a calibrated curve  

**Dark Green Criteria**
- G > 100  
- G > R  
- G > B + 30  
- Brightness < 160  

**Calibration (Semi-Quantitative)**  
- 38%+ coverage → 0–3 mg/L  
- 22–38% → 3–7 mg/L  
- 18–22% → 7–10 mg/L  
- <18% → 10+ mg/L  

Best suited for detecting presence of stress and identifying controls.

---

## Installation

**Requirements**
- Python 3.8+
- pip

Install dependencies:

```bash
pip install streamlit pillow opencv-python-headless numpy pandas

**Run the app:**

streamlit run app_ULTRA_SIMPLE.py

The app will open at:

**http://localhost:8501**

## Acknowledgments

This project was developed with mentorship from:
- **Dr. Ananda Bhattacharjee**, University of South Florida
- **Dr. Sarina Ergas**, University of South Florida

---

**Note**: This is a student research project developed for the Intel International Science and Engineering Fair 2026. While functional and scientifically grounded, it represents early-stage research and should be validated further before widespread adoption.
